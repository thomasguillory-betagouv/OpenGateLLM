"""Benchmark of the batch mode proof of concept (see adr/2026-11-11-batch-mode.md).

Question: does a bulk job hurt interactive traffic, and does the batch mode prevent it?

A mock provider stands for a GPU server: it holds `--slots` requests at a time, each takes `--service-time` seconds,
and the next ones wait in its queue. OpenGateLLM fronts it with `qos_limit = --slots`. Interactive traffic arrives at
`--rate` requests per second (Poisson) while, depending on the scenario, nothing else runs, a client sends a bulk job
through POST /v1/chat/completions as one would today, or the same job is submitted as a batch to the batch worker.

Prerequisites: a Postgres and a Redis reachable with the URLs below (the script drops and recreates the database
`ogl_batch_benchmark` and flushes the Redis database), and the API dependencies installed (uv sync --extra api).

    python adr/scripts/2026-11-11-batch-mode-benchmark.py --output benchmark.json

`--fake-tokenizer` replaces the tiktoken encoding with a length-based count, for machines that cannot download it.
"""

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time

import httpx

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "bench-model"
DATABASE_NAME = "ogl_batch_benchmark"


# --------------------------------------------------------------------------- processes


def run_mock_provider(port: int, slots: int, service_time: float) -> None:
    from fastapi import FastAPI
    import uvicorn

    app = FastAPI()
    gpu = asyncio.Semaphore(slots)
    stats = {"requests": 0, "in_flight": 0, "max_in_flight": 0}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "created": 0, "owned_by": "benchmark", "max_model_len": 8192}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict):
        stats["in_flight"] += 1
        stats["max_in_flight"] = max(stats["max_in_flight"], stats["in_flight"])
        try:
            async with gpu:
                await asyncio.sleep(service_time)
        finally:
            stats["in_flight"] -= 1
        stats["requests"] += 1
        return {
            "id": f"chatcmpl-{stats['requests']}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", MODEL_NAME),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Lorem ipsum " * 25}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 50, "total_tokens": 70},
        }

    @app.post("/stats/reset")
    async def reset_stats():
        stats.update(requests=0, max_in_flight=stats["in_flight"])
        return stats

    @app.get("/stats")
    async def get_stats():
        return stats

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def _patch_tokenizer(module) -> None:
    class LengthTokenizer:
        def encode(self, text: str) -> list[int]:
            return [0] * max(1, len(text) // 4)

    module.initialize_tokenizer = lambda configuration: LengthTokenizer()


def run_api(port: int, fake_tokenizer: bool) -> None:
    import uvicorn

    import api.lifespan

    if fake_tokenizer:
        _patch_tokenizer(api.lifespan)

    from api.app import create_app

    uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="warning")


def run_worker(fake_tokenizer: bool) -> None:
    import api.worker

    if fake_tokenizer:
        _patch_tokenizer(api.worker)

    asyncio.run(api.worker.main())


# --------------------------------------------------------------------------- setup


def write_configuration(args: argparse.Namespace, path: Path) -> None:
    postgres_url = f"{args.postgres_url.rstrip('/')}/{DATABASE_NAME}"
    path.write_text(
        json.dumps(
            {
                "dependencies": {
                    "postgres": {"url": postgres_url, "pool_size": 20},
                    "redis": {"url": args.redis_url},
                },
                "settings": {
                    "auth_secret_key": "benchmark-secret-key-benchmark-secret-key",
                    "auth_bootsrap_admin_username": "admin",
                    "auth_bootsrap_admin_password": "changeme",
                    "monitoring_prometheus_enabled": False,
                    "log_level": "WARNING",
                    "batch_worker_concurrency": args.worker_concurrency,
                    "batch_provider_capacity_share": args.capacity_share,
                },
                "models": [
                    {
                        "name": MODEL_NAME,
                        "type": "text-generation",
                        "qos_retries_before_reject": args.qos_retries,
                        "providers": [
                            {
                                "type": "vllm",
                                "url": f"http://127.0.0.1:{args.mock_port}",
                                "model_name": MODEL_NAME,
                                "qos_limit": args.slots,
                                "timeout": 120,
                            }
                        ],
                    }
                ],
            }
        )
    )


def reset_dependencies(args: argparse.Namespace, environment: dict) -> None:
    from redis import Redis
    from sqlalchemy import create_engine, make_url, text

    url = make_url(args.postgres_url.rstrip("/") + "/postgres").set(drivername="postgresql+psycopg2")
    engine = create_engine(url, isolation_level="AUTOCOMMIT")
    with engine.connect() as connection:
        connection.execute(text(f'DROP DATABASE IF EXISTS "{DATABASE_NAME}" WITH (FORCE)'))
        connection.execute(text(f'CREATE DATABASE "{DATABASE_NAME}"'))
    engine.dispose()

    Redis.from_url(args.redis_url).flushdb()
    subprocess.run(
        [sys.executable, "-m", "alembic", "-c", "api/alembic.ini", "upgrade", "head"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
    )


def start(arguments: list[str], environment: dict) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, __file__, *arguments], cwd=REPOSITORY_ROOT, env=environment)


async def wait_until_ready(url: str, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                if (await client.get(url)).status_code < 500:
                    return
            except httpx.TransportError:
                pass
            await asyncio.sleep(0.5)
    raise TimeoutError(f"{url} not ready after {timeout}s")


@asynccontextmanager
async def running(process: subprocess.Popen):
    try:
        yield process
    finally:
        process.terminate()
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            process.kill()


# --------------------------------------------------------------------------- traffic


def _chat_body(index: int) -> dict:
    return {"model": MODEL_NAME, "messages": [{"role": "user", "content": f"Summarize document {index}."}]}


async def interactive_load(client: httpx.AsyncClient, duration: float, rate: float, seed: int) -> list[tuple[float, int]]:
    """Open loop: requests keep arriving whatever the latency, as users do."""
    rng = random.Random(seed)
    results: list[tuple[float, int]] = []

    async def one_request(index: int) -> None:
        started = time.monotonic()
        try:
            response = await client.post("/v1/chat/completions", json=_chat_body(index))
            status_code = response.status_code
        except httpx.TransportError:
            status_code = 0
        results.append((time.monotonic() - started, status_code))

    tasks = []
    started = time.monotonic()
    index = 0
    while time.monotonic() - started < duration:
        await asyncio.sleep(rng.expovariate(rate))
        tasks.append(asyncio.create_task(one_request(index)))
        index += 1
    await asyncio.gather(*tasks)
    return results


async def naive_bulk_job(client: httpx.AsyncClient, requests: int, concurrency: int) -> dict:
    """What a user does today: loop over the interactive endpoint, retrying when the model is busy."""
    semaphore = asyncio.Semaphore(concurrency)
    rejected = 0

    async def one_request(index: int) -> None:
        nonlocal rejected
        async with semaphore:
            while True:
                response = await client.post("/v1/chat/completions", json=_chat_body(index))
                if response.status_code == 200:
                    return
                rejected += 1
                await asyncio.sleep(1)

    started = time.monotonic()
    await asyncio.gather(*(one_request(index) for index in range(requests)))
    return {"duration": time.monotonic() - started, "rejected_attempts": rejected}


async def batch_job(client: httpx.AsyncClient, requests: int) -> dict:
    body = {"requests": [{"custom_id": f"request-{index}", "params": _chat_body(index)} for index in range(requests)]}
    started = time.monotonic()
    response = await client.post("/v1/chat/completions/batches", json=body)
    response.raise_for_status()
    batch_id = response.json()["id"]

    while True:
        await asyncio.sleep(1)
        batch = (await client.get(f"/v1/chat/completions/batches/{batch_id}")).json()
        if batch["processing_status"] == "ended":
            break

    duration = time.monotonic() - started
    results = (await client.get(f"/v1/chat/completions/batches/{batch_id}/results")).text.splitlines()
    return {"duration": duration, "request_counts": batch["request_counts"], "result_lines": len(results)}


def summarize(results: list[tuple[float, int]]) -> dict:
    latencies = sorted(latency for latency, status_code in results if status_code == 200)
    failed = sum(1 for _, status_code in results if status_code != 200)
    percentiles = statistics.quantiles(latencies, n=100, method="inclusive") if len(latencies) > 1 else [latencies[0]] * 99
    return {
        "requests": len(results),
        "failed": failed,
        "p50": round(percentiles[49], 2),
        "p95": round(percentiles[94], 2),
        "p99": round(percentiles[98], 2),
        "max": round(latencies[-1], 2),
    }


# --------------------------------------------------------------------------- scenarios


async def benchmark(args: argparse.Namespace) -> dict:
    configuration_file = Path(tempfile.mkdtemp()) / "config.json"
    write_configuration(args, configuration_file)
    environment = {**os.environ, "CONFIG_FILE": str(configuration_file), "PYTHONPATH": str(REPOSITORY_ROOT)}
    reset_dependencies(args, environment)
    tokenizer_flag = ["--fake-tokenizer"] if args.fake_tokenizer else []

    mock_command = ["mock", "--port", str(args.mock_port), "--slots", str(args.slots), "--service-time", str(args.service_time)]
    async with running(start(mock_command, environment)):
        await wait_until_ready(f"http://127.0.0.1:{args.mock_port}/stats")
        async with running(start(["api", "--port", str(args.api_port), *tokenizer_flag], environment)):
            api_url = f"http://127.0.0.1:{args.api_port}"
            await wait_until_ready(f"{api_url}/health")

            async with httpx.AsyncClient(base_url=api_url, timeout=600, limits=httpx.Limits(max_connections=500)) as anonymous:
                login = await anonymous.post("/v1/auth/login", json={"email": "admin", "password": "changeme"})
                login.raise_for_status()
                key = login.json()["value"]

            headers = {"Authorization": f"Bearer {key}"}
            async with (
                httpx.AsyncClient(base_url=api_url, headers=headers, timeout=600, limits=httpx.Limits(max_connections=500)) as client,
                httpx.AsyncClient(base_url=f"http://127.0.0.1:{args.mock_port}") as mock,
            ):
                report = {"parameters": vars(args), "scenarios": {}}

                print("Scenario 1/4: interactive traffic alone", flush=True)
                await mock.post("/stats/reset")
                interactive = await interactive_load(client, duration=args.duration, rate=args.rate, seed=1)
                report["scenarios"]["interactive_alone"] = {"interactive": summarize(interactive), "provider": (await mock.get("/stats")).json()}

                print("Scenario 2/4: interactive traffic + bulk job through /v1/chat/completions", flush=True)
                await mock.post("/stats/reset")
                bulk = asyncio.create_task(naive_bulk_job(client, requests=args.bulk_requests, concurrency=args.bulk_concurrency))
                await asyncio.sleep(2)  # let the bulk job ramp up
                interactive = await interactive_load(client, duration=args.duration, rate=args.rate, seed=2)
                report["scenarios"]["interactive_with_naive_bulk"] = {
                    "interactive": summarize(interactive),
                    "bulk": await bulk,
                    "provider": (await mock.get("/stats")).json(),
                }

                print("Scenario 3/4: interactive traffic + the same job as a batch", flush=True)
                await mock.post("/stats/reset")
                async with running(start(["worker", *tokenizer_flag], environment)):
                    batch = asyncio.create_task(batch_job(client, requests=args.bulk_requests))
                    await asyncio.sleep(2)
                    interactive = await interactive_load(client, duration=args.duration, rate=args.rate, seed=3)
                    report["scenarios"]["interactive_with_batch"] = {
                        "interactive": summarize(interactive),
                        "batch": await batch,
                        "provider": (await mock.get("/stats")).json(),
                    }

                    print("Scenario 4/4: the batch alone", flush=True)
                    await mock.post("/stats/reset")
                    report["scenarios"]["batch_alone"] = {
                        "batch": await batch_job(client, requests=args.bulk_requests),
                        "provider": (await mock.get("/stats")).json(),
                    }

    return report


def print_report(report: dict) -> None:
    scenarios = report["scenarios"]
    print("\n| Scenario | Interactive p50 (s) | p95 (s) | p99 (s) | max (s) | Interactive failed | Bulk job duration (s) |")
    print("|---|---|---|---|---|---|---|")
    for name, label in [
        ("interactive_alone", "Interactive alone"),
        ("interactive_with_naive_bulk", "+ bulk job via /chat/completions"),
        ("interactive_with_batch", "+ same job as a batch"),
        ("batch_alone", "Batch alone"),
    ]:
        scenario = scenarios[name]
        interactive = scenario.get("interactive")
        job = scenario.get("bulk") or scenario.get("batch")
        cells = (
            [interactive["p50"], interactive["p95"], interactive["p99"], interactive["max"], f"{interactive['failed']}/{interactive['requests']}"]
            if interactive
            else ["—"] * 5
        )
        print(f"| {label} | " + " | ".join(str(cell) for cell in cells) + f" | {round(job['duration'], 1) if job else '—'} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("role", nargs="?", default="benchmark", choices=["benchmark", "mock", "api", "worker"])
    parser.add_argument("--postgres-url", default="postgresql+asyncpg://postgres:changeme@localhost:5432")
    parser.add_argument("--redis-url", default="redis://:changeme@localhost:6379/4")
    parser.add_argument("--mock-port", type=int, default=9100)
    parser.add_argument("--api-port", "--port", dest="api_port", type=int, default=8100)
    parser.add_argument("--slots", type=int, default=8, help="Requests the mock provider computes at once (its qos_limit).")
    parser.add_argument("--service-time", type=float, default=0.5, help="Seconds the mock provider takes per request.")
    parser.add_argument("--qos-retries", type=int, default=20, help="qos_retries_before_reject of the router (0.5 s each).")
    parser.add_argument("--rate", type=float, default=4.0, help="Interactive requests per second.")
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds of interactive traffic per scenario.")
    parser.add_argument("--bulk-requests", type=int, default=300, help="Requests of the bulk job.")
    parser.add_argument("--bulk-concurrency", type=int, default=32, help="Concurrency of the naive bulk client.")
    parser.add_argument("--worker-concurrency", type=int, default=4)
    parser.add_argument("--capacity-share", type=float, default=0.5)
    parser.add_argument("--fake-tokenizer", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Write the full report as JSON to this file.")
    args = parser.parse_args()

    match args.role:
        case "mock":
            run_mock_provider(port=args.mock_port, slots=args.slots, service_time=args.service_time)
        case "api":
            run_api(port=args.api_port, fake_tokenizer=args.fake_tokenizer)
        case "worker":
            run_worker(fake_tokenizer=args.fake_tokenizer)
        case "benchmark":
            report = asyncio.run(benchmark(args))
            print_report(report)
            if args.output is not None:
                args.output.write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
