"""Benchmark of the batch mode proof of concept (see adr/2026-11-11-batch-mode.md).

Question: does a bulk job hurt interactive traffic, and does the batch mode prevent it?

A mock provider stands for a GPU server: it holds `--slots` requests at a time, each takes `--service-time` seconds,
and the next ones wait in its queue. OpenGateLLM fronts it with `qos_limit = --slots`. Interactive traffic arrives at
`--rate` requests per second (Poisson) while, depending on the scenario, nothing else runs, a client sends a bulk job
through POST /v1/chat/completions as one would today, or the same job is submitted as a batch to the batch worker.

Every scenario of a repetition replays the same interactive traffic: the Poisson arrivals are drawn from a seed, the
repetition index, so the scenarios differ only by the job that runs alongside. `--repetitions` (3 by default) runs the
four scenarios that many times in the same running stack, each repetition with another seed, so that the results
capture the variance of the arrivals. The report gives, per scenario and metric, the median over the repetitions and
its min–max range; `--output` keeps the summary of every repetition.

Interactive latency is quantized: a request that finds the provider full retries its reservation every 0.5 s, so it
takes one service time, or one more 0.5 s step, or more. The p95 of a run near that step flips between two values.
The "waited" metric is steadier: the share of interactive requests, failed ones included, answered after more than
`--wait-threshold` seconds (by default the mock's `--service-time` + 0.25 s: served at once, or not).

Prerequisites: a Postgres and a Redis reachable with the URLs below (the script drops and recreates the database
`ogl_batch_benchmark` and flushes the Redis database), and the API dependencies installed (uv sync --extra api).

    python adr/scripts/2026-11-11-batch-mode-benchmark.py --output benchmark.json

`--fake-tokenizer` replaces the tiktoken encoding with a length-based count, for machines that cannot download it.

To measure a real provider instead of the mock, pass `--provider-url` (an OpenAI-compatible server, a vLLM for
instance), `--provider-model` (the model name it serves) and `--slots` (the qos_limit to apply to it), and set the
`BENCHMARK_PROVIDER_KEY` environment variable if it needs a key. Size `--rate`, `--bulk-requests` and
`--bulk-concurrency` to its capacity, and set `--wait-threshold` a little above its latency when it is not busy.
"""

import argparse
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
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
SCENARIOS = {
    "interactive_alone": "Interactive alone",
    "interactive_with_naive_bulk": "+ bulk job via /chat/completions",
    "interactive_with_batch": "+ same job as a batch",
    "batch_alone": "Batch alone",
}
INTERACTIVE_METRICS = ["p50", "p95", "p99", "max", "waited"]


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
                                "type": args.provider_type,
                                "url": args.provider_url or f"http://127.0.0.1:{args.mock_port}",
                                "key": os.getenv("BENCHMARK_PROVIDER_KEY"),
                                "model_name": args.provider_model,
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


def arrival_offsets(duration: float, rate: float, seed: int) -> list[float]:
    """Poisson arrivals, in seconds from the start of the traffic: the same seed draws the same traffic."""
    rng = random.Random(seed)
    offsets = []
    offset = rng.expovariate(rate)
    while offset < duration:
        offsets.append(offset)
        offset += rng.expovariate(rate)
    return offsets


async def interactive_load(client: httpx.AsyncClient, offsets: list[float]) -> list[tuple[float, int]]:
    """Open loop: requests keep arriving whatever the latency, as users do."""
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
    for index, offset in enumerate(offsets):
        # sleeping until an absolute time keeps a late wake-up from shifting the arrivals that follow
        await asyncio.sleep(max(0.0, started + offset - time.monotonic()))
        tasks.append(asyncio.create_task(one_request(index)))
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


def summarize(results: list[tuple[float, int]], wait_threshold: float) -> dict:
    """Percentiles of the succeeded requests; the share that waited counts the failed ones too."""
    latencies = sorted(latency for latency, status_code in results if status_code == 200)
    failed = sum(1 for _, status_code in results if status_code != 200)
    waited = sum(1 for latency, _ in results if latency > wait_threshold)
    percentiles = statistics.quantiles(latencies, n=100, method="inclusive") if len(latencies) > 1 else [latencies[0]] * 99
    return {
        "requests": len(results),
        "failed": failed,
        "waited_requests": waited,
        "waited": round(waited / len(results), 4),
        "p50": round(percentiles[49], 2),
        "p95": round(percentiles[94], 2),
        "p99": round(percentiles[98], 2),
        "max": round(latencies[-1], 2),
    }


def spread(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def aggregate(repetitions: list[dict]) -> dict:
    """Per scenario: the median and range of each metric over the repetitions, failed and waited requests summed."""
    summary = {}
    for name in SCENARIOS:
        runs = [repetition["scenarios"][name] for repetition in repetitions]
        scenario = {}
        if "interactive" in runs[0]:
            interactive = [run["interactive"] for run in runs]
            scenario["interactive"] = {metric: spread([run_summary[metric] for run_summary in interactive]) for metric in INTERACTIVE_METRICS}
            scenario["interactive"]["failed"] = sum(run_summary["failed"] for run_summary in interactive)
            scenario["interactive"]["waited_requests"] = sum(run_summary["waited_requests"] for run_summary in interactive)
            scenario["interactive"]["requests"] = sum(run_summary["requests"] for run_summary in interactive)
        jobs = [run.get("bulk") or run.get("batch") for run in runs]
        if jobs[0] is not None:
            scenario["job_duration"] = spread([round(job["duration"], 1) for job in jobs])
        summary[name] = scenario
    return summary


# --------------------------------------------------------------------------- scenarios


class ProviderStats:
    """What the mock provider saw (requests, highest concurrency). A real provider reports nothing."""

    def __init__(self, mock: httpx.AsyncClient | None) -> None:
        self.mock = mock

    async def reset(self) -> None:
        if self.mock is not None:
            await self.mock.post("/stats/reset")

    async def read(self) -> dict | None:
        return (await self.mock.get("/stats")).json() if self.mock is not None else None


async def benchmark(args: argparse.Namespace) -> dict:
    configuration_file = Path(tempfile.mkdtemp()) / "config.json"
    write_configuration(args, configuration_file)
    environment = {**os.environ, "CONFIG_FILE": str(configuration_file), "PYTHONPATH": str(REPOSITORY_ROOT)}
    reset_dependencies(args, environment)
    tokenizer_flag = ["--fake-tokenizer"] if args.fake_tokenizer else []

    async with AsyncExitStack() as stack:
        mock = None
        if args.provider_url is None:
            mock_command = ["mock", "--port", str(args.mock_port), "--slots", str(args.slots), "--service-time", str(args.service_time)]
            await stack.enter_async_context(running(start(mock_command, environment)))
            await wait_until_ready(f"http://127.0.0.1:{args.mock_port}/stats")
            mock = await stack.enter_async_context(httpx.AsyncClient(base_url=f"http://127.0.0.1:{args.mock_port}"))
        provider = ProviderStats(mock)

        async with running(start(["api", "--port", str(args.api_port), *tokenizer_flag], environment)):
            api_url = f"http://127.0.0.1:{args.api_port}"
            await wait_until_ready(f"{api_url}/health")

            async with httpx.AsyncClient(base_url=api_url, timeout=600, limits=httpx.Limits(max_connections=500)) as anonymous:
                login = await anonymous.post("/v1/auth/login", json={"email": "admin", "password": "changeme"})
                login.raise_for_status()
                key = login.json()["value"]

            headers = {"Authorization": f"Bearer {key}"}
            async with httpx.AsyncClient(base_url=api_url, headers=headers, timeout=600, limits=httpx.Limits(max_connections=500)) as client:
                report = {"parameters": vars(args), "repetitions": []}
                for repetition in range(args.repetitions):
                    # one seed per repetition: its scenarios replay the same traffic, the next repetition draws new traffic
                    offsets = arrival_offsets(duration=args.duration, rate=args.rate, seed=repetition)
                    progress = f"Repetition {repetition + 1}/{args.repetitions} (seed {repetition}, {len(offsets)} interactive requests)"
                    scenarios = {}

                    print(f"{progress}, scenario 1/4: interactive traffic alone", flush=True)
                    await provider.reset()
                    interactive = await interactive_load(client, offsets=offsets)
                    scenarios["interactive_alone"] = {
                        "interactive": summarize(interactive, wait_threshold=args.wait_threshold),
                        "provider": await provider.read(),
                    }

                    print(f"{progress}, scenario 2/4: interactive traffic + bulk job through /v1/chat/completions", flush=True)
                    await provider.reset()
                    bulk = asyncio.create_task(naive_bulk_job(client, requests=args.bulk_requests, concurrency=args.bulk_concurrency))
                    await asyncio.sleep(2)  # let the bulk job ramp up
                    interactive = await interactive_load(client, offsets=offsets)
                    scenarios["interactive_with_naive_bulk"] = {
                        "interactive": summarize(interactive, wait_threshold=args.wait_threshold),
                        "bulk": await bulk,
                        "provider": await provider.read(),
                    }

                    print(f"{progress}, scenario 3/4: interactive traffic + the same job as a batch", flush=True)
                    async with running(start(["worker", *tokenizer_flag], environment)):
                        # the worker is up once it has processed a request: its start-up is not part of the job duration
                        await batch_job(client, requests=1)
                        await provider.reset()
                        batch = asyncio.create_task(batch_job(client, requests=args.bulk_requests))
                        await asyncio.sleep(2)
                        interactive = await interactive_load(client, offsets=offsets)
                        scenarios["interactive_with_batch"] = {
                            "interactive": summarize(interactive, wait_threshold=args.wait_threshold),
                            "batch": await batch,
                            "provider": await provider.read(),
                        }

                        print(f"{progress}, scenario 4/4: the batch alone", flush=True)
                        await provider.reset()
                        scenarios["batch_alone"] = {
                            "batch": await batch_job(client, requests=args.bulk_requests),
                            "provider": await provider.read(),
                        }

                    report["repetitions"].append({"seed": repetition, "scenarios": scenarios})

    report["summary"] = aggregate(report["repetitions"])
    return report


def print_report(report: dict) -> None:
    repetitions = report["parameters"]["repetitions"]
    wait_threshold = report["parameters"]["wait_threshold"]

    def percent(share: float) -> str:
        return f"{share * 100:.1f} %"

    def value_range(metric: dict, format_value=str) -> str:
        return f"{format_value(metric['min'])}–{format_value(metric['max'])}"

    plural = "s" if repetitions > 1 else ""
    print(f"\nMedian over {repetitions} repetition{plural} (interactive failed: total over the repetition{plural})\n")
    print(f"| Scenario | Interactive p50 (s) | p95 (s) | p99 (s) | Waited > {wait_threshold:g} s | Interactive failed | Job duration (s) |")
    print("|---|---|---|---|---|---|---|")
    for name, label in SCENARIOS.items():
        scenario = report["summary"][name]
        interactive = scenario.get("interactive")
        cells = (
            [
                interactive["p50"]["median"],
                interactive["p95"]["median"],
                interactive["p99"]["median"],
                percent(interactive["waited"]["median"]),
                f"{interactive['failed']}/{interactive['requests']}",
            ]
            if interactive
            else ["—"] * 5
        )
        cells.append(scenario["job_duration"]["median"] if "job_duration" in scenario else "—")
        print(f"| {label} | " + " | ".join(str(cell) for cell in cells) + " |")

    if repetitions == 1:
        return

    print(f"\nRange (min–max) over the {repetitions} repetitions\n")
    print("| Scenario | Interactive p50 (s) | p95 (s) | p99 (s) | Waited | Job duration (s) |")
    print("|---|---|---|---|---|---|")
    for name, label in SCENARIOS.items():
        scenario = report["summary"][name]
        interactive = scenario.get("interactive")
        cells = (
            [value_range(interactive[metric]) for metric in ["p50", "p95", "p99"]] + [value_range(interactive["waited"], percent)]
            if interactive
            else ["—"] * 4
        )
        cells.append(value_range(scenario["job_duration"]) if "job_duration" in scenario else "—")
        print(f"| {label} | " + " | ".join(cells) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("role", nargs="?", default="benchmark", choices=["benchmark", "mock", "api", "worker"])
    parser.add_argument("--postgres-url", default="postgresql+asyncpg://postgres:changeme@localhost:5432")
    parser.add_argument("--redis-url", default="redis://:changeme@localhost:6379/4")
    parser.add_argument("--mock-port", type=int, default=9100)
    parser.add_argument("--api-port", "--port", dest="api_port", type=int, default=8100)
    parser.add_argument("--provider-url", default=None, help="Benchmark this OpenAI-compatible provider instead of the mock.")
    parser.add_argument("--provider-type", default="vllm", help="Provider type of --provider-url, as in the configuration file.")
    parser.add_argument("--provider-model", default=MODEL_NAME, help="Model name served by --provider-url.")
    parser.add_argument("--slots", type=int, default=8, help="qos_limit of the provider, and the capacity of the mock.")
    parser.add_argument("--service-time", type=float, default=0.5, help="Seconds the mock provider takes per request.")
    parser.add_argument("--qos-retries", type=int, default=20, help="qos_retries_before_reject of the router (0.5 s each).")
    parser.add_argument("--rate", type=float, default=4.0, help="Interactive requests per second.")
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds of interactive traffic per scenario.")
    parser.add_argument("--repetitions", type=int, default=3, help="Runs of the four scenarios, each with other arrivals.")
    parser.add_argument(
        "--wait-threshold",
        type=float,
        default=None,
        help="Latency in seconds above which an interactive request counts as waited (default: --service-time + 0.25).",
    )
    parser.add_argument("--bulk-requests", type=int, default=300, help="Requests of the bulk job.")
    parser.add_argument("--bulk-concurrency", type=int, default=32, help="Concurrency of the naive bulk client.")
    parser.add_argument("--worker-concurrency", type=int, default=4)
    parser.add_argument("--capacity-share", type=float, default=0.5)
    parser.add_argument("--fake-tokenizer", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Write the full report as JSON to this file.")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if args.wait_threshold is None:
        args.wait_threshold = args.service_time + 0.25

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
