# ADR — Batch mode for chat completions

* **Status:** Proposed
* **Date:** 2026-11-11
* **Authors:** _to be completed_
* **Related:**
  * Proof of concept: branch `feat/batch-mode` (code, tests, benchmark script [`scripts/2026-11-11-batch-mode-benchmark.py`](scripts/2026-11-11-batch-mode-benchmark.py))
  * Reference API: [Anthropic Message Batches](https://platform.claude.com/docs/en/build-with-claude/batch-processing)
* **Decision Outcome:** _pending review_. Proposal: add an asynchronous batch API for chat completions, modeled on Anthropic's Message Batches, processed by an optional worker that only uses the provider capacity interactive traffic leaves free.

---

## Context

### Bulk workloads compete with interactive users

OpenGateLLM fronts GPU servers (mostly vLLM) that are provisioned for the interactive peak. Part of the traffic is not interactive at all: classifying or extracting data from a corpus, summarizing documents, evaluating a model, generating synthetic data. Nobody waits for these results in front of a screen; they only need them "today".

Today the only way to run such a job is to loop over `POST /v1/chat/completions`. The gateway then treats each of these requests exactly like a user's chat message:

* The provider QoS (`qos_limit` + `qos_retries_before_reject`) protects the providers from overload, but it has no notion of priority: an interactive request waits in the same reservation retry loop as the bulk requests, and is rejected with a 503 just as often.
* Rate limits (RPM/TPM) are the only lever left, and they cap the bulk job without protecting the other users.
* The client has to implement concurrency control, retries and resumption itself, and keep connections open for the whole job.

> **TODO (figures):** GPU utilization over a typical week (Grafana `vllm` dashboards), to size the idle capacity outside office hours.
>
> **TODO (demand):** user requests for bulk processing (support tickets, issues, known projects looping over the API).

### Measured impact (synthetic benchmark)

The proof of concept ships a benchmark ([`scripts/2026-11-11-batch-mode-benchmark.py`](scripts/2026-11-11-batch-mode-benchmark.py)). A mock provider stands for a GPU server: it computes 8 requests at a time, 0.5 s each (16 requests/s at most), and OpenGateLLM fronts it with `qos_limit: 8` and `qos_retries_before_reject: 20`. Interactive traffic arrives at random (Poisson) for 30 s while a 300-request job runs. The job is sent either as users send it today, looping over `/v1/chat/completions` with 32 concurrent requests and a retry on 503, or as a batch.

**Interactive traffic at 4 requests/s (about 25 % of the capacity)**

| Scenario | Interactive p50 | p95 | p99 | max | Interactive failed | Job duration |
|---|---|---|---|---|---|---|
| Interactive traffic alone | 0.54 s | 0.57 s | 0.59 s | 0.61 s | 0 / 122 | — |
| + job looping over `/v1/chat/completions` | 1.21 s | **6.64 s** | **9.84 s** | 10.28 s | **3 / 104** | 27 s |
| + same job submitted as a batch | 0.54 s | **0.56 s** | **0.57 s** | 0.57 s | **0 / 100** | 58 s |
| Batch alone (no interactive traffic) | — | — | — | — | — | 43 s |

**Interactive traffic at 8 requests/s (about 50 % of the capacity)**

| Scenario | Interactive p50 | p95 | p99 | max | Interactive failed | Job duration |
|---|---|---|---|---|---|---|
| Interactive traffic alone | 0.54 s | 1.06 s | 1.22 s | 1.72 s | 0 / 245 | — |
| + job looping over `/v1/chat/completions` | 2.16 s | **8.72 s** | **9.77 s** | 10.68 s | **29 / 239** | 39 s |
| + same job submitted as a batch | 0.54 s | **0.60 s** | **1.06 s** | 1.10 s | **0 / 225** | 68 s |

Every batch request succeeded. A bulk job sent the way users send it today makes the interactive p95 latency about 10 times worse, and at 50 % load 12 % of the interactive requests fail. The same job sent as a batch leaves interactive latency at the level of interactive traffic alone (the differences between these two rows are within the run-to-run variance of random arrivals). It takes longer because it only uses the capacity left over, and it slows down as interactive load grows.

These numbers come from a mock provider on a single machine. They show the mechanism, not production capacity. Real figures need a run against a staging vLLM.

### Market reference

Anthropic (Message Batches) and OpenAI (Batch API) both offer an asynchronous mode: requests are processed within 24 hours, at half price, and the results are downloaded once the batch has ended. Developers know this contract.

---

## Decision

| # | Decision |
|---|---|
| 1 | Add an **asynchronous batch API for chat completions**, modeled on Anthropic's Message Batches: requests sent inline with a `custom_id`, `processing_status` `in_progress` → `canceling` → `ended`, `request_counts`, JSONL results, cancellation, expiration after 24 h. |
| 2 | **Persist batches in Postgres** and use it as the queue (`FOR UPDATE SKIP LOCKED` + lease). No new dependency. |
| 3 | **Process batches in a separate, optional worker process** (`python -m api.worker`, same image as the API), scaled by replicas. |
| 4 | **Batch requests only use spare capacity**: they go through the same provider concurrency limiter as interactive requests, with a lower limit. |
| 5 | **Reuse the forwarding pipeline**: routing, access control, provider selection, usage, cost and carbon footprint are those of `POST /v1/chat/completions`. |

### 1. API

| Anthropic | OpenGateLLM | Notes |
|---|---|---|
| `POST /v1/messages/batches` | `POST /v1/chat/completions/batches` | `{"requests": [{"custom_id", "params"}]}`, `params` being a `/v1/chat/completions` body; `stream: true` is rejected. The models are checked at creation (exists, `text-generation`, user has access). |
| `GET /v1/messages/batches/{id}` | `GET /v1/chat/completions/batches/{batch_id}` | `object: "batch"`, `processing_status`, `request_counts {processing, succeeded, errored, canceled, expired}`, `created`, `expires`, `ended`, `cancel_initiated`. |
| `GET /v1/messages/batches` | `GET /v1/chat/completions/batches` | Paginated with `offset`/`limit`, like every list of the API. |
| `POST /v1/messages/batches/{id}/cancel` | `POST /v1/chat/completions/batches/{batch_id}/cancel` | Pending requests are canceled at once, requests in progress run to the end. |
| `GET /v1/messages/batches/{id}/results` | `GET /v1/chat/completions/batches/{batch_id}/results` | JSON Lines, one `{"custom_id", "result": {"type", "response", "error"}}` per request, available once the batch has ended. |
| `DELETE /v1/messages/batches/{id}` | not in the POC | |

The path nests the batch under the endpoint it batches, as Anthropic does. It leaves `/v1/batches` free for an OpenAI-compatible façade (see alternatives) and makes room for `/v1/embeddings/batches`. The new `RouterName.BATCHES` can be disabled like any router.

Inline requests rather than OpenAI's file upload: OpenGateLLM has no file storage, and the Anthropic contract needs none.

### 2. Storage and queue

Two tables: `batch` (owner, key, status, timestamps) and `batch_request` (one row per request: `custom_id`, `position`, `params`, `status`, `attempts`, `available_at`, `locked_until`, `response`, `error`). Request counts are aggregated at read time.

Postgres is the queue. Redis cannot be: gunicorn flushes it at every API start (`scripts/gunicorn.conf.py`), and it is not configured for durability.

* **Claim**: one `UPDATE … WHERE id IN (SELECT … FOR UPDATE SKIP LOCKED) RETURNING …` statement marks requests `processing`, gives each a `claim_id` and a lease (`locked_until`). Several workers never claim the same request. Only requests of batches `in_progress` are claimed.
* **Lease**: the worker extends the leases of the requests it is processing every 20 s (lease: 60 s), so a claim stays valid however long the provider takes, and expires about a minute after its worker dies.
* **Crash safety**: a request whose lease expires goes back to the queue and counts an attempt, so a request that kills its worker is not retried forever. The outcome is written only by the holder of the current `claim_id`: a worker whose claim expired cannot overwrite the result of the worker that claimed the request again.
* **Fairness**: requests are claimed lowest `position` first (rank of the request in its batch), so a small batch submitted after a large one is not stuck behind it. The flip side: under a steady inflow of new batches, the tail of a large batch waits, and may expire.
* Every worker-side statement is a single statement on an `AutocommitSession`: the worker never holds a transaction, nor a pooled connection, during a provider call.

### 3. Worker

`api/worker.py` is a new composition root, like `lifespan.py`. `BatchWorker` (`api/infrastructure/worker/`) claims up to `batch_worker_concurrency` requests, processes them concurrently, extends their leases, and runs a periodic reconciliation step: release expired claims, cancel the requests that went back to the queue after their batch was canceled, expire the pending requests of overdue batches, end finished batches. On SIGTERM it stops claiming and waits up to 30 s for the requests in progress; the ones it interrupts go back to the queue when their claim expires.

The worker is optional, and the reconciliation runs inside it: without a worker, batches are accepted and stay `in_progress` indefinitely.

### 4. Spare capacity only

The provider concurrency limiter counts every request a provider holds, interactive or batch, in one Redis sorted set. Interactive requests reserve a place while the load is below `qos_limit`. Batch requests reserve one only while the load is below `batch_provider_capacity_share × qos_limit` rounded down (0.5 by default: 4 places out of 8, none out of 1), or below `batch_provider_default_concurrency` for providers without `qos_limit`.

* Interactive traffic always keeps at least the rest of the capacity, and takes all of it when batch requests are not running.
* A batch request never waits for a provider and never gets a 429: without spare capacity it goes back to the queue and the worker pauses its claims briefly.
* When Redis is unreachable and the load cannot be read, batch requests wait. Interactive requests keep failing open, as today.
* A transient provider failure (timeout, 429, 5xx) is retried with exponential backoff, up to 3 attempts. Other failures end the request as `errored`.

### 5. Reuse

`ProcessBatchRequestUseCase` subclasses `ProviderRequestForwardingUseCase` with its own `execute()`. Its flow differs (never wait, requeue instead of 503), as `CreateChatCompletionsUseCase` does for streaming. It reuses `_get_usable_router`, `_order_by_strategy`, `_get_billed_tokens`, `_compute_usage` and `_record_and_save_usage`. Access is checked again at processing time: a revoked key, an expired user or a model whose access was removed fails the request.

Each succeeded request writes a regular `usage` row with `endpoint = /v1/chat/completions/batches`, so `GET /v1/usage` can filter batch consumption.

---

## Alternatives considered

| Option | Description | Decision |
|---|---|---|
| A | Do nothing: users loop over `/v1/chat/completions` | Rejected: see the benchmark. |
| B | A priority on the interactive endpoint (header, or `user.priority` in the QoS) | Complementary, not a substitute: the client still holds connections and implements retries and resumption. Still worth doing later. |
| C | OpenAI-compatible `/v1/files` + `/v1/batches` | Deferred. Works with the OpenAI SDK, but needs a Files API and object storage (a new dependency). The queue, the worker and the capacity rule (about 80 % of the work) are the same, so this façade can be added on top later. |
| D | **Anthropic-style inline batches** | **Selected**: no file storage, a contract developers know. |
| E | Delegate to provider-native batch APIs (OpenAI, Mistral) | Out of scope: self-hosted vLLM has no online batch API. Could be an adapter later for external providers. |
| F | A job queue library (Celery, arq, procrastinate) | Not needed: the Postgres queue fits in one repository class. Celery or arq would add a broker; procrastinate (Postgres-based) remains an option if the queue grows more complex. |
| G | Run the worker inside the API processes (lifespan task) | Rejected: it couples batch load with API latency, multiplies workers by `gunicorn workers × replicas`, and constrains shutdown. |

---

## Consequences

**Becomes easier**

* Bulk jobs no longer degrade interactive latency, and use the GPUs when interactive traffic is low.
* Users get a managed asynchronous API: no concurrency control, retries or resumption to write, and results kept server-side.
* Batch consumption is visible in `GET /v1/usage` and can be priced on its own.

**Becomes harder**

* One more process to deploy and monitor (the worker), with its own Helm/compose service.
* Two more tables, whose size grows with the results kept: a retention period and a purge job are needed.
* At-least-once processing: if a worker dies after the provider answered but before the result is stored, the request is processed, and billed, twice. Only one result is stored.
* Maintenance surface: the POC adds about 1,900 lines of production code and 1,450 lines of tests.

---

## Proof of concept

The POC on branch `feat/batch-mode` implements decisions 1–5 for chat completions, with unit tests for every use case branch and for the worker loop, integration tests for the repository, the endpoints and the migration, and the benchmark above. The existing test suite passes unchanged. An independent code review found two state-transition bugs (a stale outcome overwriting a newer one after a lease expiry, canceled batches still processing their requeued requests), fixed in the POC.

Not in the POC (needed before production):

* **quotas for batches (blocking)**: the POC neither checks nor charges any rate limit for batch requests, so a user with an RPD limit could bypass it through a batch. The router must ship disabled until open question 1 is settled;
* a batch price (open question 2);
* retention and purge of results, `DELETE`;
* Prometheus metrics (queue depth, batch throughput), Langfuse traces for batch requests (the worker uses a no-op tracer);
* the `test_autocommitsession.py`-style probe for the worker;
* claiming per router: a worker currently claims before knowing whether the router has spare capacity, and pauses all claims when one router is full;
* refactoring rather than copying: `ProcessBatchRequestUseCase` copies the provider reservation loop of the forwarding template with another limit, and carries a rate limiter it never calls. Passing the limit to the template, and moving `UsageRecorder` out of `infrastructure/fastapi/`, would remove both;
* Helm/compose service, documentation, playground page, `AGENTS.md` section on the worker.

---

## Open questions for the maintainers

1. **Rate limits (blocking).** The POC neither checks nor charges RPM/TPM/RPD/TPD for batch requests; Anthropic keeps separate batch limits. Should batches count toward the daily limits (TPD/RPD)? Should a new limit type cap the number of pending batch requests per user or role?
2. **Price.** A batch discount (a multiplier on the router cost, global or per router)?
3. **Capacity share.** Is 50 % of `qos_limit` the right default? Should it be configurable per router? What should the default be for providers without `qos_limit`?
4. **Fairness unit.** Batches are interleaved by progress. Should fairness be per user or per organization instead, and should it use `user.priority`?
5. **Limits and retention.** 10,000 requests per batch in the POC (Anthropic: 100,000 or 256 MB, too heavy to parse in an API worker). How long are results kept (Anthropic: 29 days)?
6. **URL.** `/v1/chat/completions/batches` (proposed) or `/v1/batches`?
7. **Results errors.** The POC reports domain error names (`ModelNotFoundError`, `StatusCodeModelError`…). Do we want a public error taxonomy?
8. **Deployment.** Is a separate worker service acceptable for the deployments we know of (Albert, others)?

---

## Plan if accepted

1. Domain, tables, repository (Postgres queue).
2. API endpoints, router disabled by default.
3. Worker and capacity-capped processing.
4. Quotas, price, retention and purge, metrics, deployment and documentation.
5. Later: playground page, `/v1/embeddings/batches`, OpenAI-compatible façade, provider-native batch adapters.
