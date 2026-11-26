# Pipeline Component Overview

A plain-English guide to every file, module, and database table in the current **Discogs Tracklist Discovery & Evaluation Pipeline**.

---

## In 30 Seconds

The pipeline takes messy release metadata (`barcode`, `album_title`, `album_artist`), finds the matching release edition on Discogs, pulls out its ordered tracklist and separated credits, checks how trustworthy the match is, and stores everything in PostgreSQL so repeated lookups are instant.

Every run returns an honest status instead of guessing: `matched` (confident match), `needs_review` (matched, but a human should double-check diverging editions or missing details), `not_found` (404), or `upstream_error` (502/503 if Discogs fails and no local backup exists).

```mermaid
flowchart TD
    Entry["Entry Point: REST API / Dashboard (main.py, dashboard.py) or CLI (cli.py)"] --> Norm["1. Normalize Input (normalizer.py)"]
    Norm --> SaveIn["2. Save Input Record (repository.py -> input_records)"]
    SaveIn --> DbCheck{"3. Already in local DB? (Layers 1 & 2)"}
    DbCheck -- "Fresh 200 match" --> ReturnDB["Return 200 OK (source: database)"]
    DbCheck -- "404 within TTL" --> Return404Cache["Return 404 Not Found (source: database_cache)"]
    DbCheck -- "Miss or force_refresh" --> Discogs["4. 3-Tier Discogs Search & Fetch (discogs_client.py + api_cache)"]
    Discogs -- "Matched release" --> Extract["5. Extract & Evaluate (extractor.py, evaluator.py)"]
    Discogs -- "Not found / error" --> ExtFB["6. External Fallbacks: MusicBrainz / JPC / Deezer / iTunes (fallback_providers.py)"]
    ExtFB -- "Matched release" --> Extract
    ExtFB -- "Exhausted" --> AgentFB["7. Optional Gemini + Google Search Agent Fallback (agent_client.py)"]
    AgentFB -- "Discogs URL bridge" --> Extract
    AgentFB -- "Verified answer" --> SaveFB["Save Agent Fallback (agent_fallbacks, agent_fallback_tracks)"]
    AgentFB -- "No match / unconfigured" --> Out404["Return 404 Not Found or 502/503 (if no DB backup)"]
    Extract --> Enrich["8. 5-Tier Track Enrichment: Tier 0 -> 1 -> 2a -> 2b -> opt-in Tier 3 Gemini (enricher.py)"]
    SaveFB --> Enrich
    Enrich --> OutMatch["Persist Provenance + Coverages & Return 200 (matched / needs_review)"]
```

---

## Component Map at a Glance

| Component / File | Layer | What It Does in One Sentence |
| --- | --- | --- |
| [docker-compose.yml](../docker-compose.yml) | Infrastructure | Starts the PostgreSQL 16 database and FastAPI container together with persistent storage, queue/enrichment env defaults, and volume mounts. |
| [.env.example](../.env.example) | Infrastructure | Template listing Discogs, Gemini, fallback, worker queue, and idle auto-enrichment environment variables. |
| [Dockerfile](pipeline/Dockerfile) & [requirements.txt](pipeline/requirements.txt) | Infrastructure | Packages Python 3.12 and dependencies (`fastapi`, `uvicorn`, `httpx`, `psycopg`, `pydantic`, `google-genai`, `pytest`). |
| [config.py](pipeline/app/config.py) | Infrastructure | Loads environment settings (DB, Discogs, Gemini, fallbacks, worker concurrency, track enrichment, idle auto-sweep) into a cached `Settings` object. |
| [main.py](pipeline/app/main.py) | Entry Point | Runs the FastAPI server, Swagger UI (`/docs`), HTML dashboard routes, REST lookup/batch/review endpoints, and async `/jobs/*` queue API. |
| [dashboard.py](pipeline/app/dashboard.py) | Entry Point | Renders interactive `/results`, `/results/{input_id}` (with `Featured` credits and provenance tags), and `/queue` dashboards with opt-in `Allow Gemini` controls. |
| [cli.py](pipeline/app/cli.py) | Entry Point | Provides `python -m app.cli` commands (`lookup`, `batch`, `enqueue-batch`, `retry-not-found`, `enrich`, `enqueue-enrich`) for terminal and script workflows. |
| [models.py](pipeline/app/models.py) | Data Models | Defines strict Pydantic schemas for validation, tracks, credits, provenance, evaluation metrics, queue status (`idle_auto_enrichment_enabled`), and token/cost accounting. |
| [normalizer.py](pipeline/app/normalizer.py) | Core Logic | Cleans barcodes, artist names, titles, format tags (`(2CD)`, `RSD`, `Various`), and label/catno hints, hashing them into a deduplication key. |
| [service.py](pipeline/app/service.py) | Core Logic | Orchestrates DB deduplication, Discogs 3-tier lookup, external/agent fallbacks, track enrichment persistence, and atomic JSON/Markdown batch reports (`ISRC Cov` + live coverages). |
| [discogs_client.py](pipeline/app/discogs_client.py) | Core Logic | Talks to Discogs REST API v2 behind `CatalogProvider` with PostgreSQL `api_cache`, master version queries, rate pacing, retries, and typed errors. |
| [fallback_providers.py](pipeline/app/fallback_providers.py) | Core Logic | Implements MusicBrainz, JPC, Deezer, and iTunes album fallbacks (mapped into canonical negative-ID releases) between Discogs and Gemini. |
| [agent_client.py](pipeline/app/agent_client.py) | Core Logic | Runs optional Gemini + Google Search two-pass structured album fallback and batched missing-track enrichment with usage token and cost tracking. |
| [enricher.py](pipeline/app/enricher.py) | Core Logic | Runs 5-tier provenance-tracked track enrichment (`Tier 0 -> 1 -> 2a -> 2b -> opt-in Tier 3`), using unquoted Deezer track queries, base-title parenthetical fallback, live-release guards, and >25-track Gemini chunking. |
| [queue_worker.py](pipeline/app/queue_worker.py) | Core Logic | Runs 3 PostgreSQL worker pools (`queued_discogs` -> `queued_agent` -> `queued_enrich`), crash recovery, and Idle-Queue Auto-Enrichment (`track_enrich_auto`). |
| [extractor.py](pipeline/app/extractor.py) | Core Logic | Parses raw Discogs/fallback JSON into ordered multi-disc tracks, resolved performers, separated credit buckets (`composers`, `lyricists`, `producers`, `featured`, `other`), and `tracklist_hash`. |
| [evaluator.py](pipeline/app/evaluator.py) | Core Logic | Scores candidate releases, rejects bad matches, checks whether competing editions disagree on tracks, and computes match confidence & track coverage metrics. |
| [init.sql](pipeline/sql/init.sql) & [repository.py](pipeline/app/repository.py) | Storage | Defines the 13-table PostgreSQL schema with idempotent migrations and runs pooled queries, enrichment token/cost tracking, reviews, and atomic job transitions. |
| [tests/](pipeline/tests/) | Tests | Houses 61 unit and PostgreSQL integration tests (`pytest`) across 10 modules covering every stage, worker pool, enricher tier, and dashboard UI. |

---

## How Every Component Works

### Infrastructure & Configuration

- **[docker-compose.yml](../docker-compose.yml) & [.env.example](../.env.example)**:
  - Runs two containers: `db` (`postgres:16-alpine` with a persistent `pgdata` volume, a `pg_isready` healthcheck, and automatic schema initialization from [init.sql](pipeline/sql/init.sql) on first boot) and `api` (exposed on port `8000` by default).
  - Mounts `./data` read-only at `/app/data` (for [development.csv](../data/development.csv)) and `./outputs` at `/app/outputs` (so generated batch reports appear directly on your host machine).
  - Reads settings from `.env` (see [.env.example](../.env.example)): Discogs (`DISCOGS_TOKEN`, `DISCOGS_USER_AGENT`), Gemini (`GEMINI_API_KEY`, `GOOGLE_GENAI_MODEL`, `ENABLE_AGENT_FALLBACK`), PostgreSQL, queue worker concurrency (`DISCOGS_QUEUE_CONCURRENCY`, `AGENT_QUEUE_CONCURRENCY`, `ENRICH_QUEUE_CONCURRENCY`), and enrichment controls (`ENABLE_TRACK_ENRICHMENT`, `ENABLE_AGENT_TRACK_ENRICHMENT`, `ENABLE_IDLE_AUTO_ENRICHMENT`, `IDLE_ENRICHMENT_INTERVAL_SEC`, `IDLE_ENRICHMENT_BATCH_SIZE`).
- **[Dockerfile](pipeline/Dockerfile) & [requirements.txt](pipeline/requirements.txt)**:
  - Builds a slim Python 3.12 image containing `app/`, `sql/`, and `tests/`, installs FastAPI, Uvicorn, HTTPX, `psycopg[binary,pool]`, Pydantic Settings, `google-genai`, and `pytest`, and launches `uvicorn app.main:app` on port `8000`.
- **[config.py](pipeline/app/config.py)**:
  - Defines the Pydantic `Settings` class (cached via `@lru_cache`) for DB, Discogs, external fallbacks, Gemini fallback/enrichment, queue concurrency, idle auto-sweep intervals, and rate limits.

### Entry Points (How You Run It)

- **Web API & Dashboards — [main.py](pipeline/app/main.py) & [dashboard.py](pipeline/app/dashboard.py)**:
  - Uses FastAPI's `lifespan` hook to open a `psycopg_pool.ConnectionPool`, run idempotent schema migrations (`repo.ensure_schema()`), start `QueueManager` workers (when `ENABLE_QUEUE_WORKERS=true`), and serve Swagger UI at `/docs`.
  - **HTML Dashboards:**
    - `GET /results` & `GET /results/{input_id}` — Interactive record browser and track/credit inspector showing provenance tags (`[musicbrainz]`, `[deezer]`, `[itunes]`, `[gemini_agent]`), ISRCs, `Featured` credits, human/agent reviews, single/bulk `Enrich Missing` controls with opt-in `Allow Gemini`, and Gemini retries.
    - `GET /queue` — Live-polling 3-pool queue monitor (`Discogs`, `Agent`, `Enrich`, plus `Idle:on/off` indicator), single-lookup enqueue form, bulk agent retry, bulk enrichment (`Allow Gemini`), and per-job item inspector/cancellation.
  - **JSON Endpoints:**
    - `GET /health`, `POST /lookup`, `POST /batch/csv`, `GET /records`, `GET /records/{input_id}`, `POST /records/{input_id}/review`, `GET /releases/{discogs_release_id}`.
    - Sync & Async Enrichment/Queue: `POST /records/{input_id}/enrich`, `POST /records/enrich-missing`, `POST /jobs/batch`, `POST /jobs/lookup`, `POST /jobs/agent-retry/{input_id}`, `POST /jobs/retry-not-found`, `POST /jobs/enrich/{input_id}`, `POST /jobs/enrich-missing`, `GET /jobs/status`, `GET /jobs`, `GET /jobs/{job_id}`, and `POST /jobs/{job_id}/cancel`.
- **Command Line Interface — [cli.py](pipeline/app/cli.py)**:
  - Invoked via `python -m app.cli` with the same `LookupService` and PostgreSQL backend:
    - `lookup --barcode ... --title ... --artist ... [--refresh]` (`0` = matched/needs_review, `1` = not_found, `2` = upstream_error)
    - `batch --csv ... [--rows 1,3,7,13,16] [--output /app/outputs/batch_report.json] [--refresh]`
    - `enqueue-batch`, `retry-not-found`, `enrich --input-id ... | --all-missing [--use-agent] [--refresh]`, and `enqueue-enrich` (`--input-id ... | --all-missing [--use-agent]`).

### Data Contracts

- **[models.py](pipeline/app/models.py)**:
  - Holds all Pydantic v2 request, internal, and response models so inputs and outputs always follow an explicit schema:
    - `LookupRequest`: Requires at least a valid non-zero barcode or a non-empty `album_title` (so empty inputs or `"0"` barcodes without a title fail validation immediately with `422`).
    - `NormalizedInput` & `FormatHint`: Carry the cleaned fields, barcode variants, detected format hints (`medium`, `qty`, `tags`), `is_compilation` flag, and `input_hash`.
    - `LookupSuccessResponse` & `LookupNotFoundResponse`: Wrap the matched `ReleaseOut`, ordered `TrackOut` list (with `TrackArtistOut` and credit buckets of `TrackCreditOut`), `AlternativeEditionOut` list, and `EvaluationOut` metrics (or `queries_attempted` + `rejection_reasons` on `404`).
    - `BatchRequest`, `BatchRowOutcome`, `BatchSummaryResponse`, `RecordSummaryOut`, `RecordDetailOut`, and `HealthResponse`.

### Core Pipeline Stages

#### 1. Input Normalization — [normalizer.py](pipeline/app/normalizer.py)
Turns inconsistent spreadsheet/API strings into clean, comparable inputs before any search happens:
- **Barcodes:** Strips spaces, hyphens, and non-digits; treats empty values and all-zero placeholders like `"0"` as missing; generates 12-digit UPC ↔ 13-digit EAN variants (e.g., `"198028875721"` ↔ `"0198028875721"`).
- **Artists:** Maps compilation aliases (`various`, `various artists`, `va`, `v.a.`, `various (sampler)`, `sampler`) to `is_compilation=True`, `artist_norm="various"`; un-inverts trailing articles (`"Fantastischen Vier, Die"` → `"Die Fantastischen Vier"`); strips Discogs disambiguators like `"Casper (9)"`.
- **Titles & Format Hints:** Normalizes smart quotes/dashes and Unicode (`NFKC`); strips duplicated leading artist prefixes separated by `;` (e.g., `"JETHRO TULL;BALOISE SESSION(2CD)"` → `"BALOISE SESSION"`); extracts leading format prefixes (`"CD - schönhauser EP"`) and parenthetical hints (`(2CD)`, `(2LP)`, `(LP)`, `(RSD Exklusive)`, `(Deluxe)`, `(Live...)` — keeping `Live` inside the cleaned title while tagging it) into `FormatHint(medium, qty, tags)`.
- **`input_hash`:** Computes a SHA-256 hash over `(barcode_norm, title_norm, artist_norm, format_hint)` so differently formatted submissions of the same record map to one canonical `input_records` row.

#### 2. Pipeline Orchestrator — [service.py](pipeline/app/service.py)
`LookupService` connects normalization, PostgreSQL deduplication, Discogs search, extraction, evaluation, and batch reporting:
- **3-Layer Deduplication (when `force_refresh=False`):**
  1. **Same-input history:** Checks `lookup_runs` for this `input_id`. A prior `matched`/`needs_review` returns `200 OK` (`source="database"`). A prior `not_found` within `not_found_ttl_hours` (`24`h) returns `404` (`source="database_cache"`).
  2. **Cross-input local DB lookup:** Searches `release_barcodes` (all UPC/EAN variants) and `releases` (`artist_norm`, `title_norm`, or `title_norm` alone for compilations). If stored releases pass `evaluate_candidates()`, returns `200 OK` (`source="database"`) with zero Discogs calls.
  3. **Canonical release storage:** Every Discogs release is upserted once by `discogs_release_id`, so multiple inputs share one canonical release/tracklist.
- **3-Tier Discogs Search Cascade:**
  - **Tier 1 (`tier1_barcode`):** Searches `/database/search?barcode=...` across normalized barcode variants, stopping as soon as candidate IDs appear. Fetches up to `max_candidates_to_fetch` (`4`) full releases and evaluates them.
  - **Tier 2 (`tier2_structured`):** Runs if Tier 1 found no candidates **or** all Tier 1 candidates failed evaluation (e.g., empty tracklists or barcode collisions). Searches `release_title` (plus `artist` unless `is_compilation=True`).
  - **Tier 3 (`tier3_freetext`):** Runs if Tier 2 returned `0` new candidates, falling back to a general `q="<artist> <title>"` query.
- **Offline Fallback:** If Discogs throws `DiscogsError` (`401`/`403`/`429`/`5xx`/network failure), checks prior successful runs for this input and then cross-input local DB matches. If found, returns `200 OK` (`source="database_fallback"`, `fallback_used=True`, plus a `warning`) without overwriting valid `input_matches`. If no local backup exists, logs `upstream_error` and raises `UpstreamServiceError` (`503` for rate limit `429`, `502` otherwise).
- **Batch Processing (`run_batch`):** Loops through selected CSV rows, isolates per-row `UpstreamServiceError` (`upstream_error`) and `ValueError` (`validation_error`, HTTP `422`), and writes both a full JSON report and a companion Markdown summary table.

#### 3. Discogs API Client — [discogs_client.py](pipeline/app/discogs_client.py)
`DiscogsClient` implements the `CatalogProvider` protocol (making it easy to plug in additional sources like MusicBrainz later):
- **Response caching:** Hashes `(endpoint, params)` and checks PostgreSQL's `api_cache` table before sending an HTTP GET (unless `bypass_cache=True`). Only `200` responses are cached.
- **Auth & headers:** Requires a non-placeholder `DISCOGS_TOKEN` and sends mandatory `User-Agent` and `Authorization: Discogs token=...` headers.
- **Rate pacing & retries:** Reads `X-Discogs-Ratelimit-Remaining` on every response and sleeps `1.1s` when remaining quota drops to `<= discogs_min_remaining` (`5`). Retries up to 2 times on `429` (honoring `Retry-After` capped at `10s`), `5xx` errors, and transient `httpx.RequestError`s.
- **Truthful error mapping:** Treats HTTP `404` on `/releases/{id}` as `{}` (missing release), validates JSON parsing, and raises `DiscogsAuthError`, `DiscogsRateLimitError`, or `DiscogsUpstreamError` so outages are never disguised as `404 not_found`.

#### 4. Tracklist & Credit Extractor — [extractor.py](pipeline/app/extractor.py)
`extract_release()` turns raw Discogs release JSON into structured `ExtractedRelease`, `ExtractedTrack`, and `ExtractedCredit` objects:
- **Multi-disc & hierarchy handling:**
  - Reads `"heading"` items (`CD 1`, `Side B`) to update `disc_number` and `disc_label` without emitting headings as songs.
  - Infers `disc_number` from position prefixes (`1-1`, `CD2-3`), vinyl sides restricted to `A–L` (`A`/`B` = disc 1, `C`/`D` = disc 2, avoiding false disc bumps on `"V1"` video tracks), and numeric track resets (track `1` following track `>1`), carrying disc advances forward to trailing unprefixed tracks.
  - Unpacks `"index"` items' `"sub_tracks"` while retaining `parent_position` and inheriting parent artists, disc number, and parent `extraartists`.
- **Performing artists & durations:** Uses track-level `artists[]` or inherits release-level `artists[]` (`inherited=True`), joins multi-artist credits cleanly, flags `has_resolved_performer` (`False` for `"Various"`/`"Unknown"`), and parses `mm:ss` / `h:mm:ss` durations into `duration_sec` and `has_valid_duration`.
- **Separated credits:** Merges track-level `extraartists` and release-level `extraartists`:
  - `tracks == ""` → applied to every track (`credit_source="release_all"`).
  - Range strings like `"1, 3-5"`, `"A1-A3"`, `"1 to 4"`, `"01"`, en/em-dashes → matched to track positions (`credit_source="release_range"`).
  - Unparseable range strings → saved in `unscoped_credits` rather than guessed.
  - Splits multi-role strings on commas outside `[...]` qualifiers and classifies each into `composer`, `lyricist`, `producer`, `featured`, or `other` (kept separate from performing artists).
- **Fingerprints & barcodes:** Extracts normalized barcode variants from `identifiers[]` and computes `tracklist_hash` (SHA-256 over `[disc_number, normalized_title, duration_sec]` per track) to compare editions reliably.

#### 5. Candidate Evaluator — [evaluator.py](pipeline/app/evaluator.py)
Decides which fetched releases are valid matches, picks the best edition, and flags cases that need human review:
- **Rejection gates (`evaluate_candidate`):**
  - Rejects releases with `0` tracks.
  - Rejects barcode collisions where title/artist strongly contradict the input (`title_sim < 0.45` on compilations, or `title_sim < 0.25 and artist_sim < 0.25` on single acts).
  - Rejects non-barcode matches with low similarity (`title_sim < 0.65` for compilations; `title_sim < 0.55` or `artist_sim < 0.55` for single acts) or total score `< 0.65`.
- **Confidence scoring (`0.0–1.0`):**
  - **Barcode verified:** `0.55 + 0.20*title_sim + 0.15*artist_sim + 0.10*format_score + tie_breaker`.
  - **No barcode match:** `0.52*title_sim + 0.36*artist_sim + 0.12*format_score + tie_breaker`.
  - Similarity (`string_similarity`) accent-folds text and takes the maximum of `SequenceMatcher` ratio, token Jaccard overlap, containment (`0.85`), and EP-stripped title comparison. `tie_breaker` adds up to `+0.05` from `community_have` and `+0.02` for `"Complete and Correct"` / `"Correct"` data quality.
- **Edition disagreement & quality audit (`evaluate_candidates`):**
  - Sorts passing candidates by `(score, community_have)` and compares `tracklist_hash` across all passing editions.
  - If editions have different tracklists (`editions_agree=False`), auto-picks the top scorer, lists the others in `alternatives[]`, and sets `status="needs_review"` — unless the top pick is a `clear_winner` (confirmed barcode + confirmed format + beats the highest-scoring diverging edition by `>= 0.15`).
  - Also appends `review_reasons` (triggering `needs_review`) when a supplied barcode wasn't confirmed, Discogs `data_quality` is poor, any track lacks a resolved performer, inferred disc count mismatches format quantity, or requested format tags aren't confirmed.
  - Computes coverage metrics: `duration_coverage`, `performer_coverage`, `writer_credit_coverage`, and `producer_credit_coverage`.

#### 6. Fallbacks, Track Enrichment & Worker Queues — [fallback_providers.py](pipeline/app/fallback_providers.py), [agent_client.py](pipeline/app/agent_client.py), [enricher.py](pipeline/app/enricher.py), [queue_worker.py](pipeline/app/queue_worker.py)

- **External & Gemini Fallbacks:** If Discogs misses or errors, `LookupService` tries deterministic catalog fallbacks (MusicBrainz, JPC, Deezer, iTunes) before falling back to optional Gemini + Google Search (`ENABLE_AGENT_FALLBACK`). If Gemini discovers a verified Discogs release URL, it bridges back to extract canonical tracks and credits.
- **5-Tier Track Enrichment (`TrackEnricher`):**
  - **Tier 0:** Deterministic album-artist inheritance onto unresolved compilation/single-act performers (`is_resolved_artist_name`) and raw duration formatting.
  - **Tier 1:** Sibling candidate releases and Discogs `master` CD/Digital/File versions with strict title alignment and `tracklist_hash` immutability.
  - **Tier 2a:** Album-level donor fill via MusicBrainz, Deezer (UPC then unquoted `"{artist} {title}"` search), and iTunes.
  - **Tier 2b:** Per-track compilation/studio fill via Deezer (`/search/track` using unquoted `"{artist} {title}"` + parenthetical-stripped base-title fallback while validating strictly against the original title, plus `/track/{id}` for ISRCs), iTunes, and MusicBrainz. Blocked automatically on live releases (checked across title, input title, formats, format tags, and `release.notes`).
  - **Tier 3 (opt-in `use_agent=True`):** Batched Gemini + Google Search in chunks of up to 25 missing tracks per request, accumulating token counts and cost onto `releases` (`enrichment_*`) and `agent_fallbacks`.
- **3-Pool PostgreSQL Worker Queue & Idle Auto-Enrichment (`QueueManager`):**
  - Runs dedicated `discogs-*`, `agent-*`, and `enrich-*` threads over `queued_discogs` -> `queued_agent` -> `queued_enrich`, recovering stale `checking_*` items cleanly after restarts and writing atomic JSON/Markdown batch reports (with `Confidence`, `Duration Cov`, `Performer Cov`, `Writer Cov`, and `ISRC Cov`) before flipping `queue_jobs.status` to `completed`.
  - When `ENABLE_IDLE_AUTO_ENRICHMENT=true` and `is_queue_idle()` is `True`, worker `enrich-1` sweeps up to `IDLE_ENRICHMENT_BATCH_SIZE` (`10`) un-enriched (`enriched_at IS NULL`) matched records every `IDLE_ENRICHMENT_INTERVAL_SEC` (`15.0`s) into a background `job_type="track_enrich_auto"` job (free Tiers 0–2b only, `use_agent=false`).

---

### Storage Layer — [init.sql](pipeline/sql/init.sql) & [repository.py](pipeline/app/repository.py)

`PipelineRepository` wraps a `psycopg_pool.ConnectionPool` to run idempotent upserts and queries across **13 PostgreSQL tables** (plus idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` migrations for provenance and enrichment billing):

| Table | Purpose |
| --- | --- |
| `input_records` | Stores raw and normalized request fields, `format_hint`, `is_compilation`, and unique `input_hash`. |
| `releases` | Stores canonical Discogs/fallback release metadata (`discogs_release_id` PK), `tracklist_hash`, `enriched_at`, `enrichment_*` Gemini token/cost counters, and `raw_json`. |
| `release_barcodes` | Indexes every normalized barcode (and UPC/EAN variant) per release for fast local lookups. |
| `tracks` | Ordered tracks (`disc_number`, `position`, `entry_type`, `parent_position`, `title`, `duration_sec`, `performing_artists_json`, `isrc`, per-field provenance columns, and `enrichment_notes_json`). |
| `track_credits` | Normalized per-track credits separated by `role_category` (`composer`, `lyricist`, `producer`, `featured`, `other`), `credit_source`, and `source_url`. |
| `input_matches` | Links an `input_record` to ranked candidate `releases` with scores, `is_selected`, and evaluation details. |
| `agent_fallbacks` | Stores Gemini agent answers, sources, `enriched_at`, and cumulative input/output token counts + `cost_usd`. |
| `agent_fallback_tracks` | Ordered tracks, durations, performers, credits (`credits_json`), ISRCs, and provenance fields extracted for Gemini agent matches. |
| `lookup_runs` | Append-only audit log of every lookup attempt (`status`, `http_code`, `source`, evaluation coverages, review reasons, `error_detail`). |
| `api_cache` | Caches raw HTTP `200` JSON responses from Discogs, MusicBrainz, Deezer, iTunes, and JPC keyed by SHA-256 of `(endpoint, params)`. |
| `record_reviews` | Stores human/agent `confirm`, `deny`, and `request_review` review decisions and audit history. |
| `queue_jobs` | Tracks async batch, lookup, retry, manual enrichment (`track_enrich`), and idle auto-enrichment (`track_enrich_auto`) jobs, counts, and report paths. |
| `queue_items` | Tracks per-row async work across `queued_discogs`, `checking_discogs`, `queued_agent`, `checking_agent`, `queued_enrich`, `checking_enrich`, `done`, and `cancelled`. |

```mermaid
erDiagram
    input_records ||--o{ lookup_runs : "logs runs"
    input_records ||--o{ input_matches : "ranks candidates"
    input_records ||--o{ record_reviews : "has reviews"
    queue_jobs ||--o{ queue_items : "contains items"
    releases ||--o{ input_matches : "matched as"
    releases ||--o{ lookup_runs : "selected in"
    releases ||--o{ release_barcodes : "has barcodes"
    releases ||--o{ tracks : "contains"
    tracks ||--o{ track_credits : "has credits"
    agent_fallbacks ||--o{ agent_fallback_tracks : "contains"
```

---

### Automated Test Suite — [tests/](pipeline/tests/)

The suite runs via `pytest` (**61 tests** across **10 modules** plus shared fixtures in [conftest.py](pipeline/tests/conftest.py)):

| Test File | Tests | What It Verifies |
| --- | --- | --- |
| [test_normalizer.py](pipeline/tests/test_normalizer.py) | 6 | Normalizes all 25 rows of [development.csv](../data/development.csv), barcode sanitization & UPC/EAN variants, artist un-inversion & compilations, `;` prefix/format/label-catno hints, `input_hash` stability, and `LookupRequest` validation. |
| [test_extractor.py](pipeline/tests/test_extractor.py) | 5 | Duration & role parsing, multi-disc headings/prefixes/vinyl sides/`"V1"` safety, `sub_tracks` credit & disc merging, release-level credit range selectors + unscoped fallback, and artist `join`/`anv`. |
| [test_evaluator.py](pipeline/tests/test_evaluator.py) | 5 | `matched` vs. `clear_winner` scoring, `needs_review` on diverging editions, empty-tracklist & barcode-collision rejections, label/catno matching, and quality/disc-mismatch/unconfirmed-barcode review flags. |
| [test_discogs_client.py](pipeline/tests/test_discogs_client.py) | 3 | Uses `httpx.MockTransport` to verify missing-token checks, required headers, `api_cache` hit/bypass, `429`/`5xx`/network retries, `401`/`404` handling, and invalid JSON errors. |
| [test_agent_client.py](pipeline/tests/test_agent_client.py) | 7 | JSON fence parsing, Discogs release URL extraction & dedup, track dedup/sorting, two-pass Gemini grounding mocks, token usage & cost calculation, and retry/error behavior. |
| [test_fallback_providers.py](pipeline/tests/test_fallback_providers.py) | 4 | Deterministic negative IDs, JPC multi-disc HTML parsing & caching, network error isolation, and iTunes album search pre-filtering. |
| [test_enricher.py](pipeline/tests/test_enricher.py) | 11 | Live/version qualifier guards, Discogs master/MusicBrainz/Deezer/iTunes enrichment, unquoted Deezer track & parenthetical base-title fallback, >25-track Gemini chunking & token/cost billing, `queued_enrich` recovery & cancellation, queued report live coverages & `Featured`/`Allow Gemini` UI, and idle-queue `track_enrich_auto` sweep. |
| [test_end_to_end.py](pipeline/tests/test_end_to_end.py) | 6 | Full PostgreSQL + FastAPI + CLI integration: real `DiscogsClient` + `api_cache` + UPC/EAN dedup, Tier 1 empty stub fallback to Tier 2/3, `404` TTL caching & expiry, offline DB fallback (`200`) vs. `502`/`503` outages, validation/route errors + degraded `/health`, and `/batch/csv` + CLI exit codes (`0`/`1`/`2`). |
| [test_agent_fallback_e2e.py](pipeline/tests/test_agent_fallback_e2e.py) | 7 | End-to-end external fallback ordering, Gemini agent DB cache hits, Discogs URL bridge, double failure handling, dashboard review workflows, and mixed batch/CLI execution. |
| [test_queue_e2e.py](pipeline/tests/test_queue_e2e.py) | 7 | Two-stage async batch queue, parallel agent workers, external fallback handoff & atomic report outputs, duplicate `409` guards, job cancellation, crash recovery, `/queue` UI, and queue CLI commands. |

---

## Understanding Outcomes & Source Labels

### Result Statuses (`status` & HTTP Code)

| Status | HTTP | Meaning |
| --- | --- | --- |
| `matched` | `200` | Candidate passed scoring (`>= 0.65`), tracklist is valid, and either all editions agree or one is a clear winner with no review flags. |
| `needs_review` | `200` | Best edition was selected, but a human should review it (e.g., diverging candidate tracklists, unconfirmed barcode, unresolved track performers, disc count mismatch, or poor Discogs `data_quality`). |
| `not_found` | `404` | No candidates returned or all candidates failed verification (empty tracklist, low similarity, or barcode collision). |
| `validation_error` | `422` | Input row had neither a usable non-zero barcode nor a non-empty album title. |
| `upstream_error` | `502` / `503` | Discogs failed (`401`/`403`/`5xx`/network → `502`; rate limit `429` → `503`) and no local DB backup existed. |

### Data Source Labels (`source`)

| Source Label | Meaning |
| --- | --- |
| `database` | Answered directly from PostgreSQL (exact `input_hash` match or cross-input local DB match) with 0 Discogs API calls. |
| `database_cache` | Returned a cached `404 not_found` because a prior identical search failed within `NOT_FOUND_TTL_HOURS` (default `24`h). |
| `database_fallback` | Live Discogs lookup failed, so a previously stored match was served safely (`fallback_used: true` + `warning`). |
| `discogs_cached` | Lookup ran through the search/extraction flow using cached Discogs payloads from PostgreSQL's `api_cache`. |
| `discogs_live` | Live HTTP calls were made to the Discogs API. |

---

## Quick-Start Commands

Run from the repository root (after putting your `DISCOGS_TOKEN` into `.env`):

```powershell
# 1. Start PostgreSQL + FastAPI (http://localhost:8000/docs for Swagger UI)
docker compose up -d --build

# 2. Check container health and run the automated test suite
curl.exe -s http://localhost:8000/health
docker compose exec -T api pytest -v

# 3. Look up a single record via CLI
docker compose exec api python -m app.cli lookup --barcode 4065629733131 --title Aspiral --artist EPICA

# 4. Run a batch on 5 rows from data/development.csv (writes JSON + Markdown to outputs/)
docker compose exec api python -m app.cli batch --csv /app/data/development.csv --rows 1,3,7,13,16 --output /app/outputs/dev5_report.json
```
