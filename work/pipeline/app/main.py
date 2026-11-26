import csv
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from psycopg_pool import ConnectionPool

from app.agent_client import (
    AgentFallbackProvider,
    AzureOpenAIAgentClient,
    GeminiAgentClient,
    build_agent_client,
)
from app.config import Settings, get_settings
from app.dashboard import (
    render_queue_dashboard,
    render_record_detail,
    render_results_table,
)
from app.discogs_client import DiscogsClient
from app.enricher import TrackEnricher, build_default_track_enricher
from app.extractor import extracted_track_to_out
from app.fallback_providers import FallbackProvider, build_default_fallback_providers
from app.models import (
    BatchRequest,
    BatchSummaryResponse,
    EnrichRecordResponse,
    EnrichRequest,
    EnqueueLookupRequest,
    HealthResponse,
    LookupNotFoundResponse,
    LookupRequest,
    LookupSuccessResponse,
    QueueJobDetailOut,
    QueueJobOut,
    QueueStatusOut,
    RecordDetailOut,
    RecordSummaryOut,
    ReleaseOut,
    RestartDemoRequest,
    ReviewOut,
    ReviewRequest,
)
from app.queue_worker import QueueManager
from app.repository import PipelineRepository
from app.service import LookupService, UpstreamServiceError


def _build_queue_status(
    repo: PipelineRepository, qm: QueueManager, service: LookupService
) -> QueueStatusOut:
    counts, active_items, pending_items, recent_jobs = repo.get_queue_status_snapshot()
    return QueueStatusOut(
        counts=counts,
        active_items=active_items,
        pending_items=pending_items,
        recent_jobs=recent_jobs,
        discogs_workers=qm.discogs_worker_count,
        agent_workers=qm.agent_worker_count,
        enrich_workers=qm.enrich_worker_count,
        workers_running=qm.is_running,
        agent_configured=service.has_agent_fallback(),
        idle_auto_enrichment_enabled=qm.idle_auto_enrichment_enabled,
    )


def _load_csv_queue_items(
    csv_path: str, rows: list[int] | None = None, force_refresh: bool = False
) -> tuple[Path, list[dict[str, Any]]]:
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_file}")

    with csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))

    selected_indices = (
        set(rows) if rows else set(range(1, len(all_rows) + 1))
    )
    items: list[dict[str, Any]] = []
    for idx, row in enumerate(all_rows, start=1):
        if idx not in selected_indices:
            continue
        items.append(
            {
                "row_number": idx,
                "barcode": str(row.get("barcode") or ""),
                "album_title": str(row.get("album_title") or ""),
                "album_artist": str(row.get("album_artist") or ""),
                "force_refresh": force_refresh,
                "mode": "full",
                "stage": "queued_discogs",
                "status_message": "Queued for Discogs",
            }
        )
    return csv_file, items


def create_app(
    settings: Settings | None = None,
    provider: Any | None = None,
    agent_provider: AgentFallbackProvider | None = None,
    fallback_providers: list[FallbackProvider] | None = None,
    enricher: TrackEnricher | None = None,
) -> FastAPI:
    cfg = settings or get_settings()
    restart_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pool = ConnectionPool(conninfo=cfg.database_url, min_size=1, max_size=20, open=True)
        repo = PipelineRepository(pool)
        repo.ensure_schema()

        client: DiscogsClient | None = None
        active_provider = provider
        if active_provider is None:
            client = DiscogsClient(cfg, cache_store=repo)
            active_provider = client

        own_agent_client: GeminiAgentClient | AzureOpenAIAgentClient | None = None
        active_agent_provider = agent_provider
        if active_agent_provider is None:
            own_agent_client = build_agent_client(cfg)
            active_agent_provider = own_agent_client

        own_fallbacks: list[FallbackProvider] = []
        if fallback_providers is not None:
            active_fallbacks = list(fallback_providers)
        elif provider is None:
            own_fallbacks = build_default_fallback_providers(cfg, cache_store=repo)
            active_fallbacks = own_fallbacks
        else:
            active_fallbacks = []

        own_enricher: TrackEnricher | None = None
        if enricher is not None:
            active_enricher: TrackEnricher | None = enricher
        elif provider is None and cfg.enable_track_enrichment:
            own_enricher = build_default_track_enricher(
                cfg,
                active_provider,
                cache_store=repo,
                agent_provider=active_agent_provider,
            )
            active_enricher = own_enricher
        else:
            active_enricher = None

        service = LookupService(
            repo=repo,
            provider=active_provider,
            settings=cfg,
            agent_provider=active_agent_provider,
            fallback_providers=active_fallbacks,
            enricher=active_enricher,
        )
        queue_manager = QueueManager(repo=repo, service=service, settings=cfg)

        app.state.settings = cfg
        app.state.pool = pool
        app.state.repo = repo
        app.state.service = service
        app.state.queue_manager = queue_manager

        if cfg.enable_queue_workers:
            queue_manager.start()

        try:
            yield
        finally:
            queue_manager.stop(timeout=5.0)
            if own_enricher is not None:
                own_enricher.close()
            for fb in own_fallbacks:
                fb.close()
            if own_agent_client is not None:
                own_agent_client.close()
            if client is not None:
                client.close()
            pool.close()

    app = FastAPI(
        title="Discogs Tracklist Discovery & Evaluation API",
        version="1.0.0",
        description=(
            "Discogs tracklist lookup with PostgreSQL 3-layer deduplication, "
            "per-track credit extraction, deterministic edition evaluation, and offline fallback."
        ),
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/results", response_class=HTMLResponse, include_in_schema=False)
    def results_dashboard(
        request: Request, limit: int = Query(default=100, ge=1, le=500)
    ) -> HTMLResponse:
        repo: PipelineRepository = request.app.state.repo
        records = repo.list_records(limit=limit)
        return HTMLResponse(content=render_results_table(records))

    @app.get("/results/{input_id}", response_class=HTMLResponse, include_in_schema=False)
    def record_detail_page(input_id: int, request: Request) -> HTMLResponse:
        repo: PipelineRepository = request.app.state.repo
        detail = repo.get_record_detail(input_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"Input record {input_id} not found.")
        return HTMLResponse(content=render_record_detail(detail))

    @app.get("/queue", response_class=HTMLResponse, include_in_schema=False)
    def queue_dashboard_page(request: Request) -> HTMLResponse:
        status = _build_queue_status(
            request.app.state.repo,
            request.app.state.queue_manager,
            request.app.state.service,
        )
        return HTMLResponse(content=render_queue_dashboard(status))

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        repo: PipelineRepository = request.app.state.repo
        s: Settings = request.app.state.settings
        try:
            db_ok = repo.ping()
        except Exception:
            db_ok = False
        token = (s.discogs_token or "").strip()
        token_configured = bool(token and token != "PASTE_YOUR_DISCOGS_PERSONAL_ACCESS_TOKEN")
        return HealthResponse(
            status="ok" if db_ok else "degraded",
            database_ok=db_ok,
            discogs_token_configured=token_configured,
            gemini_agent_configured=s.is_agent_configured(),
        )

    @app.post(
        "/lookup",
        response_model=LookupSuccessResponse,
        responses={
            404: {"model": LookupNotFoundResponse, "description": "Release not found or rejected"},
            502: {"description": "Upstream Discogs auth/server error with no DB fallback"},
            503: {"description": "Upstream Discogs rate limit (429) with no DB fallback"},
        },
    )
    def lookup(payload: LookupRequest, request: Request):
        service: LookupService = request.app.state.service
        try:
            result = service.lookup(payload, source_tag="api")
        except UpstreamServiceError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "status": "upstream_error",
                    "http_code": exc.status_code,
                    "error_type": exc.error_type,
                    "detail": exc.detail,
                },
            )

        if isinstance(result, LookupNotFoundResponse):
            return JSONResponse(status_code=404, content=result.model_dump())
        return result

    @app.post("/batch/csv", response_model=BatchSummaryResponse)
    def batch_csv(payload: BatchRequest, request: Request) -> BatchSummaryResponse:
        service: LookupService = request.app.state.service
        try:
            return service.run_batch(payload)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/jobs/batch", status_code=202, response_model=QueueJobOut)
    def enqueue_batch_job(payload: BatchRequest, request: Request) -> QueueJobOut:
        csv_file, items = _load_csv_queue_items(
            payload.csv_path, rows=payload.rows, force_refresh=payload.force_refresh
        )

        repo: PipelineRepository = request.app.state.repo
        qm: QueueManager = request.app.state.queue_manager
        job = repo.create_queue_job(
            job_type="batch_csv",
            label=f"Batch: {csv_file.name}",
            items=items,
            csv_path=payload.csv_path,
            output_path=payload.output_path,
            force_refresh=payload.force_refresh,
        )
        if items:
            qm.notify_discogs()
        return job

    @app.post("/jobs/restart-demo", status_code=202, response_model=QueueJobOut)
    def restart_demo_batch(
        request: Request,
        payload: RestartDemoRequest | None = Body(default=None),
    ) -> QueueJobOut:
        req = payload or RestartDemoRequest()
        csv_file, items = _load_csv_queue_items(
            req.csv_path, rows=None, force_refresh=req.clear_api_cache
        )
        if not items:
            raise HTTPException(status_code=400, detail="CSV file contains no rows.")

        repo: PipelineRepository = request.app.state.repo
        qm: QueueManager = request.app.state.queue_manager
        if not restart_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Restart already in progress.")

        try:
            was_running = qm.is_running
            if was_running:
                stopped = qm.stop(timeout=15.0)
                if not stopped:
                    raise HTTPException(
                        status_code=409,
                        detail="Workers are still finishing an active upstream request; please try again in a few seconds.",
                    )

            repo.reset_pipeline_state(clear_api_cache=req.clear_api_cache)
            job = repo.create_queue_job(
                job_type="batch_csv",
                label=f"Demo Restart: {csv_file.name} ({len(items)} songs)",
                items=items,
                csv_path=req.csv_path,
                output_path=req.output_path,
                force_refresh=req.clear_api_cache,
            )
            if was_running:
                qm.start()
                qm.notify_discogs()
            return job
        finally:
            restart_lock.release()

    @app.post("/jobs/lookup", status_code=202, response_model=QueueJobDetailOut)
    def enqueue_lookup_job(
        payload: EnqueueLookupRequest, request: Request
    ) -> QueueJobDetailOut:
        repo: PipelineRepository = request.app.state.repo
        service: LookupService = request.app.state.service
        qm: QueueManager = request.app.state.queue_manager
        if payload.mode == "agent_only" and not service.has_agent_fallback():
            raise HTTPException(
                status_code=400,
                detail="Gemini agent fallback is not configured or disabled.",
            )

        stage = "queued_agent" if payload.mode == "agent_only" else "queued_discogs"
        msg = (
            "Queued for Gemini agent"
            if payload.mode == "agent_only"
            else "Queued for Discogs"
        )
        label = (
            f"{payload.album_artist} - {payload.album_title}".strip(" -")
            or payload.barcode
        )
        job = repo.create_queue_job(
            job_type="single_lookup",
            label=label,
            items=[
                {
                    "row_number": 1,
                    "barcode": payload.barcode,
                    "album_title": payload.album_title,
                    "album_artist": payload.album_artist,
                    "force_refresh": payload.force_refresh,
                    "mode": payload.mode,
                    "stage": stage,
                    "status_message": msg,
                }
            ],
            force_refresh=payload.force_refresh,
        )
        if payload.mode == "agent_only":
            qm.notify_agent()
        else:
            qm.notify_discogs()
        detail = repo.get_queue_job(job.id)
        assert detail is not None
        return detail

    @app.post("/jobs/agent-retry/{input_id}", status_code=202, response_model=QueueJobDetailOut)
    def enqueue_agent_retry(input_id: int, request: Request) -> QueueJobDetailOut:
        repo: PipelineRepository = request.app.state.repo
        service: LookupService = request.app.state.service
        qm: QueueManager = request.app.state.queue_manager

        rec = repo.get_input_record(input_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"Input record {input_id} not found.")
        if not service.has_agent_fallback():
            raise HTTPException(
                status_code=400,
                detail="Gemini agent fallback is not configured or disabled.",
            )
        if repo.has_active_queue_item(input_id):
            raise HTTPException(
                status_code=409,
                detail=f"Input record {input_id} already has an active queued/running job.",
            )

        label = (
            f"Agent retry #{input_id}: {rec['album_artist_raw']} - {rec['album_title_raw']}"
        ).strip()
        job = repo.create_queue_job(
            job_type="agent_retry",
            label=label,
            items=[
                {
                    "row_number": 1,
                    "input_id": input_id,
                    "barcode": rec["barcode_raw"],
                    "album_title": rec["album_title_raw"],
                    "album_artist": rec["album_artist_raw"],
                    "force_refresh": True,
                    "mode": "agent_only",
                    "stage": "queued_agent",
                    "status_message": "Queued for Gemini agent retry",
                }
            ],
            force_refresh=True,
        )
        qm.notify_agent()
        detail = repo.get_queue_job(job.id)
        assert detail is not None
        return detail

    @app.post("/jobs/retry-not-found", status_code=202, response_model=QueueJobOut)
    def enqueue_retry_not_found(request: Request) -> QueueJobOut:
        repo: PipelineRepository = request.app.state.repo
        service: LookupService = request.app.state.service
        qm: QueueManager = request.app.state.queue_manager

        if not service.has_agent_fallback():
            raise HTTPException(
                status_code=400,
                detail="Gemini agent fallback is not configured or disabled.",
            )

        records = repo.list_not_found_records_for_agent_retry()
        items = [
            {
                "row_number": idx,
                "input_id": int(rec["id"]),
                "barcode": rec["barcode_raw"],
                "album_title": rec["album_title_raw"],
                "album_artist": rec["album_artist_raw"],
                "force_refresh": True,
                "mode": "agent_only",
                "stage": "queued_agent",
                "status_message": "Queued for Gemini agent retry",
            }
            for idx, rec in enumerate(records, start=1)
        ]
        job = repo.create_queue_job(
            job_type="agent_retry",
            label=f"Retry not_found ({len(items)} records)",
            items=items,
            force_refresh=True,
        )
        if items:
            qm.notify_agent()
        return job

    @app.post("/jobs/enrich/{input_id}", status_code=202, response_model=QueueJobDetailOut)
    def enqueue_enrich_record(
        input_id: int,
        request: Request,
        payload: EnrichRequest | None = Body(default=None),
    ) -> QueueJobDetailOut:
        req = payload or EnrichRequest()
        repo: PipelineRepository = request.app.state.repo
        service: LookupService = request.app.state.service
        qm: QueueManager = request.app.state.queue_manager

        rec = repo.get_input_record(input_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"Input record {input_id} not found.")
        latest = repo.find_latest_run(input_id)
        if latest is None or latest["status"] not in ("matched", "needs_review"):
            raise HTTPException(
                status_code=400,
                detail=f"Input record {input_id} has no matched/needs_review result to enrich.",
            )
        if (
            latest.get("selected_release_id") is None
            and latest.get("selected_agent_fallback_id") is None
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Input record {input_id} has no selected release or agent fallback.",
            )
        if req.use_agent and not service.has_agent_fallback():
            raise HTTPException(
                status_code=400,
                detail="Gemini agent fallback is not configured or disabled.",
            )
        if repo.has_active_queue_item(input_id):
            raise HTTPException(
                status_code=409,
                detail=f"Input record {input_id} already has an active queued/running job.",
            )

        label = (
            f"Enrich #{input_id}: {rec['album_artist_raw']} - {rec['album_title_raw']}"
        ).strip()
        job = repo.create_queue_job(
            job_type="track_enrich",
            label=label,
            items=[
                {
                    "row_number": 1,
                    "input_id": input_id,
                    "barcode": rec["barcode_raw"],
                    "album_title": rec["album_title_raw"],
                    "album_artist": rec["album_artist_raw"],
                    "force_refresh": req.force_refresh,
                    "mode": "enrich",
                    "stage": "queued_enrich",
                    "status_message": "Queued for track enrichment",
                    "context_json": {
                        "use_agent": req.use_agent,
                        "force_refresh": req.force_refresh,
                        "result_status": latest["status"],
                        "result_source": latest["source"],
                    },
                }
            ],
            force_refresh=req.force_refresh,
        )
        qm.notify_enrich()
        detail = repo.get_queue_job(job.id)
        assert detail is not None
        return detail

    @app.post("/jobs/enrich-missing", status_code=202, response_model=QueueJobOut)
    def enqueue_enrich_missing(
        request: Request,
        limit: int = Query(default=100, ge=1, le=500),
        payload: EnrichRequest | None = Body(default=None),
    ) -> QueueJobOut:
        req = payload or EnrichRequest()
        repo: PipelineRepository = request.app.state.repo
        service: LookupService = request.app.state.service
        qm: QueueManager = request.app.state.queue_manager

        if req.use_agent and not service.has_agent_fallback():
            raise HTTPException(
                status_code=400,
                detail="Gemini agent fallback is not configured or disabled.",
            )

        records = repo.list_matched_inputs_with_missing_track_data(
            limit=limit, force_refresh=req.force_refresh
        )
        items = [
            {
                "row_number": idx,
                "input_id": int(rec["id"]),
                "barcode": rec["barcode_raw"],
                "album_title": rec["album_title_raw"],
                "album_artist": rec["album_artist_raw"],
                "force_refresh": req.force_refresh,
                "mode": "enrich",
                "stage": "queued_enrich",
                "status_message": "Queued for track enrichment",
                "context_json": {
                    "use_agent": req.use_agent,
                    "force_refresh": req.force_refresh,
                    "result_status": rec["latest_status"],
                    "result_source": rec["latest_source"],
                },
            }
            for idx, rec in enumerate(records, start=1)
        ]
        job = repo.create_queue_job(
            job_type="track_enrich",
            label=f"Enrich missing ({len(items)} records)",
            items=items,
            force_refresh=req.force_refresh,
        )
        if items:
            qm.notify_enrich()
        return job

    @app.post("/records/{input_id}/enrich", response_model=EnrichRecordResponse)
    def enrich_record_sync(
        input_id: int,
        request: Request,
        payload: EnrichRequest | None = Body(default=None),
    ) -> EnrichRecordResponse:
        req = payload or EnrichRequest()
        service: LookupService = request.app.state.service
        if req.use_agent and not service.has_agent_fallback():
            raise HTTPException(
                status_code=400,
                detail="Gemini agent fallback is not configured or disabled.",
            )
        try:
            res = service.enrich_record(
                input_id, use_agent=req.use_agent, force_refresh=req.force_refresh
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if res is None:
            raise HTTPException(status_code=404, detail=f"Input record {input_id} not found.")
        return res

    @app.get("/jobs/status", response_model=QueueStatusOut)
    def get_queue_status(request: Request) -> QueueStatusOut:
        return _build_queue_status(
            request.app.state.repo,
            request.app.state.queue_manager,
            request.app.state.service,
        )

    @app.get("/jobs", response_model=list[QueueJobOut])
    def list_jobs(
        request: Request,
        limit: int = Query(default=20, ge=1, le=200),
        status: str | None = Query(default=None),
    ) -> list[QueueJobOut]:
        repo: PipelineRepository = request.app.state.repo
        return repo.list_queue_jobs(limit=limit, status=status)

    @app.get("/jobs/{job_id}", response_model=QueueJobDetailOut)
    def get_job_detail(job_id: int, request: Request) -> QueueJobDetailOut:
        repo: PipelineRepository = request.app.state.repo
        detail = repo.get_queue_job(job_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
        return detail

    @app.post("/jobs/{job_id}/cancel", response_model=QueueJobOut)
    def cancel_job(job_id: int, request: Request) -> QueueJobOut:
        repo: PipelineRepository = request.app.state.repo
        job = repo.cancel_queue_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
        return job

    @app.get("/records", response_model=list[RecordSummaryOut])
    def list_records(
        request: Request,
        limit: int = Query(default=100, ge=1, le=500),
        status: str | None = Query(default=None),
    ) -> list[RecordSummaryOut]:
        repo: PipelineRepository = request.app.state.repo
        return repo.list_records(limit=limit, status=status)

    @app.get("/records/{input_id}", response_model=RecordDetailOut)
    def get_record(input_id: int, request: Request) -> RecordDetailOut:
        repo: PipelineRepository = request.app.state.repo
        detail = repo.get_record_detail(input_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"Input record {input_id} not found.")
        return detail

    @app.post("/records/{input_id}/review", response_model=ReviewOut)
    def review_record(
        input_id: int, payload: ReviewRequest, request: Request
    ) -> ReviewOut:
        repo: PipelineRepository = request.app.state.repo
        try:
            review = repo.record_review(input_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if review is None:
            raise HTTPException(status_code=404, detail=f"Input record {input_id} not found.")
        return review

    @app.get("/releases/{discogs_release_id}")
    def get_release(discogs_release_id: int, request: Request) -> dict[str, Any]:
        repo: PipelineRepository = request.app.state.repo
        rel = repo.get_extracted_release(discogs_release_id)
        if rel is None:
            raise HTTPException(
                status_code=404, detail=f"Release {discogs_release_id} not found in database."
            )
        return {
            "release": ReleaseOut(
                discogs_release_id=rel.discogs_release_id,
                master_id=rel.master_id,
                url=rel.url,
                title=rel.title,
                artist_display=rel.artist_display,
                year=rel.year,
                country=rel.country,
                formats=rel.formats,
                identifiers=rel.identifiers,
                labels=rel.labels,
                data_quality=rel.data_quality,
                community_status=rel.community_status,
                community_have=rel.community_have,
                tracklist_hash=rel.tracklist_hash,
                unscoped_credits=rel.unscoped_credits,
            ).model_dump(),
            "tracks": [extracted_track_to_out(t).model_dump() for t in rel.tracks],
        }

    return app


app = create_app()
