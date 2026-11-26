import logging
import threading
import time
from typing import Any

from pydantic import ValidationError

from app.config import Settings
from app.models import (
    LookupNotFoundResponse,
    LookupRequest,
    LookupSuccessResponse,
)
from app.repository import PipelineRepository
from app.service import DiscogsHandoff, LookupService, UpstreamServiceError

logger = logging.getLogger(__name__)


class QueueManager:
    """PostgreSQL-backed multi-stage worker pools for Discogs, Gemini Agent, and Enrichment."""

    def __init__(
        self,
        repo: PipelineRepository,
        service: LookupService,
        settings: Settings,
    ) -> None:
        self._repo = repo
        self._service = service
        self._settings = settings
        self._stop_event = threading.Event()
        self._discogs_wakeup = threading.Event()
        self._agent_wakeup = threading.Event()
        self._enrich_wakeup = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lifecycle_lock = threading.Lock()
        self._idle_lock = threading.Lock()
        self._last_idle_sweep = 0.0
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def idle_auto_enrichment_enabled(self) -> bool:
        return bool(
            self._settings.enable_track_enrichment
            and self._settings.enable_idle_auto_enrichment
        )

    @property
    def discogs_worker_count(self) -> int:
        return max(1, self._settings.discogs_queue_concurrency)

    @property
    def agent_worker_count(self) -> int:
        return max(1, self._settings.agent_queue_concurrency)

    @property
    def enrich_worker_count(self) -> int:
        return max(1, self._settings.enrich_queue_concurrency)

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._running:
                return
            recovered = self._repo.recover_stale_queue_items()
            if recovered:
                logger.info("Recovered %s stale queue item(s) from previous run", recovered)

            self._stop_event.clear()
            self._discogs_wakeup.clear()
            self._agent_wakeup.clear()
            self._enrich_wakeup.clear()
            self._last_idle_sweep = time.monotonic()
            self._threads = []

            for idx in range(1, self.discogs_worker_count + 1):
                name = f"discogs-{idx}"
                t = threading.Thread(
                    target=self._discogs_loop,
                    args=(name,),
                    name=name,
                    daemon=True,
                )
                self._threads.append(t)
                t.start()

            for idx in range(1, self.agent_worker_count + 1):
                name = f"agent-{idx}"
                t = threading.Thread(
                    target=self._agent_loop,
                    args=(name,),
                    name=name,
                    daemon=True,
                )
                self._threads.append(t)
                t.start()

            for idx in range(1, self.enrich_worker_count + 1):
                name = f"enrich-{idx}"
                t = threading.Thread(
                    target=self._enrich_loop,
                    args=(name,),
                    name=name,
                    daemon=True,
                )
                self._threads.append(t)
                t.start()

            self._running = True
            logger.info(
                "QueueManager started (%s Discogs, %s Agent, %s Enrich worker(s))",
                self.discogs_worker_count,
                self.agent_worker_count,
                self.enrich_worker_count,
            )

    def stop(self, timeout: float = 5.0) -> bool:
        with self._lifecycle_lock:
            if not self._running:
                return True
            self._stop_event.set()
            self._discogs_wakeup.set()
            self._agent_wakeup.set()
            self._enrich_wakeup.set()
            threads = list(self._threads)
            self._running = False

        deadline = time.monotonic() + max(0.1, float(timeout))
        for t in threads:
            remaining = max(0.0, deadline - time.monotonic())
            t.join(timeout=remaining)

        alive = [t for t in threads if t.is_alive()]
        with self._lifecycle_lock:
            self._threads = alive
        if alive:
            logger.warning(
                "QueueManager stop timed out with %s worker thread(s) still running: %s",
                len(alive),
                [t.name for t in alive],
            )
            return False
        logger.info("QueueManager stopped")
        return True

    def notify_discogs(self) -> None:
        self._discogs_wakeup.set()

    def notify_agent(self) -> None:
        self._agent_wakeup.set()

    def notify_enrich(self) -> None:
        self._enrich_wakeup.set()

    def notify_all(self) -> None:
        self._discogs_wakeup.set()
        self._agent_wakeup.set()
        self._enrich_wakeup.set()

    def _discogs_loop(self, worker_name: str) -> None:
        poll_sec = max(0.05, float(self._settings.queue_poll_interval_sec))
        while not self._stop_event.is_set():
            try:
                item = self._repo.claim_next_queue_item(
                    from_stage="queued_discogs",
                    to_stage="checking_discogs",
                    worker_name=worker_name,
                    status_message="Starting Discogs lookup",
                )
            except Exception:
                logger.exception("[%s] Error claiming Discogs queue item", worker_name)
                self._stop_event.wait(poll_sec)
                continue

            if item is None:
                self._discogs_wakeup.wait(timeout=poll_sec)
                self._discogs_wakeup.clear()
                continue

            self._process_discogs_item(worker_name, item)

    def _agent_loop(self, worker_name: str) -> None:
        poll_sec = max(0.05, float(self._settings.queue_poll_interval_sec))
        while not self._stop_event.is_set():
            try:
                item = self._repo.claim_next_queue_item(
                    from_stage="queued_agent",
                    to_stage="checking_agent",
                    worker_name=worker_name,
                    status_message="Starting Gemini agent lookup",
                )
            except Exception:
                logger.exception("[%s] Error claiming Agent queue item", worker_name)
                self._stop_event.wait(poll_sec)
                continue

            if item is None:
                self._agent_wakeup.wait(timeout=poll_sec)
                self._agent_wakeup.clear()
                continue

            self._process_agent_item(worker_name, item)

    def _enrich_loop(self, worker_name: str) -> None:
        poll_sec = max(0.05, float(self._settings.queue_poll_interval_sec))
        while not self._stop_event.is_set():
            try:
                item = self._repo.claim_next_queue_item(
                    from_stage="queued_enrich",
                    to_stage="checking_enrich",
                    worker_name=worker_name,
                    status_message="Starting track enrichment",
                )
            except Exception:
                logger.exception("[%s] Error claiming Enrich queue item", worker_name)
                self._stop_event.wait(poll_sec)
                continue

            if item is None:
                if worker_name == "enrich-1":
                    try:
                        if self.maybe_enqueue_idle_enrichment() > 0:
                            continue
                    except Exception:
                        logger.exception(
                            "[%s] Error during idle auto-enrichment sweep",
                            worker_name,
                        )
                self._enrich_wakeup.wait(timeout=poll_sec)
                self._enrich_wakeup.clear()
                continue

            self._process_enrich_item(worker_name, item)

    def maybe_enqueue_idle_enrichment(self, force_now: bool = False) -> int:
        """Enqueue un-enriched matched records for TrackEnricher when all queues are idle."""
        if not self.idle_auto_enrichment_enabled:
            return 0
        if not self._service.has_track_enricher():
            return 0
        interval = max(0.05, float(self._settings.idle_enrichment_interval_sec))
        with self._idle_lock:
            now = time.monotonic()
            if not force_now and (now - self._last_idle_sweep) < interval:
                return 0
            self._last_idle_sweep = now

            if not self._repo.is_queue_idle():
                return 0

            batch_size = max(1, int(self._settings.idle_enrichment_batch_size))
            records = self._repo.list_matched_inputs_with_missing_track_data(
                limit=batch_size, force_refresh=False
            )
            if not records:
                return 0

            items = [
                {
                    "row_number": idx,
                    "input_id": int(rec["id"]),
                    "barcode": rec["barcode_raw"],
                    "album_title": rec["album_title_raw"],
                    "album_artist": rec["album_artist_raw"],
                    "force_refresh": False,
                    "mode": "enrich",
                    "stage": "queued_enrich",
                    "status_message": "Auto-queued during queue idle",
                    "context_json": {
                        "use_agent": (
                            self._settings.enable_agent_track_enrichment
                            and self._service.has_agent_fallback()
                        ),
                        "force_refresh": False,
                        "result_status": rec["latest_status"],
                        "result_source": rec["latest_source"],
                    },
                }
                for idx, rec in enumerate(records, start=1)
            ]
            self._repo.create_queue_job(
                job_type="track_enrich_auto",
                label=f"Auto-enrich idle ({len(items)} records)",
                items=items,
                force_refresh=False,
            )
            logger.info(
                "Idle queue auto-enrichment enqueued %s un-enriched record(s)",
                len(items),
            )
            self.notify_enrich()
            return len(items)

    def _should_handoff_to_enrich(
        self, res: LookupSuccessResponse, force_refresh: bool
    ) -> bool:
        if not self._service.has_track_enricher():
            return False
        if not self._service.needs_track_enrichment(res.tracks):
            return False
        if not force_refresh and res.source == "database":
            rel_id = res.release.discogs_release_id if res.release else None
            fb_id = res.agent_result.agent_fallback_id if res.agent_result else None
            if self._repo.is_target_already_enriched(rel_id, fb_id):
                return False
        return True

    def _complete_or_handoff_to_enrich(
        self,
        worker_name: str,
        item_id: int,
        job_id: int,
        res: LookupSuccessResponse | LookupNotFoundResponse,
        force_refresh: bool,
    ) -> bool:
        if isinstance(res, LookupSuccessResponse) and self._should_handoff_to_enrich(
            res, force_refresh
        ):
            rel_id = res.release.discogs_release_id if res.release else None
            fb_id = res.agent_result.agent_fallback_id if res.agent_result else None
            ctx = {
                "use_agent": (
                    self._settings.enable_agent_track_enrichment
                    and self._service.has_agent_fallback()
                ),
                "force_refresh": force_refresh,
                "result_status": res.status,
                "result_source": res.source,
                "warning": res.warning or "",
            }
            handed_off, just_finished = self._repo.handoff_item_to_enrich(
                item_id=item_id,
                job_id=job_id,
                input_id=res.input_id,
                selected_release_id=rel_id,
                selected_agent_fallback_id=fb_id,
                track_count=len(res.tracks),
                context=ctx,
                status_message=f"Matched via {res.source}; queued for track enrichment",
            )
            if handed_off:
                logger.info(
                    "[%s] Handed off item #%s (job #%s) -> queued_enrich",
                    worker_name,
                    item_id,
                    job_id,
                )
                self.notify_enrich()
                return False
            return just_finished
        return self._complete_from_response(item_id, job_id, res)

    def _finalize_job_report(self, job_id: int) -> None:
        try:
            detail = self._repo.get_queue_job(job_id)
            if detail is None or not detail.job.output_path:
                return
            paths = self._service.write_job_reports(detail)
            if paths is not None:
                self._repo.set_queue_job_report_paths(
                    job_id, paths[0], paths[1], mark_completed=True
                )
                logger.info("Job #%s wrote reports to %s and %s", job_id, paths[0], paths[1])
            else:
                self._repo.set_queue_job_report_paths(job_id, "", "", mark_completed=True)
        except Exception:
            logger.exception("Failed writing reports for completed job #%s", job_id)
            try:
                self._repo.set_queue_job_report_paths(job_id, "", "", mark_completed=True)
            except Exception:
                logger.exception("Failed marking job #%s completed after report error", job_id)

    def _complete_from_response(
        self,
        item_id: int,
        job_id: int,
        res: LookupSuccessResponse | LookupNotFoundResponse,
    ) -> bool:
        if isinstance(res, LookupSuccessResponse):
            rel_id = res.release.discogs_release_id if res.release else None
            fb_id = res.agent_result.agent_fallback_id if res.agent_result else None
            return self._repo.complete_queue_item(
                item_id=item_id,
                job_id=job_id,
                input_id=res.input_id,
                result_status=res.status,
                result_http_code=200,
                result_source=res.source,
                selected_release_id=rel_id,
                selected_agent_fallback_id=fb_id,
                track_count=len(res.tracks),
                status_message=f"Completed via {res.source} ({res.status})",
                error_detail=res.warning or "",
            )

        return self._repo.complete_queue_item(
            item_id=item_id,
            job_id=job_id,
            input_id=res.input_id,
            result_status="not_found",
            result_http_code=404,
            result_source=res.source,
            track_count=0,
            status_message=f"Not found ({res.source})",
            error_detail="; ".join(res.rejection_reasons[:3]),
        )

    def _process_discogs_item(self, worker_name: str, item: dict[str, Any]) -> None:
        item_id = int(item["id"])
        job_id = int(item["job_id"])
        barcode = str(item.get("barcode") or "")
        title = str(item.get("album_title") or "")
        artist = str(item.get("album_artist") or "")
        logger.info(
            "[%s] Claimed item #%s (job #%s): %s - %s",
            worker_name,
            item_id,
            job_id,
            artist,
            title,
        )

        try:
            req = LookupRequest(
                barcode=barcode,
                album_title=title,
                album_artist=artist,
                force_refresh=bool(item.get("force_refresh", False)),
            )
        except ValidationError as exc:
            just_finished = self._repo.complete_queue_item(
                item_id=item_id,
                job_id=job_id,
                input_id=item.get("input_id"),
                result_status="validation_error",
                result_http_code=422,
                result_source="input_validation",
                status_message="Validation error",
                error_detail=str(exc),
            )
            if just_finished:
                self._finalize_job_report(job_id)
            return

        def _progress(msg: str, attempt: int | None) -> None:
            self._repo.update_queue_item_progress(item_id, msg, attempt=attempt)

        try:
            outcome = self._service.run_discogs_stage(
                req,
                source_tag=f"queue_job_{job_id}_row_{item['row_number']}",
                progress_cb=_progress,
            )
            if outcome.completed is not None:
                just_finished = self._complete_or_handoff_to_enrich(
                    worker_name, item_id, job_id, outcome.completed, req.force_refresh
                )
                if just_finished:
                    self._finalize_job_report(job_id)
                return

            assert outcome.handoff is not None
            handoff = outcome.handoff
            self._repo.update_queue_item_progress(
                item_id, "Checking external fallbacks", input_id=handoff.input_id
            )
            fb_resp = self._service.run_external_fallback_stage(
                handoff,
                force_refresh=req.force_refresh,
                progress_cb=_progress,
            )
            if fb_resp is not None:
                just_finished = self._complete_or_handoff_to_enrich(
                    worker_name, item_id, job_id, fb_resp, req.force_refresh
                )
                if just_finished:
                    self._finalize_job_report(job_id)
                return

            if self._service.has_agent_fallback():
                msg = (
                    f"Discogs failed ({handoff.discogs_error_http_code}); queued for Gemini agent"
                    if handoff.discogs_error_http_code is not None
                    else "Discogs missed; queued for Gemini agent"
                )
                handed_off, just_finished = self._repo.handoff_item_to_agent(
                    item_id=item_id,
                    job_id=job_id,
                    input_id=handoff.input_id,
                    context=handoff.to_context_dict(),
                    status_message=msg,
                )
                if handed_off:
                    logger.info(
                        "[%s] Handed off item #%s (job #%s) -> queued_agent",
                        worker_name,
                        item_id,
                        job_id,
                    )
                    self.notify_agent()
                elif just_finished:
                    self._finalize_job_report(job_id)
                return

            try:
                nf_resp = self._service.finalize_without_agent(handoff)
                just_finished = self._complete_from_response(item_id, job_id, nf_resp)
            except UpstreamServiceError as exc:
                just_finished = self._repo.complete_queue_item(
                    item_id=item_id,
                    job_id=job_id,
                    input_id=handoff.input_id,
                    result_status="upstream_error",
                    result_http_code=exc.status_code,
                    result_source="discogs_live",
                    status_message=f"Discogs error ({exc.status_code})",
                    error_detail=exc.detail,
                )
            if just_finished:
                self._finalize_job_report(job_id)
        except Exception as exc:
            logger.exception("[%s] Unexpected failure on item #%s", worker_name, item_id)
            just_finished = self._repo.complete_queue_item(
                item_id=item_id,
                job_id=job_id,
                input_id=item.get("input_id"),
                result_status="upstream_error",
                result_http_code=500,
                result_source="worker_error",
                status_message="Worker error",
                error_detail=str(exc),
            )
            if just_finished:
                self._finalize_job_report(job_id)

    def _process_agent_item(self, worker_name: str, item: dict[str, Any]) -> None:
        item_id = int(item["id"])
        job_id = int(item["job_id"])
        mode = str(item.get("mode") or "full")
        logger.info(
            "[%s] Claimed agent item #%s (job #%s, mode=%s): %s - %s",
            worker_name,
            item_id,
            job_id,
            mode,
            item.get("album_artist") or "",
            item.get("album_title") or "",
        )

        def _progress(msg: str, attempt: int | None) -> None:
            self._repo.update_queue_item_progress(item_id, msg, attempt=attempt)

        try:
            force_refresh = bool(item.get("force_refresh", False))
            if mode == "agent_only":
                force_refresh = bool(item.get("force_refresh", True))
                res = self._service.run_agent_only_for_input(
                    barcode=str(item.get("barcode") or ""),
                    album_title=str(item.get("album_title") or ""),
                    album_artist=str(item.get("album_artist") or ""),
                    input_id=item.get("input_id"),
                    force_refresh=force_refresh,
                    source_tag=f"queue_job_{job_id}_row_{item['row_number']}",
                    progress_cb=_progress,
                )
            else:
                handoff = DiscogsHandoff.from_context_dict(item.get("context_json") or {})
                res = self._service.run_agent_stage(
                    handoff,
                    force_refresh=force_refresh,
                    progress_cb=_progress,
                )

            self._repo.update_queue_item_progress(
                item_id, f"Finished agent stage ({res.status})", input_id=res.input_id
            )
            just_finished = self._complete_or_handoff_to_enrich(
                worker_name, item_id, job_id, res, force_refresh
            )
            logger.info(
                "[%s] Finished agent item #%s (job #%s): status=%s source=%s",
                worker_name,
                item_id,
                job_id,
                res.status,
                res.source,
            )
        except UpstreamServiceError as exc:
            just_finished = self._repo.complete_queue_item(
                item_id=item_id,
                job_id=job_id,
                input_id=item.get("input_id"),
                result_status="upstream_error",
                result_http_code=exc.status_code,
                result_source="discogs_live",
                status_message=f"Discogs & agent failed ({exc.status_code})",
                error_detail=exc.detail,
            )
        except Exception as exc:
            logger.exception("[%s] Unexpected failure on agent item #%s", worker_name, item_id)
            just_finished = self._repo.complete_queue_item(
                item_id=item_id,
                job_id=job_id,
                input_id=item.get("input_id"),
                result_status="upstream_error",
                result_http_code=500,
                result_source="worker_error",
                status_message="Agent worker error",
                error_detail=str(exc),
            )

        if just_finished:
            self._finalize_job_report(job_id)

    def _process_enrich_item(self, worker_name: str, item: dict[str, Any]) -> None:
        item_id = int(item["id"])
        job_id = int(item["job_id"])
        input_id = item.get("input_id")
        ctx = item.get("context_json") or {}
        use_agent = bool(ctx.get("use_agent", self._settings.enable_agent_track_enrichment))
        force_refresh = bool(ctx.get("force_refresh", item.get("force_refresh", False)))
        logger.info(
            "[%s] Claimed enrich item #%s (job #%s, input_id=%s, use_agent=%s)",
            worker_name,
            item_id,
            job_id,
            input_id,
            use_agent,
        )

        def _progress(msg: str, attempt: int | None) -> None:
            self._repo.update_queue_item_progress(item_id, msg, attempt=attempt)

        try:
            if input_id is None:
                raise ValueError("Queue enrich item is missing input_id.")
            res = self._service.enrich_record(
                int(input_id),
                use_agent=use_agent,
                force_refresh=force_refresh,
                progress_cb=_progress,
            )
            if res is None:
                raise ValueError(f"Input record {input_id} not found.")

            latest = self._repo.find_latest_run(int(input_id))
            final_status = (
                (latest or {}).get("status")
                or str(ctx.get("result_status") or "matched")
            )
            final_source = str(
                ctx.get("result_source") or (latest or {}).get("source") or "track_enrich"
            )
            rel_id = res.target_id if res.target_type == "release" else None
            fb_id = res.target_id if res.target_type == "agent_fallback" else None
            prov_str = ", ".join(res.providers_used) if res.providers_used else "none"
            status_msg = (
                f"Enriched (+{res.durations_filled} dur, +{res.performers_filled} art, "
                f"+{res.credits_added} cred, +{res.isrcs_filled} isrc) via {prov_str}"
            )
            just_finished = self._repo.complete_queue_item(
                item_id=item_id,
                job_id=job_id,
                input_id=int(input_id),
                result_status=final_status,
                result_http_code=200,
                result_source=final_source,
                selected_release_id=rel_id,
                selected_agent_fallback_id=fb_id,
                track_count=res.tracks_total,
                status_message=status_msg,
                error_detail=str(ctx.get("warning") or ""),
            )
        except Exception as exc:
            logger.exception("[%s] Failure on enrich item #%s", worker_name, item_id)
            prior_status = ctx.get("result_status")
            if prior_status:
                just_finished = self._repo.complete_queue_item(
                    item_id=item_id,
                    job_id=job_id,
                    input_id=input_id,
                    result_status=str(prior_status),
                    result_http_code=200,
                    result_source=str(ctx.get("result_source") or "track_enrich"),
                    selected_release_id=item.get("selected_release_id"),
                    selected_agent_fallback_id=item.get("selected_agent_fallback_id"),
                    track_count=int(item.get("track_count") or 0),
                    status_message=f"Enrichment failed (kept {ctx.get('result_source')} match)",
                    error_detail=str(exc),
                )
            else:
                just_finished = self._repo.complete_queue_item(
                    item_id=item_id,
                    job_id=job_id,
                    input_id=input_id,
                    result_status="upstream_error",
                    result_http_code=500,
                    result_source="track_enrich",
                    status_message="Enrichment error",
                    error_detail=str(exc),
                )

        if just_finished:
            self._finalize_job_report(job_id)
