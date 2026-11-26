import csv
import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.agent_client import (
    AgentFallbackError,
    AgentFallbackProvider,
    AgentTrack,
    extract_discogs_release_ids,
)
from app.config import Settings
from app.discogs_client import CatalogProvider, DiscogsError, DiscogsRateLimitError
from app.enricher import TrackEnricher, compute_track_coverages, track_needs_album_enrichment
from app.evaluator import (
    EvaluationBundle,
    build_evaluation_out,
    evaluate_candidates,
)
from app.extractor import (
    ExtractedRelease,
    extract_release,
    extracted_track_to_out,
    format_duration_raw,
    is_resolved_artist_name,
)
from app.fallback_providers import FallbackProvider
from app.models import (
    AlternativeEditionOut,
    BatchEnrichResponse,
    BatchRequest,
    BatchRowOutcome,
    BatchSummaryResponse,
    EnrichRecordResponse,
    EvaluationOut,
    LookupNotFoundResponse,
    LookupRequest,
    LookupSuccessResponse,
    NormalizedInput,
    QueueJobDetailOut,
    ReleaseOut,
    TrackArtistOut,
    TrackCreditOut,
    TrackOut,
)
from app.normalizer import extract_label_catno_hint, normalize_input, normalize_text
from app.repository import PipelineRepository

logger = logging.getLogger(__name__)


@dataclass
class DiscogsHandoff:
    input_id: int
    norm: NormalizedInput
    queries_attempted: list[dict[str, Any]]
    fetched_ids: set[int]
    candidates_checked: int
    source_label: str
    allow_discogs_bridge: bool
    warning: str | None = None
    discogs_error_detail: str = ""
    discogs_error_http_code: int | None = None
    discogs_error_type: str = ""
    discogs_rejection_reasons: list[str] = field(default_factory=list)
    replace_matches: bool = True

    def to_context_dict(self) -> dict[str, Any]:
        return {
            "input_id": self.input_id,
            "norm": self.norm.model_dump(),
            "queries_attempted": self.queries_attempted,
            "fetched_ids": sorted(self.fetched_ids),
            "candidates_checked": self.candidates_checked,
            "source_label": self.source_label,
            "allow_discogs_bridge": self.allow_discogs_bridge,
            "warning": self.warning,
            "discogs_error_detail": self.discogs_error_detail,
            "discogs_error_http_code": self.discogs_error_http_code,
            "discogs_error_type": self.discogs_error_type,
            "discogs_rejection_reasons": self.discogs_rejection_reasons,
            "replace_matches": self.replace_matches,
        }

    @classmethod
    def from_context_dict(cls, data: dict[str, Any]) -> "DiscogsHandoff":
        return cls(
            input_id=int(data["input_id"]),
            norm=NormalizedInput.model_validate(data["norm"]),
            queries_attempted=list(data.get("queries_attempted") or []),
            fetched_ids={int(x) for x in (data.get("fetched_ids") or [])},
            candidates_checked=int(data.get("candidates_checked") or 0),
            source_label=str(data.get("source_label") or "discogs_live"),
            allow_discogs_bridge=bool(data.get("allow_discogs_bridge", True)),
            warning=data.get("warning"),
            discogs_error_detail=str(data.get("discogs_error_detail") or ""),
            discogs_error_http_code=data.get("discogs_error_http_code"),
            discogs_error_type=str(data.get("discogs_error_type") or ""),
            discogs_rejection_reasons=list(data.get("discogs_rejection_reasons") or []),
            replace_matches=bool(data.get("replace_matches", True)),
        )


@dataclass
class DiscogsStageOutcome:
    completed: LookupSuccessResponse | LookupNotFoundResponse | None = None
    handoff: DiscogsHandoff | None = None


class UpstreamServiceError(Exception):
    def __init__(self, status_code: int, detail: str, error_type: str = "upstream_error") -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.error_type = error_type


def _release_to_out(rel: ExtractedRelease) -> ReleaseOut:
    return ReleaseOut(
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
    )


def _agent_tracks_to_track_outs(
    tracks: list[AgentTrack],
    norm: NormalizedInput | None = None,
    sources: list[str] | None = None,
) -> tuple[list[tuple[int, TrackOut]], str]:
    has_multi_disc = any(t.disc > 1 for t in tracks)
    numbered: list[tuple[int, TrackOut]] = []
    signature_items: list[list[Any]] = []
    default_url = sources[0] if sources else ""

    for order, t in enumerate(tracks, start=1):
        pos = f"{t.disc}-{t.number}" if has_multi_disc else str(t.number)
        dur_sec = t.length_sec if (t.length_sec is not None and t.length_sec > 0) else None
        has_valid_dur = dur_sec is not None

        raw_artist = (t.artist or "").strip()
        if is_resolved_artist_name(raw_artist):
            artists = [TrackArtistOut(name=raw_artist, inherited=False)]
            artist_disp = raw_artist
            resolved_perf = True
            perf_src = "gemini_agent"
            perf_url = default_url
        elif (
            norm is not None
            and not norm.is_compilation
            and norm.artist_clean
            and is_resolved_artist_name(norm.artist_clean)
        ):
            artists = [TrackArtistOut(name=norm.artist_clean, inherited=True)]
            artist_disp = norm.artist_clean
            resolved_perf = True
            perf_src = "album_artist_inherited"
            perf_url = default_url
        else:
            artists = []
            artist_disp = ""
            resolved_perf = False
            perf_src = ""
            perf_url = ""

        credits_map: dict[str, list[TrackCreditOut]] = {
            "composers": [
                TrackCreditOut(
                    artist_name=c.strip(),
                    role_raw="Composer",
                    role_category="composer",
                    credit_source="gemini_agent",
                    source_url=default_url,
                )
                for c in t.composers
                if c and c.strip()
            ],
            "lyricists": [
                TrackCreditOut(
                    artist_name=c.strip(),
                    role_raw="Lyricist",
                    role_category="lyricist",
                    credit_source="gemini_agent",
                    source_url=default_url,
                )
                for c in t.lyricists
                if c and c.strip()
            ],
            "producers": [
                TrackCreditOut(
                    artist_name=c.strip(),
                    role_raw="Producer",
                    role_category="producer",
                    credit_source="gemini_agent",
                    source_url=default_url,
                )
                for c in t.producers
                if c and c.strip()
            ],
            "featured": [],
            "other": [],
        }
        isrc = (t.isrc or "").strip().upper()

        track_out = TrackOut(
            track_order=order,
            disc_number=t.disc,
            disc_label=f"Disc {t.disc}" if has_multi_disc else "",
            position=pos,
            entry_type="track",
            parent_position="",
            title=t.title,
            duration_raw=format_duration_raw(dur_sec),
            duration_sec=dur_sec,
            performing_artists=artists,
            performing_artist_display=artist_disp,
            has_valid_duration=has_valid_dur,
            has_resolved_performer=resolved_perf,
            isrc=isrc,
            isrc_source="gemini_agent" if isrc else "",
            isrc_source_url=default_url if isrc else "",
            duration_source="gemini_agent" if has_valid_dur else "",
            duration_source_url=default_url if has_valid_dur else "",
            performer_source=perf_src,
            performer_source_url=perf_url,
            credits=credits_map,
        )
        numbered.append((t.number, track_out))
        signature_items.append([t.disc, normalize_text(t.title), dur_sec])
    tracklist_hash = hashlib.sha256(
        json.dumps(signature_items, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return numbered, tracklist_hash


class LookupService:
    def __init__(
        self,
        repo: PipelineRepository,
        provider: CatalogProvider,
        settings: Settings,
        agent_provider: AgentFallbackProvider | None = None,
        fallback_providers: list[FallbackProvider] | None = None,
        enricher: TrackEnricher | None = None,
    ) -> None:
        self._repo = repo
        self._provider = provider
        self._settings = settings
        self._agent_provider = agent_provider
        self._fallback_providers: list[FallbackProvider] = list(fallback_providers or [])
        self._enricher = enricher

    def has_track_enricher(self) -> bool:
        return self._enricher is not None and self._settings.enable_track_enrichment

    @staticmethod
    def needs_track_enrichment(tracks: list[TrackOut]) -> bool:
        return any(track_needs_album_enrichment(t) for t in tracks)

    def _build_success_from_stored(
        self,
        input_id: int,
        norm: NormalizedInput,
        latest_run: dict[str, Any],
        source_override: str,
        fallback_used: bool = False,
        warning: str | None = None,
    ) -> LookupSuccessResponse | None:
        selected_id = latest_run.get("selected_release_id")
        selected_agent_id = latest_run.get("selected_agent_fallback_id")
        status = "needs_review" if latest_run["status"] == "needs_review" else "matched"
        summary = latest_run.get("evaluation_summary_json") or {}

        if selected_id is None and selected_agent_id is not None:
            loaded = self._repo.get_agent_fallback(int(selected_agent_id))
            if loaded is None:
                return None
            agent_fb, tracks = loaded
            dur_cov, perf_cov, wri_cov, prod_cov = compute_track_coverages(tracks)
            inferred_discs = max((t.disc_number for t in tracks), default=1)
            disc_consistent = (
                bool(summary["disc_format_consistent"])
                if "disc_format_consistent" in summary
                else not (
                    norm.format_hint.qty
                    and norm.format_hint.qty > 1
                    and inferred_discs < norm.format_hint.qty
                )
            )
            evaluation = EvaluationOut(
                match_confidence=float(summary.get("match_confidence", 0.0)),
                match_method=str(summary.get("match_method", "gemini_agent_search")),
                candidates_checked=int(latest_run.get("candidates_checked", 0)),
                candidates_passing=int(summary.get("candidates_passing", 0)),
                editions_agree=bool(summary.get("editions_agree", False)),
                track_count=len(tracks),
                duration_coverage=dur_cov,
                performer_coverage=perf_cov,
                writer_credit_coverage=wri_cov,
                producer_credit_coverage=prod_cov,
                disc_format_consistent=disc_consistent,
                discogs_data_quality="",
                review_reasons=list(latest_run.get("review_reasons_json") or []),
                rejection_reasons=list(summary.get("rejection_reasons") or []),
            )
            return LookupSuccessResponse(
                status=status,
                http_code=200,
                source=source_override,
                fallback_used=True,
                warning=warning,
                input_id=input_id,
                normalized_input=norm,
                release=None,
                agent_result=agent_fb,
                tracks=tracks,
                alternatives=[],
                evaluation=evaluation,
            )

        if selected_id is None:
            return None
        selected_rel = self._repo.get_extracted_release(int(selected_id))
        if selected_rel is None:
            return None

        matches = self._repo.get_input_matches(input_id)
        alternatives: list[AlternativeEditionOut] = []
        for m in matches:
            rid = int(m["discogs_release_id"])
            if rid == selected_rel.discogs_release_id:
                continue
            alt_rel = self._repo.get_extracted_release(rid)
            if alt_rel is None:
                continue
            diff_msg = (
                "Identical tracklist"
                if alt_rel.tracklist_hash == selected_rel.tracklist_hash
                else f"Different tracklist ({alt_rel.track_count} tracks vs {selected_rel.track_count})"
            )
            alternatives.append(
                AlternativeEditionOut(
                    discogs_release_id=alt_rel.discogs_release_id,
                    url=alt_rel.url,
                    title=alt_rel.title,
                    artist_display=alt_rel.artist_display,
                    year=alt_rel.year,
                    country=alt_rel.country,
                    formats=alt_rel.formats,
                    track_count=alt_rel.track_count,
                    tracklist_hash=alt_rel.tracklist_hash,
                    score=float(m["score"]),
                    match_method=str(m["match_method"]),
                    diff_summary=diff_msg,
                )
            )

        tracks_out = [extracted_track_to_out(t) for t in selected_rel.tracks]
        dur_cov, perf_cov, wri_cov, prod_cov = compute_track_coverages(tracks_out)
        evaluation = EvaluationOut(
            match_confidence=float(summary.get("match_confidence", 0.0)),
            match_method=str(summary.get("match_method", "database")),
            candidates_checked=int(latest_run.get("candidates_checked", 0)),
            candidates_passing=int(summary.get("candidates_passing", len(matches))),
            editions_agree=bool(summary.get("editions_agree", True)),
            track_count=selected_rel.track_count,
            duration_coverage=dur_cov,
            performer_coverage=perf_cov,
            writer_credit_coverage=wri_cov,
            producer_credit_coverage=prod_cov,
            disc_format_consistent=bool(summary.get("disc_format_consistent", True)),
            discogs_data_quality=selected_rel.data_quality,
            review_reasons=list(latest_run.get("review_reasons_json") or []),
            rejection_reasons=list(summary.get("rejection_reasons") or []),
        )
        return LookupSuccessResponse(
            status=status,
            http_code=200,
            source=source_override,
            fallback_used=fallback_used,
            warning=warning,
            input_id=input_id,
            normalized_input=norm,
            release=_release_to_out(selected_rel),
            agent_result=None,
            tracks=tracks_out,
            alternatives=alternatives,
            evaluation=evaluation,
        )

    def _try_local_cross_input_match(
        self,
        input_id: int,
        norm: NormalizedInput,
        source_label: str,
        fallback_used: bool = False,
        warning: str | None = None,
        error_detail: str = "",
    ) -> LookupSuccessResponse | None:
        method_map: dict[int, str] = {}
        candidate_ids: list[int] = []

        for rid in self._repo.find_release_ids_by_barcodes(norm.barcode_variants):
            if rid not in method_map:
                method_map[rid] = "local_db_barcode"
                candidate_ids.append(rid)

        for rid in self._repo.find_release_ids_by_artist_title(
            norm.artist_norm, norm.title_norm, norm.is_compilation
        ):
            if rid not in method_map:
                method_map[rid] = "local_db_title_artist"
                candidate_ids.append(rid)

        if not candidate_ids:
            return None

        local_releases: list[ExtractedRelease] = []
        for rid in candidate_ids:
            rel = self._repo.get_extracted_release(rid)
            if rel is not None:
                local_releases.append(rel)

        if not local_releases:
            return None

        bundle = evaluate_candidates(norm, local_releases, method_map)
        if bundle.status == "not_found" or bundle.selected is None:
            return None

        eval_out = build_evaluation_out(bundle, len(local_releases))
        matches_rows = self._bundle_to_match_rows(bundle)
        self._repo.save_matches_and_run(
            input_id=input_id,
            status=bundle.status,
            http_code=200,
            source=source_label,
            selected_release_id=bundle.selected.discogs_release_id,
            queries_attempted=[{"tier": "local_db", "candidates": candidate_ids}],
            candidates_checked=len(local_releases),
            evaluation_summary=eval_out.model_dump(),
            review_reasons=eval_out.review_reasons,
            matches=matches_rows,
            error_detail=error_detail,
            replace_matches=True,
        )
        return LookupSuccessResponse(
            status=bundle.status,
            http_code=200,
            source=source_label,
            fallback_used=fallback_used,
            warning=warning,
            input_id=input_id,
            normalized_input=norm,
            release=_release_to_out(bundle.selected),
            tracks=[extracted_track_to_out(t) for t in bundle.selected.tracks],
            alternatives=bundle.alternatives,
            evaluation=eval_out,
        )

    @staticmethod
    def _bundle_to_match_rows(bundle: EvaluationBundle) -> list[dict[str, Any]]:
        assert bundle.selected is not None
        rows: list[dict[str, Any]] = []
        for idx, (rel, ev) in enumerate(bundle.ranked, start=1):
            rows.append(
                {
                    "discogs_release_id": rel.discogs_release_id,
                    "is_selected": rel.discogs_release_id == bundle.selected.discogs_release_id,
                    "match_rank": idx,
                    "match_method": ev.match_method,
                    "score": ev.score,
                    "evaluation_json": {
                        "barcode_matched": ev.barcode_matched,
                        "title_sim": ev.title_sim,
                        "artist_sim": ev.artist_sim,
                        "format_matched": ev.format_matched,
                        "disc_format_consistent": ev.disc_format_consistent,
                        "duration_coverage": ev.duration_coverage,
                        "performer_coverage": ev.performer_coverage,
                        "writer_credit_coverage": ev.writer_credit_coverage,
                        "producer_credit_coverage": ev.producer_credit_coverage,
                        "review_reasons": ev.review_reasons,
                    },
                }
            )
        return rows

    def has_agent_fallback(self) -> bool:
        return self._agent_provider is not None and self._settings.enable_agent_fallback

    def _run_agent_fallback(
        self,
        input_id: int,
        norm: NormalizedInput,
        queries_attempted: list[dict[str, Any]],
        candidates_checked: int,
        already_fetched_ids: set[int],
        force_refresh: bool,
        allow_discogs_bridge: bool,
        source_label: str,
        warning: str | None = None,
        discogs_error_detail: str = "",
        discogs_rejection_reasons: list[str] | None = None,
        replace_matches: bool = True,
        progress_cb: Callable[[str, int | None], None] | None = None,
    ) -> LookupSuccessResponse | None:
        if self._agent_provider is None or not self._settings.enable_agent_fallback:
            return None

        try:
            research_with_progress = getattr(
                self._agent_provider, "research_album_with_progress", None
            )
            if callable(research_with_progress):
                def _on_agent_progress(attempt: int, _total: int, msg: str) -> None:
                    if progress_cb is not None:
                        progress_cb(msg, attempt)

                answer = research_with_progress(
                    norm.barcode_raw,
                    norm.album_title_raw,
                    norm.album_artist_raw,
                    on_progress=_on_agent_progress,
                )
            else:
                if progress_cb is not None:
                    progress_cb(
                        f"Searching via Gemini agent ({self._agent_provider.model_name})",
                        1,
                    )
                answer = self._agent_provider.research_album(
                    norm.barcode_raw, norm.album_title_raw, norm.album_artist_raw
                )
        except AgentFallbackError as exc:
            queries_attempted.append(
                {
                    "tier": "agent_fallback",
                    "model": self._agent_provider.model_name,
                    "error": str(exc),
                }
            )
            return None

        queries_attempted.append(
            {
                "tier": "agent_fallback",
                "model": self._agent_provider.model_name,
                "track_count": len(answer.tracks),
                "sources": answer.sources,
            }
        )
        if not answer.tracks:
            return None

        rejection_reasons = list(discogs_rejection_reasons or [])
        if allow_discogs_bridge:
            bridge_ids = [
                rid
                for rid in extract_discogs_release_ids(answer.sources)
                if rid not in already_fetched_ids
            ][: self._settings.max_candidates_to_fetch]
            if bridge_ids:
                if progress_cb is not None:
                    progress_cb(
                        f"Verifying Discogs release URL(s) {bridge_ids} from agent sources",
                        None,
                    )
                bridged_releases: list[ExtractedRelease] = []
                bridge_methods: dict[int, str] = {}
                bridge_error = False
                for rid in bridge_ids:
                    already_fetched_ids.add(rid)
                    try:
                        raw_rel, _ = self._provider.get_release(
                            rid, bypass_cache=force_refresh
                        )
                    except DiscogsError as exc:
                        bridge_error = True
                        queries_attempted.append(
                            {
                                "tier": "agent_discogs_bridge",
                                "release_ids": bridge_ids,
                                "error": str(exc),
                            }
                        )
                        break
                    if not raw_rel or "id" not in raw_rel:
                        continue
                    extracted = extract_release(raw_rel)
                    self._repo.upsert_release_with_tracks(extracted)
                    bridged_releases.append(extracted)
                    bridge_methods[rid] = "agent_discogs_url"

                if not bridge_error:
                    bridge_bundle = (
                        evaluate_candidates(norm, bridged_releases, bridge_methods)
                        if bridged_releases
                        else None
                    )
                    bridge_matched = (
                        bridge_bundle is not None
                        and bridge_bundle.status != "not_found"
                        and bridge_bundle.selected is not None
                    )
                    queries_attempted.append(
                        {
                            "tier": "agent_discogs_bridge",
                            "release_ids": bridge_ids,
                            "fetched": len(bridged_releases),
                            "matched": bridge_matched,
                        }
                    )
                    if bridge_matched:
                        assert bridge_bundle is not None
                        assert bridge_bundle.selected is not None
                        total_checked = candidates_checked + len(bridged_releases)
                        eval_out = build_evaluation_out(bridge_bundle, total_checked)
                        matches_rows = self._bundle_to_match_rows(bridge_bundle)
                        self._repo.save_matches_and_run(
                            input_id=input_id,
                            status=bridge_bundle.status,
                            http_code=200,
                            source="discogs_via_agent",
                            selected_release_id=bridge_bundle.selected.discogs_release_id,
                            selected_agent_fallback_id=None,
                            queries_attempted=queries_attempted,
                            candidates_checked=total_checked,
                            evaluation_summary=eval_out.model_dump(),
                            review_reasons=eval_out.review_reasons,
                            matches=matches_rows,
                            replace_matches=True,
                        )
                        return LookupSuccessResponse(
                            status=bridge_bundle.status,
                            http_code=200,
                            source="discogs_via_agent",
                            fallback_used=True,
                            warning="Recovered Discogs release via Gemini agent source URL.",
                            input_id=input_id,
                            normalized_input=norm,
                            release=_release_to_out(bridge_bundle.selected),
                            agent_result=None,
                            tracks=[
                                extracted_track_to_out(t)
                                for t in bridge_bundle.selected.tracks
                            ],
                            alternatives=bridge_bundle.alternatives,
                            evaluation=eval_out,
                        )
                    if bridge_bundle is not None:
                        rejection_reasons.extend(bridge_bundle.rejection_reasons)

        numbered_tracks, tracklist_hash = _agent_tracks_to_track_outs(
            answer.tracks, norm=norm, sources=answer.sources
        )
        track_outs = [t for _, t in numbered_tracks]
        agent_fb = self._repo.save_agent_fallback(
            input_id=input_id,
            model_name=self._agent_provider.model_name,
            sources=answer.sources,
            tracks=numbered_tracks,
            tracklist_hash=tracklist_hash,
            raw_json=answer.model_dump(),
            input_tokens=answer.input_tokens,
            output_tokens=answer.output_tokens,
            cost_usd=answer.cost_usd,
        )
        dur_cov, perf_cov, wri_cov, prod_cov = compute_track_coverages(track_outs)
        inferred_discs = max((t.disc_number for t in track_outs), default=1)
        disc_consistent = not (
            norm.format_hint.qty
            and norm.format_hint.qty > 1
            and inferred_discs < norm.format_hint.qty
        )
        review_reasons = [
            f"Discovered via Gemini agent search ({self._agent_provider.model_name}); verify track metadata."
        ]
        if warning:
            review_reasons.append(warning)
        if not disc_consistent:
            review_reasons.append(
                "Inferred disc count in tracklist does not match release format quantity."
            )

        eval_out = EvaluationOut(
            match_confidence=0.0,
            match_method="gemini_agent_search",
            candidates_checked=candidates_checked,
            candidates_passing=0,
            editions_agree=False,
            track_count=len(track_outs),
            duration_coverage=dur_cov,
            performer_coverage=perf_cov,
            writer_credit_coverage=wri_cov,
            producer_credit_coverage=prod_cov,
            disc_format_consistent=disc_consistent,
            discogs_data_quality="",
            review_reasons=review_reasons,
            rejection_reasons=list(dict.fromkeys(rejection_reasons)),
        )
        self._repo.save_matches_and_run(
            input_id=input_id,
            status="needs_review",
            http_code=200,
            source=source_label,
            selected_release_id=None,
            selected_agent_fallback_id=agent_fb.agent_fallback_id,
            queries_attempted=queries_attempted,
            candidates_checked=candidates_checked,
            evaluation_summary=eval_out.model_dump(),
            review_reasons=review_reasons,
            matches=[],
            error_detail=discogs_error_detail,
            replace_matches=replace_matches,
        )
        return LookupSuccessResponse(
            status="needs_review",
            http_code=200,
            source=source_label,
            fallback_used=True,
            warning=warning,
            input_id=input_id,
            normalized_input=norm,
            release=None,
            agent_result=agent_fb,
            tracks=track_outs,
            alternatives=[],
            evaluation=eval_out,
        )

    def run_discogs_stage(
        self,
        req: LookupRequest,
        source_tag: str = "api",
        progress_cb: Callable[[str, int | None], None] | None = None,
    ) -> DiscogsStageOutcome:
        norm = normalize_input(req.barcode, req.album_title, req.album_artist)
        input_id = self._repo.upsert_input_record(norm, source_tag=source_tag)

        # Layer 1 & Layer 2 DB deduplication when not forcing refresh
        if not req.force_refresh:
            latest_run = self._repo.find_latest_run(input_id)
            if latest_run is not None:
                if latest_run["status"] in ("matched", "needs_review"):
                    stored = self._build_success_from_stored(
                        input_id=input_id,
                        norm=norm,
                        latest_run=latest_run,
                        source_override="database",
                    )
                    if stored is not None:
                        return DiscogsStageOutcome(completed=stored)
                    logger.info(
                        "Stored run for input_id=%s had missing target release/fallback; re-running discovery.",
                        input_id,
                    )
                elif latest_run["status"] == "not_found":
                    created_at: datetime = latest_run["created_at"]
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                    cutoff = datetime.now(timezone.utc) - timedelta(
                        hours=self._settings.not_found_ttl_hours
                    )
                    if created_at >= cutoff:
                        summary = latest_run.get("evaluation_summary_json") or {}
                        return DiscogsStageOutcome(
                            completed=LookupNotFoundResponse(
                                http_code=404,
                                source="database_cache",
                                input_id=input_id,
                                normalized_input=norm,
                                queries_attempted=list(
                                    latest_run.get("queries_attempted_json") or []
                                ),
                                candidates_checked=int(
                                    latest_run.get("candidates_checked", 0)
                                ),
                                rejection_reasons=list(
                                    summary.get("rejection_reasons") or []
                                ),
                            )
                        )

            local_hit = self._try_local_cross_input_match(
                input_id=input_id,
                norm=norm,
                source_label="database",
            )
            if local_hit is not None:
                return DiscogsStageOutcome(completed=local_hit)

        # Live/Cached Discogs cascading search
        queries_attempted: list[dict[str, Any]] = []
        candidate_ids: list[int] = []
        candidate_methods: dict[int, str] = {}
        all_from_cache = True

        def _run_search(tier: str, params: dict[str, Any], method: str) -> int:
            nonlocal all_from_cache
            if progress_cb is not None:
                progress_cb(f"Discogs {tier} search", None)
            payload, from_cache = self._provider.search_releases(
                params, bypass_cache=req.force_refresh
            )
            if not from_cache:
                all_from_cache = False
            results = list(payload.get("results") or [])
            queries_attempted.append(
                {
                    "tier": tier,
                    "params": params,
                    "result_count": len(results),
                    "from_cache": from_cache,
                }
            )
            added = 0
            for item in results:
                rid_raw = item.get("id")
                if rid_raw is None:
                    continue
                rid = int(rid_raw)
                if rid not in candidate_methods:
                    candidate_methods[rid] = method
                    candidate_ids.append(rid)
                    added += 1
            return added

        fetched_releases: list[ExtractedRelease] = []
        fetched_ids: set[int] = set()

        try:
            def _fetch_new_candidates(ids_to_consider: list[int]) -> None:
                nonlocal all_from_cache
                newly_fetched = 0
                for rid in ids_to_consider:
                    if rid in fetched_ids:
                        continue
                    if newly_fetched >= self._settings.max_candidates_to_fetch:
                        break
                    fetched_ids.add(rid)
                    if progress_cb is not None:
                        progress_cb(f"Fetching Discogs candidate #{rid}", None)
                    raw_rel, from_cache = self._provider.get_release(
                        rid, bypass_cache=req.force_refresh
                    )
                    if not from_cache:
                        all_from_cache = False
                    if not raw_rel or "id" not in raw_rel:
                        continue
                    extracted = extract_release(raw_rel)
                    self._repo.upsert_release_with_tracks(extracted)
                    fetched_releases.append(extracted)
                    newly_fetched += 1

            # Tier 1: Barcode search (with 12 <-> 13 leading-zero variants)
            for idx, b_var in enumerate(norm.barcode_variants):
                method = "barcode_exact" if idx == 0 else "barcode_variant"
                _run_search("tier1_barcode", {"barcode": b_var}, method)
                if candidate_ids:
                    break

            if candidate_ids:
                _fetch_new_candidates(candidate_ids)

            bundle = evaluate_candidates(norm, fetched_releases, candidate_methods)

            # Tier 2 & Tier 3 fallback if barcode search missed OR all Tier 1 candidates failed evaluation
            if (not candidate_ids or bundle.status == "not_found") and norm.title_clean:
                tier1_count = len(candidate_ids)
                t2_params: dict[str, Any] = {"release_title": norm.title_clean}
                if norm.artist_clean and not norm.is_compilation:
                    t2_params["artist"] = norm.artist_clean
                added_t2 = _run_search(
                    "tier2_structured", t2_params, "title_artist_structured"
                )

                if added_t2 == 0:
                    added_t2b = 0
                    label_catno_hint = extract_label_catno_hint(norm.title_clean)
                    if label_catno_hint is not None:
                        added_t2b = _run_search(
                            "tier2b_label_catno",
                            {"label": label_catno_hint[0], "catno": label_catno_hint[1]},
                            "label_catno_structured",
                        )
                    if added_t2b == 0:
                        q_parts = []
                        if norm.artist_clean and not norm.is_compilation:
                            q_parts.append(norm.artist_clean)
                        q_parts.append(norm.title_clean)
                        _run_search("tier3_freetext", {"q": " ".join(q_parts)}, "freetext_q")

                fallback_ids = candidate_ids[tier1_count:]
                if fallback_ids:
                    _fetch_new_candidates(fallback_ids)
                    bundle = evaluate_candidates(norm, fetched_releases, candidate_methods)

        except DiscogsError as exc:
            logger.warning(
                "Discogs stage failed for %s - %s: %s",
                norm.album_artist_raw,
                norm.album_title_raw,
                exc,
            )
            warning_msg = f"Live Discogs lookup failed ({exc}); served stored database fallback."
            prior_ok = self._repo.find_latest_run(
                input_id, statuses=("matched", "needs_review")
            )
            if prior_ok is not None:
                fallback_resp = self._build_success_from_stored(
                    input_id=input_id,
                    norm=norm,
                    latest_run=prior_ok,
                    source_override="database_fallback",
                    fallback_used=True,
                    warning=warning_msg,
                )
                if fallback_resp is not None:
                    self._repo.save_matches_and_run(
                        input_id=input_id,
                        status=prior_ok["status"],
                        http_code=200,
                        source="database_fallback",
                        selected_release_id=(
                            fallback_resp.release.discogs_release_id
                            if fallback_resp.release
                            else None
                        ),
                        selected_agent_fallback_id=(
                            fallback_resp.agent_result.agent_fallback_id
                            if fallback_resp.agent_result
                            else None
                        ),
                        queries_attempted=queries_attempted,
                        candidates_checked=fallback_resp.evaluation.candidates_checked,
                        evaluation_summary=fallback_resp.evaluation.model_dump(),
                        review_reasons=fallback_resp.evaluation.review_reasons,
                        matches=[],
                        error_detail=str(exc),
                        replace_matches=False,
                    )
                    return DiscogsStageOutcome(completed=fallback_resp)

            local_fb = self._try_local_cross_input_match(
                input_id=input_id,
                norm=norm,
                source_label="database_fallback",
                fallback_used=True,
                warning=warning_msg,
                error_detail=str(exc),
            )
            if local_fb is not None:
                return DiscogsStageOutcome(completed=local_fb)

            http_code = 503 if isinstance(exc, DiscogsRateLimitError) else 502
            return DiscogsStageOutcome(
                handoff=DiscogsHandoff(
                    input_id=input_id,
                    norm=norm,
                    queries_attempted=queries_attempted,
                    fetched_ids=fetched_ids,
                    candidates_checked=len(fetched_releases),
                    source_label="discogs_live",
                    allow_discogs_bridge=False,
                    warning=f"Live Discogs lookup failed ({exc}); served Gemini agent fallback.",
                    discogs_error_detail=str(exc),
                    discogs_error_http_code=http_code,
                    discogs_error_type=type(exc).__name__,
                    discogs_rejection_reasons=[],
                    replace_matches=False,
                )
            )

        source_label = (
            "discogs_cached"
            if (all_from_cache and not req.force_refresh and queries_attempted)
            else "discogs_live"
        )

        if bundle.status == "not_found" or bundle.selected is None:
            return DiscogsStageOutcome(
                handoff=DiscogsHandoff(
                    input_id=input_id,
                    norm=norm,
                    queries_attempted=queries_attempted,
                    fetched_ids=fetched_ids,
                    candidates_checked=len(fetched_releases),
                    source_label=source_label,
                    allow_discogs_bridge=True,
                    warning=None,
                    discogs_error_detail="",
                    discogs_error_http_code=None,
                    discogs_error_type="",
                    discogs_rejection_reasons=list(bundle.rejection_reasons),
                    replace_matches=True,
                )
            )

        eval_out = build_evaluation_out(bundle, len(fetched_releases))
        matches_rows = self._bundle_to_match_rows(bundle)
        self._repo.save_matches_and_run(
            input_id=input_id,
            status=bundle.status,
            http_code=200,
            source=source_label,
            selected_release_id=bundle.selected.discogs_release_id,
            queries_attempted=queries_attempted,
            candidates_checked=len(fetched_releases),
            evaluation_summary=eval_out.model_dump(),
            review_reasons=eval_out.review_reasons,
            matches=matches_rows,
            replace_matches=True,
        )
        return DiscogsStageOutcome(
            completed=LookupSuccessResponse(
                status=bundle.status,
                http_code=200,
                source=source_label,
                fallback_used=False,
                warning=None,
                input_id=input_id,
                normalized_input=norm,
                release=_release_to_out(bundle.selected),
                tracks=[extracted_track_to_out(t) for t in bundle.selected.tracks],
                alternatives=bundle.alternatives,
                evaluation=eval_out,
            )
        )

    def finalize_without_agent(self, handoff: DiscogsHandoff) -> LookupNotFoundResponse:
        if handoff.discogs_error_http_code is not None:
            self._repo.save_matches_and_run(
                input_id=handoff.input_id,
                status="upstream_error",
                http_code=handoff.discogs_error_http_code,
                source="discogs_live",
                selected_release_id=None,
                queries_attempted=handoff.queries_attempted,
                candidates_checked=0,
                evaluation_summary={},
                review_reasons=[],
                matches=[],
                error_detail=handoff.discogs_error_detail,
                replace_matches=False,
            )
            raise UpstreamServiceError(
                status_code=handoff.discogs_error_http_code,
                detail=handoff.discogs_error_detail,
                error_type=handoff.discogs_error_type or "DiscogsError",
            )

        self._repo.save_matches_and_run(
            input_id=handoff.input_id,
            status="not_found",
            http_code=404,
            source=handoff.source_label,
            selected_release_id=None,
            queries_attempted=handoff.queries_attempted,
            candidates_checked=handoff.candidates_checked,
            evaluation_summary={"rejection_reasons": handoff.discogs_rejection_reasons},
            review_reasons=[],
            matches=[],
            replace_matches=handoff.replace_matches,
        )
        return LookupNotFoundResponse(
            http_code=404,
            source=handoff.source_label,
            input_id=handoff.input_id,
            normalized_input=handoff.norm,
            queries_attempted=handoff.queries_attempted,
            candidates_checked=handoff.candidates_checked,
            rejection_reasons=handoff.discogs_rejection_reasons,
        )

    def run_external_fallback_stage(
        self,
        handoff: DiscogsHandoff,
        force_refresh: bool = False,
        progress_cb: Callable[[str, int | None], None] | None = None,
    ) -> LookupSuccessResponse | None:
        """Query deterministic external catalog fallbacks between Discogs and Gemini Agent."""
        if not self._fallback_providers or not self._settings.enable_external_fallbacks:
            return None

        for fb in self._fallback_providers:
            if progress_cb is not None:
                progress_cb(f"Checking {fb.name} fallback", None)
            try:
                fb_pairs, fb_queries, fb_from_cache = fb.lookup_candidates(
                    handoff.norm, bypass_cache=force_refresh
                )
            except Exception as exc:
                logger.warning("External fallback %s failed unexpectedly: %s", fb.name, exc)
                handoff.queries_attempted.append(
                    {"tier": f"fallback_{fb.name}", "error": str(exc)}
                )
                continue

            handoff.queries_attempted.extend(fb_queries)
            if not fb_pairs:
                continue

            fb_releases = [rel for rel, _ in fb_pairs]
            fb_methods = {rel.discogs_release_id: method for rel, method in fb_pairs}
            handoff.candidates_checked += len(fb_releases)
            fb_bundle = evaluate_candidates(handoff.norm, fb_releases, fb_methods)
            if fb_bundle.status == "not_found" or fb_bundle.selected is None:
                handoff.discogs_rejection_reasons.extend(fb_bundle.rejection_reasons)
                continue

            for rel, _ in fb_bundle.ranked:
                self._repo.upsert_release_with_tracks(rel)

            fb_source = (
                f"{fb.name}_cached"
                if (fb_from_cache and not force_refresh and fb_queries)
                else f"{fb.name}_live"
            )
            warn_msg = (
                f"Live Discogs lookup failed ({handoff.discogs_error_detail}); resolved via {fb.name} fallback."
                if handoff.discogs_error_http_code is not None
                else f"Discogs returned no passing match; resolved via {fb.name} fallback."
            )
            eval_out = build_evaluation_out(fb_bundle, handoff.candidates_checked)
            matches_rows = self._bundle_to_match_rows(fb_bundle)
            self._repo.save_matches_and_run(
                input_id=handoff.input_id,
                status=fb_bundle.status,
                http_code=200,
                source=fb_source,
                selected_release_id=fb_bundle.selected.discogs_release_id,
                selected_agent_fallback_id=None,
                queries_attempted=handoff.queries_attempted,
                candidates_checked=handoff.candidates_checked,
                evaluation_summary=eval_out.model_dump(),
                review_reasons=eval_out.review_reasons,
                matches=matches_rows,
                error_detail=handoff.discogs_error_detail,
                replace_matches=handoff.replace_matches,
            )
            return LookupSuccessResponse(
                status=fb_bundle.status,
                http_code=200,
                source=fb_source,
                fallback_used=True,
                warning=warn_msg,
                input_id=handoff.input_id,
                normalized_input=handoff.norm,
                release=_release_to_out(fb_bundle.selected),
                agent_result=None,
                tracks=[extracted_track_to_out(t) for t in fb_bundle.selected.tracks],
                alternatives=fb_bundle.alternatives,
                evaluation=eval_out,
            )

        handoff.discogs_rejection_reasons = list(
            dict.fromkeys(handoff.discogs_rejection_reasons)
        )
        return None

    def run_agent_stage(
        self,
        handoff: DiscogsHandoff,
        force_refresh: bool = False,
        progress_cb: Callable[[str, int | None], None] | None = None,
    ) -> LookupSuccessResponse | LookupNotFoundResponse:
        agent_source_label = (
            "gemini_agent_fallback"
            if handoff.discogs_error_http_code is not None
            else "gemini_agent"
        )
        agent_resp = self._run_agent_fallback(
            input_id=handoff.input_id,
            norm=handoff.norm,
            queries_attempted=handoff.queries_attempted,
            candidates_checked=handoff.candidates_checked,
            already_fetched_ids=set(handoff.fetched_ids),
            force_refresh=force_refresh,
            allow_discogs_bridge=handoff.allow_discogs_bridge,
            source_label=agent_source_label,
            warning=handoff.warning,
            discogs_error_detail=handoff.discogs_error_detail,
            discogs_rejection_reasons=handoff.discogs_rejection_reasons,
            replace_matches=handoff.replace_matches,
            progress_cb=progress_cb,
        )
        if agent_resp is not None:
            return agent_resp
        return self.finalize_without_agent(handoff)

    def run_agent_only_for_input(
        self,
        barcode: str,
        album_title: str,
        album_artist: str,
        input_id: int | None = None,
        force_refresh: bool = True,
        source_tag: str = "queue_agent",
        progress_cb: Callable[[str, int | None], None] | None = None,
    ) -> LookupSuccessResponse | LookupNotFoundResponse:
        norm = normalize_input(barcode, album_title, album_artist)
        resolved_input_id = input_id or self._repo.upsert_input_record(
            norm, source_tag=source_tag
        )
        if progress_cb is not None:
            progress_cb("Preparing Gemini agent search", None)
        handoff = DiscogsHandoff(
            input_id=resolved_input_id,
            norm=norm,
            queries_attempted=[],
            fetched_ids=set(),
            candidates_checked=0,
            source_label="agent_only",
            allow_discogs_bridge=True,
            warning=None,
            discogs_error_detail="",
            discogs_error_http_code=None,
            discogs_rejection_reasons=["Direct Gemini agent lookup/retry requested."],
            replace_matches=True,
        )
        return self.run_agent_stage(
            handoff, force_refresh=force_refresh, progress_cb=progress_cb
        )

    def enrich_record(
        self,
        input_id: int,
        use_agent: bool = False,
        force_refresh: bool = False,
        progress_cb: Callable[[str, int | None], None] | None = None,
    ) -> EnrichRecordResponse | None:
        rec = self._repo.get_input_record(input_id)
        if rec is None:
            return None

        latest = self._repo.find_latest_run(input_id)
        if latest is None or latest["status"] not in ("matched", "needs_review"):
            raise ValueError(
                f"Input record {input_id} has no matched/needs_review result to enrich."
            )

        selected_release_id = latest.get("selected_release_id")
        selected_agent_id = latest.get("selected_agent_fallback_id")
        norm = normalize_input(
            rec["barcode_raw"], rec["album_title_raw"], rec["album_artist_raw"]
        )

        if selected_release_id is not None:
            selected_rel = self._repo.get_extracted_release(int(selected_release_id))
            if selected_rel is None:
                raise ValueError(
                    f"Selected release {selected_release_id} for input {input_id} was not found."
                )
            target_type = "release"
            target_id = selected_rel.discogs_release_id
            release_meta: ReleaseOut | None = _release_to_out(selected_rel)
            tracks = [extracted_track_to_out(t) for t in selected_rel.tracks]
            siblings = self._repo.find_sibling_releases(
                input_id, selected_rel.discogs_release_id, selected_rel.master_id
            )
            source_provider = selected_rel.source_provider
            agent_sources: list[str] = []
            release_notes = selected_rel.notes or ""
        elif selected_agent_id is not None:
            loaded = self._repo.get_agent_fallback(int(selected_agent_id))
            if loaded is None:
                raise ValueError(
                    f"Selected agent fallback {selected_agent_id} for input {input_id} was not found."
                )
            agent_fb, tracks = loaded
            target_type = "agent_fallback"
            target_id = agent_fb.agent_fallback_id
            release_meta = None
            siblings = []
            source_provider = "gemini_agent"
            agent_sources = list(agent_fb.sources)
            release_notes = ""
        else:
            raise ValueError(
                f"Input record {input_id} has no selected release or agent fallback."
            )

        active_enricher = self._enricher or TrackEnricher(settings=self._settings)
        stats = active_enricher.enrich_tracks(
            tracks=tracks,
            norm=norm,
            release_meta=release_meta,
            release_notes=release_notes,
            source_provider=source_provider,
            sibling_releases=siblings,
            agent_sources=agent_sources,
            use_agent=use_agent,
            bypass_cache=force_refresh,
            progress_cb=progress_cb,
        )

        if target_type == "release":
            self._repo.update_release_track_enrichments(
                target_id,
                stats.tracks,
                mark_enriched=True,
                add_input_tokens=stats.agent_input_tokens,
                add_output_tokens=stats.agent_output_tokens,
                add_cost_usd=stats.agent_cost_usd,
            )
        else:
            self._repo.update_agent_fallback_track_enrichments(
                target_id,
                stats.tracks,
                mark_enriched=True,
                add_input_tokens=stats.agent_input_tokens,
                add_output_tokens=stats.agent_output_tokens,
                add_cost_usd=stats.agent_cost_usd,
            )

        self._repo.refresh_latest_run_coverage(
            input_id=input_id,
            duration_cov=stats.duration_coverage_after,
            performer_cov=stats.performer_coverage_after,
            writer_cov=stats.writer_credit_coverage_after,
            producer_cov=stats.producer_credit_coverage_after,
        )

        rem_dur = sum(1 for t in stats.tracks if not t.has_valid_duration)
        rem_perf = sum(1 for t in stats.tracks if not t.has_resolved_performer)
        return EnrichRecordResponse(
            input_id=input_id,
            target_type=target_type,
            target_id=target_id,
            tracks_total=stats.tracks_total,
            durations_filled=stats.durations_filled,
            performers_filled=stats.performers_filled,
            credits_added=stats.credits_added,
            isrcs_filled=stats.isrcs_filled,
            duration_coverage_before=stats.duration_coverage_before,
            duration_coverage_after=stats.duration_coverage_after,
            performer_coverage_before=stats.performer_coverage_before,
            performer_coverage_after=stats.performer_coverage_after,
            writer_credit_coverage_before=stats.writer_credit_coverage_before,
            writer_credit_coverage_after=stats.writer_credit_coverage_after,
            producer_credit_coverage_before=stats.producer_credit_coverage_before,
            producer_credit_coverage_after=stats.producer_credit_coverage_after,
            providers_used=stats.providers_used,
            agent_input_tokens=stats.agent_input_tokens,
            agent_output_tokens=stats.agent_output_tokens,
            agent_cost_usd=stats.agent_cost_usd,
            remaining_missing_duration=rem_dur,
            remaining_missing_performer=rem_perf,
            tracks=stats.tracks,
        )

    def enrich_missing_records(
        self,
        limit: int = 100,
        use_agent: bool = False,
        force_refresh: bool = False,
    ) -> BatchEnrichResponse:
        rows = self._repo.list_matched_inputs_with_missing_track_data(
            limit=limit, force_refresh=force_refresh
        )
        outcomes: list[EnrichRecordResponse] = []
        records_enriched = 0
        tot_dur = 0
        tot_perf = 0
        tot_cred = 0
        tot_isrc = 0
        tot_cost = 0.0
        for row in rows:
            res = self.enrich_record(
                int(row["id"]), use_agent=use_agent, force_refresh=force_refresh
            )
            if res is None:
                continue
            outcomes.append(res)
            if (
                res.durations_filled > 0
                or res.performers_filled > 0
                or res.credits_added > 0
                or res.isrcs_filled > 0
            ):
                records_enriched += 1
            tot_dur += res.durations_filled
            tot_perf += res.performers_filled
            tot_cred += res.credits_added
            tot_isrc += res.isrcs_filled
            tot_cost += res.agent_cost_usd

        return BatchEnrichResponse(
            records_checked=len(outcomes),
            records_enriched=records_enriched,
            total_durations_filled=tot_dur,
            total_performers_filled=tot_perf,
            total_credits_added=tot_cred,
            total_isrcs_filled=tot_isrc,
            total_agent_cost_usd=round(tot_cost, 6),
            outcomes=outcomes,
        )

    def _maybe_enrich_success_response(
        self, res: LookupSuccessResponse, force_refresh: bool
    ) -> LookupSuccessResponse:
        if not self.has_track_enricher() or not self.needs_track_enrichment(res.tracks):
            return res
        if not force_refresh and res.source == "database":
            rel_id = res.release.discogs_release_id if res.release else None
            fb_id = res.agent_result.agent_fallback_id if res.agent_result else None
            if self._repo.is_target_already_enriched(rel_id, fb_id):
                return res

        enriched = self.enrich_record(
            res.input_id,
            use_agent=self._settings.enable_agent_track_enrichment and self.has_agent_fallback(),
            force_refresh=force_refresh,
        )
        if enriched is None:
            return res

        res.tracks = enriched.tracks
        res.evaluation.duration_coverage = enriched.duration_coverage_after
        res.evaluation.performer_coverage = enriched.performer_coverage_after
        res.evaluation.writer_credit_coverage = enriched.writer_credit_coverage_after
        res.evaluation.producer_credit_coverage = enriched.producer_credit_coverage_after
        if enriched.performer_coverage_after >= 1.0:
            res.evaluation.review_reasons = [
                r
                for r in res.evaluation.review_reasons
                if not r.startswith("Performing artists missing/unresolved")
            ]
            if (
                res.status == "needs_review"
                and not res.evaluation.review_reasons
                and not res.source.startswith("gemini_agent")
            ):
                res.status = "matched"
        return res

    def lookup(
        self, req: LookupRequest, source_tag: str = "api"
    ) -> LookupSuccessResponse | LookupNotFoundResponse:
        outcome = self.run_discogs_stage(req, source_tag=source_tag)
        if outcome.completed is not None:
            if isinstance(outcome.completed, LookupSuccessResponse):
                return self._maybe_enrich_success_response(
                    outcome.completed, req.force_refresh
                )
            return outcome.completed
        assert outcome.handoff is not None
        fb_resp = self.run_external_fallback_stage(
            outcome.handoff, force_refresh=req.force_refresh
        )
        if fb_resp is not None:
            return self._maybe_enrich_success_response(fb_resp, req.force_refresh)
        agent_resp = self.run_agent_stage(
            outcome.handoff, force_refresh=req.force_refresh
        )
        if isinstance(agent_resp, LookupSuccessResponse):
            return self._maybe_enrich_success_response(agent_resp, req.force_refresh)
        return agent_resp

    def run_batch(self, req: BatchRequest) -> BatchSummaryResponse:
        csv_file = Path(req.csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        with csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            all_rows = list(reader)

        selected_indices = (
            set(req.rows) if req.rows else set(range(1, len(all_rows) + 1))
        )

        outcomes: list[BatchRowOutcome] = []
        full_results: list[dict[str, Any]] = []
        matched_count = 0
        needs_review_count = 0
        not_found_count = 0
        error_count = 0

        for idx, row in enumerate(all_rows, start=1):
            if idx not in selected_indices:
                continue

            barcode = str(row.get("barcode") or "")
            album_title = str(row.get("album_title") or "")
            album_artist = str(row.get("album_artist") or "")

            try:
                lookup_req = LookupRequest(
                    barcode=barcode,
                    album_title=album_title,
                    album_artist=album_artist,
                    force_refresh=req.force_refresh,
                )
                result = self.lookup(lookup_req, source_tag=f"batch_row_{idx}")
                full_results.append({"row_number": idx, "result": result.model_dump()})

                if isinstance(result, LookupSuccessResponse):
                    if result.status == "matched":
                        matched_count += 1
                    else:
                        needs_review_count += 1
                    outcomes.append(
                        BatchRowOutcome(
                            row_number=idx,
                            barcode=barcode,
                            album_title=album_title,
                            album_artist=album_artist,
                            http_code=200,
                            status=result.status,
                            source=result.source,
                            fallback_used=result.fallback_used,
                            input_id=result.input_id,
                            discogs_release_id=(
                                result.release.discogs_release_id
                                if result.release
                                else None
                            ),
                            agent_fallback_id=(
                                result.agent_result.agent_fallback_id
                                if result.agent_result
                                else None
                            ),
                            release_url=(
                                result.release.url
                                if result.release
                                else (
                                    result.agent_result.sources[0]
                                    if result.agent_result and result.agent_result.sources
                                    else None
                                )
                            ),
                            agent_sources=(
                                list(result.agent_result.sources)
                                if result.agent_result
                                else []
                            ),
                            track_count=result.evaluation.track_count,
                            match_confidence=result.evaluation.match_confidence,
                            duration_coverage=result.evaluation.duration_coverage,
                            performer_coverage=result.evaluation.performer_coverage,
                            writer_credit_coverage=result.evaluation.writer_credit_coverage,
                            producer_credit_coverage=result.evaluation.producer_credit_coverage,
                            isrc_coverage=(
                                round(
                                    sum(1 for t in result.tracks if t.isrc)
                                    / len(result.tracks),
                                    4,
                                )
                                if result.tracks
                                else None
                            ),
                            review_reasons=result.evaluation.review_reasons,
                        )
                    )
                else:
                    not_found_count += 1
                    outcomes.append(
                        BatchRowOutcome(
                            row_number=idx,
                            barcode=barcode,
                            album_title=album_title,
                            album_artist=album_artist,
                            http_code=404,
                            status="not_found",
                            source=result.source,
                            input_id=result.input_id,
                            rejection_reasons=result.rejection_reasons,
                        )
                    )
            except UpstreamServiceError as exc:
                error_count += 1
                full_results.append(
                    {"row_number": idx, "error": {"http_code": exc.status_code, "detail": exc.detail}}
                )
                outcomes.append(
                    BatchRowOutcome(
                        row_number=idx,
                        barcode=barcode,
                        album_title=album_title,
                        album_artist=album_artist,
                        http_code=exc.status_code,
                        status="upstream_error",
                        source="discogs_live",
                        error_detail=exc.detail,
                    )
                )
            except ValueError as exc:
                error_count += 1
                detail = str(exc)
                full_results.append(
                    {"row_number": idx, "error": {"http_code": 422, "detail": detail}}
                )
                outcomes.append(
                    BatchRowOutcome(
                        row_number=idx,
                        barcode=barcode,
                        album_title=album_title,
                        album_artist=album_artist,
                        http_code=422,
                        status="validation_error",
                        source="input_validation",
                        error_detail=detail,
                    )
                )

        summary = BatchSummaryResponse(
            total_processed=len(outcomes),
            matched_count=matched_count,
            needs_review_count=needs_review_count,
            not_found_count=not_found_count,
            error_count=error_count,
            outcomes=outcomes,
        )

        if req.output_path:
            out_json = Path(req.output_path)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_md = out_json.with_suffix(".md")
            summary.output_json_path = str(out_json)
            summary.output_markdown_path = str(out_md)

            out_json.write_text(
                json.dumps(
                    {
                        "summary": summary.model_dump(),
                        "details": full_results,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            out_md.write_text(self._render_markdown_report(summary), encoding="utf-8")

        return summary

    def write_job_reports(self, detail: QueueJobDetailOut) -> tuple[str, str] | None:
        if not detail.job.output_path:
            return None

        outcomes: list[BatchRowOutcome] = []
        matched_count = 0
        needs_review_count = 0
        not_found_count = 0
        error_count = 0

        for item in detail.items:
            if item.stage == "cancelled":
                continue
            status = item.result_status or "not_found"
            if status == "matched":
                matched_count += 1
            elif status == "needs_review":
                needs_review_count += 1
            elif status == "not_found":
                not_found_count += 1
            else:
                error_count += 1

            release_url: str | None = None
            agent_sources: list[str] = []
            dur_cov: float | None = None
            perf_cov: float | None = None
            wri_cov: float | None = None
            prod_cov: float | None = None
            isrc_cov: float | None = None
            tracks_out: list[TrackOut] = []
            if item.selected_release_id is not None:
                rel = self._repo.get_extracted_release(item.selected_release_id)
                if rel is not None:
                    release_url = rel.url
                    tracks_out = [extracted_track_to_out(t) for t in rel.tracks]
            elif item.selected_agent_fallback_id is not None:
                fb = self._repo.get_agent_fallback(item.selected_agent_fallback_id)
                if fb is not None:
                    agent_sources = list(fb[0].sources)
                    release_url = agent_sources[0] if agent_sources else None
                    tracks_out = fb[1]

            if tracks_out:
                dur_cov, perf_cov, wri_cov, prod_cov = compute_track_coverages(tracks_out)
                isrc_cov = round(
                    sum(1 for t in tracks_out if t.isrc) / len(tracks_out), 4
                )

            match_conf: float | None = None
            review_reasons: list[str] = []
            rejection_reasons: list[str] = []
            if item.input_id is not None:
                latest = self._repo.find_latest_run(int(item.input_id))
                if latest is not None:
                    summary_json = latest.get("evaluation_summary_json") or {}
                    raw_conf = summary_json.get("match_confidence")
                    if raw_conf is not None:
                        match_conf = float(raw_conf)
                    review_reasons = [
                        str(r) for r in (latest.get("review_reasons_json") or [])
                    ]
                    rejection_reasons = [
                        str(r) for r in (summary_json.get("rejection_reasons") or [])
                    ]

            item_src = item.result_source or ""
            outcomes.append(
                BatchRowOutcome(
                    row_number=item.row_number,
                    barcode=item.barcode,
                    album_title=item.album_title,
                    album_artist=item.album_artist,
                    http_code=item.result_http_code or 0,
                    status=status,
                    source=item_src,
                    fallback_used=item_src
                    in {"gemini_agent", "gemini_agent_fallback", "discogs_via_agent", "database_fallback"}
                    or (item_src.endswith(("_live", "_cached")) and not item_src.startswith("discogs_")),
                    input_id=item.input_id,
                    discogs_release_id=item.selected_release_id,
                    agent_fallback_id=item.selected_agent_fallback_id,
                    release_url=release_url,
                    agent_sources=agent_sources,
                    track_count=item.track_count or len(tracks_out),
                    match_confidence=match_conf,
                    duration_coverage=dur_cov,
                    performer_coverage=perf_cov,
                    writer_credit_coverage=wri_cov,
                    producer_credit_coverage=prod_cov,
                    isrc_coverage=isrc_cov,
                    review_reasons=review_reasons,
                    rejection_reasons=rejection_reasons,
                    error_detail=item.error_detail or None,
                )
            )

        out_json = Path(detail.job.output_path)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_md = out_json.with_suffix(".md")

        summary = BatchSummaryResponse(
            total_processed=len(outcomes),
            matched_count=matched_count,
            needs_review_count=needs_review_count,
            not_found_count=not_found_count,
            error_count=error_count,
            output_json_path=str(out_json),
            output_markdown_path=str(out_md),
            outcomes=outcomes,
        )
        tmp_json = out_json.with_name(f".{out_json.name}.tmp")
        tmp_md = out_md.with_name(f".{out_md.name}.tmp")
        tmp_json.write_text(
            json.dumps(
                {
                    "summary": summary.model_dump(),
                    "job": detail.job.model_dump(),
                    "items": [i.model_dump() for i in detail.items],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        tmp_json.replace(out_json)
        tmp_md.write_text(self._render_markdown_report(summary), encoding="utf-8")
        tmp_md.replace(out_md)
        return str(out_json), str(out_md)

    @staticmethod
    def _render_markdown_report(summary: BatchSummaryResponse) -> str:
        lines = [
            "# Batch Evaluation Report",
            "",
            f"- **Processed:** {summary.total_processed}",
            f"- **Matched:** {summary.matched_count}",
            f"- **Needs review:** {summary.needs_review_count}",
            f"- **Not found (404):** {summary.not_found_count}",
            f"- **Errors:** {summary.error_count}",
            "",
            "| Row | Barcode | Title | Artist | HTTP | Status | Source | Tracks | Confidence | Duration Cov | Performer Cov | Writer Cov | ISRC Cov | Release URL |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for o in summary.outcomes:
            conf = f"{o.match_confidence:.2f}" if o.match_confidence is not None else "-"
            dur = f"{o.duration_coverage:.0%}" if o.duration_coverage is not None else "-"
            perf = f"{o.performer_coverage:.0%}" if o.performer_coverage is not None else "-"
            wri = f"{o.writer_credit_coverage:.0%}" if o.writer_credit_coverage is not None else "-"
            isrc_s = f"{o.isrc_coverage:.0%}" if o.isrc_coverage is not None else "-"
            url = o.release_url or "-"
            lines.append(
                f"| {o.row_number} | {o.barcode or '-'} | {o.album_title} | {o.album_artist} | "
                f"{o.http_code} | {o.status} | {o.source} | {o.track_count} | {conf} | {dur} | {perf} | {wri} | {isrc_s} | {url} |"
            )
        return "\n".join(lines) + "\n"
