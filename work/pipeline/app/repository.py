from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from app.extractor import ExtractedCredit, ExtractedRelease, ExtractedTrack
from app.models import (
    AgentFallbackOut,
    NormalizedInput,
    QueueItemOut,
    QueueJobDetailOut,
    QueueJobOut,
    QueueStageCountsOut,
    RecordDetailOut,
    RecordSummaryOut,
    ReleaseOut,
    ReviewOut,
    ReviewRequest,
    TrackArtistOut,
    TrackCreditOut,
    TrackOut,
)
from app.extractor import CREDIT_BUCKETS, extracted_track_to_out, format_duration_raw

ACTIVE_QUEUE_STAGES = (
    "queued_discogs",
    "checking_discogs",
    "queued_agent",
    "checking_agent",
    "queued_enrich",
    "checking_enrich",
)


def _iso_or_none(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _row_to_queue_job(row: dict[str, Any]) -> QueueJobOut:
    total = int(row["total_items"])
    completed = int(row["completed_items"])
    progress_pct = round(100.0 * completed / total, 1) if total > 0 else 100.0
    return QueueJobOut(
        id=int(row["id"]),
        job_type=row["job_type"],
        label=row["label"],
        status=row["status"],
        csv_path=row["csv_path"],
        output_path=row["output_path"],
        force_refresh=bool(row["force_refresh"]),
        total_items=total,
        completed_items=completed,
        matched_count=int(row["matched_count"]),
        needs_review_count=int(row["needs_review_count"]),
        agent_recovered_count=int(row["agent_recovered_count"]),
        not_found_count=int(row["not_found_count"]),
        error_count=int(row["error_count"]),
        cancelled_count=int(row["cancelled_count"]),
        progress_pct=progress_pct,
        output_json_path=row["output_json_path"],
        output_markdown_path=row["output_markdown_path"],
        created_at=row["created_at"].isoformat(),
        started_at=_iso_or_none(row["started_at"]),
        finished_at=_iso_or_none(row["finished_at"]),
        updated_at=row["updated_at"].isoformat(),
    )


def _row_to_queue_item(row: dict[str, Any]) -> QueueItemOut:
    started = row.get("started_at")
    finished = row.get("finished_at")
    elapsed_sec: float | None = None
    if started is not None:
        s_dt = started if started.tzinfo else started.replace(tzinfo=timezone.utc)
        if finished is not None:
            e_dt = finished if finished.tzinfo else finished.replace(tzinfo=timezone.utc)
        else:
            e_dt = datetime.now(timezone.utc)
        elapsed_sec = round(max(0.0, (e_dt - s_dt).total_seconds()), 2)

    return QueueItemOut(
        id=int(row["id"]),
        job_id=int(row["job_id"]),
        row_number=int(row["row_number"]),
        input_id=row["input_id"],
        barcode=row["barcode"],
        album_title=row["album_title"],
        album_artist=row["album_artist"],
        mode=row["mode"],
        stage=row["stage"],
        worker_name=row["worker_name"],
        attempt=int(row["attempt"]),
        status_message=row["status_message"],
        result_status=row["result_status"],
        result_http_code=row["result_http_code"],
        result_source=row["result_source"],
        selected_release_id=row["selected_release_id"],
        selected_agent_fallback_id=row["selected_agent_fallback_id"],
        track_count=int(row["track_count"]),
        error_detail=row["error_detail"],
        elapsed_sec=elapsed_sec,
        created_at=row["created_at"].isoformat(),
        started_at=_iso_or_none(started),
        finished_at=_iso_or_none(finished),
        updated_at=row["updated_at"].isoformat(),
    )


class PipelineRepository:
    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def ensure_schema(self) -> None:
        sql_path = Path(__file__).resolve().parent.parent / "sql" / "init.sql"
        ddl = sql_path.read_text(encoding="utf-8")
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()

    def ping(self) -> bool:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True

    def reset_pipeline_state(self, clear_api_cache: bool = False) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                if clear_api_cache:
                    cur.execute(
                        "TRUNCATE TABLE queue_jobs, input_records, releases, agent_fallbacks, api_cache RESTART IDENTITY CASCADE"
                    )
                else:
                    cur.execute(
                        "TRUNCATE TABLE queue_jobs, input_records, releases, agent_fallbacks RESTART IDENTITY CASCADE"
                    )
            conn.commit()

    def get_api_cache(self, cache_key: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT cache_key, endpoint, status_code, payload_json, fetched_at
                    FROM api_cache
                    WHERE cache_key = %s
                    """,
                    (cache_key,),
                )
                return cur.fetchone()

    def set_api_cache(
        self, cache_key: str, endpoint: str, status_code: int, payload: dict[str, Any]
    ) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO api_cache (cache_key, endpoint, status_code, payload_json, fetched_at)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON CONFLICT (cache_key) DO UPDATE SET
                        endpoint = EXCLUDED.endpoint,
                        status_code = EXCLUDED.status_code,
                        payload_json = EXCLUDED.payload_json,
                        fetched_at = NOW()
                    """,
                    (cache_key, endpoint, status_code, Jsonb(payload)),
                )
            conn.commit()

    def upsert_input_record(
        self, norm: NormalizedInput, source_tag: str = "api"
    ) -> int:
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    INSERT INTO input_records (
                        barcode_raw, album_title_raw, album_artist_raw,
                        barcode_norm, title_norm, artist_norm,
                        format_hint, is_compilation, input_hash, source_tag, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (input_hash) DO UPDATE SET
                        barcode_raw = EXCLUDED.barcode_raw,
                        album_title_raw = EXCLUDED.album_title_raw,
                        album_artist_raw = EXCLUDED.album_artist_raw,
                        updated_at = NOW()
                    RETURNING id
                    """,
                    (
                        norm.barcode_raw,
                        norm.album_title_raw,
                        norm.album_artist_raw,
                        norm.barcode_norm,
                        norm.title_norm,
                        norm.artist_norm,
                        Jsonb(norm.format_hint.model_dump()),
                        norm.is_compilation,
                        norm.input_hash,
                        source_tag,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return int(row["id"])

    def upsert_release_with_tracks(self, rel: ExtractedRelease) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO releases (
                        discogs_release_id, master_id, url, resource_url,
                        title, title_norm, artist_display, artist_norm,
                        year, country, formats_json, identifiers_json, labels_json,
                        data_quality, community_status, community_have, community_want,
                        notes, track_count, tracklist_hash, unscoped_credits_json,
                        raw_json, source_provider, fetched_at
                    )
                    VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, NOW()
                    )
                    ON CONFLICT (discogs_release_id) DO UPDATE SET
                        master_id = EXCLUDED.master_id,
                        url = EXCLUDED.url,
                        resource_url = EXCLUDED.resource_url,
                        title = EXCLUDED.title,
                        title_norm = EXCLUDED.title_norm,
                        artist_display = EXCLUDED.artist_display,
                        artist_norm = EXCLUDED.artist_norm,
                        year = EXCLUDED.year,
                        country = EXCLUDED.country,
                        formats_json = EXCLUDED.formats_json,
                        identifiers_json = EXCLUDED.identifiers_json,
                        labels_json = EXCLUDED.labels_json,
                        data_quality = EXCLUDED.data_quality,
                        community_status = EXCLUDED.community_status,
                        community_have = EXCLUDED.community_have,
                        community_want = EXCLUDED.community_want,
                        notes = EXCLUDED.notes,
                        track_count = EXCLUDED.track_count,
                        tracklist_hash = EXCLUDED.tracklist_hash,
                        unscoped_credits_json = EXCLUDED.unscoped_credits_json,
                        raw_json = EXCLUDED.raw_json,
                        source_provider = EXCLUDED.source_provider,
                        fetched_at = NOW()
                    """,
                    (
                        rel.discogs_release_id,
                        rel.master_id,
                        rel.url,
                        rel.resource_url,
                        rel.title,
                        rel.title_norm,
                        rel.artist_display,
                        rel.artist_norm,
                        rel.year,
                        rel.country,
                        Jsonb(rel.formats),
                        Jsonb(rel.identifiers),
                        Jsonb(rel.labels),
                        rel.data_quality,
                        rel.community_status,
                        rel.community_have,
                        rel.community_want,
                        rel.notes,
                        rel.track_count,
                        rel.tracklist_hash,
                        Jsonb(rel.unscoped_credits),
                        Jsonb(rel.raw_json),
                        rel.source_provider,
                    ),
                )

                cur.execute(
                    "DELETE FROM release_barcodes WHERE discogs_release_id = %s",
                    (rel.discogs_release_id,),
                )
                for b_norm, b_raw in rel.barcode_entries:
                    cur.execute(
                        """
                        INSERT INTO release_barcodes (discogs_release_id, barcode_norm, barcode_raw)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (discogs_release_id, barcode_norm) DO UPDATE SET
                            barcode_raw = EXCLUDED.barcode_raw
                        """,
                        (rel.discogs_release_id, b_norm, b_raw),
                    )

                cur.execute(
                    "DELETE FROM tracks WHERE discogs_release_id = %s",
                    (rel.discogs_release_id,),
                )
                for track in rel.tracks:
                    cur.execute(
                        """
                        INSERT INTO tracks (
                            discogs_release_id, track_order, disc_number, disc_label,
                            position, entry_type, parent_position, title,
                            duration_raw, duration_sec, performing_artists_json,
                            performing_artist_display, has_valid_duration, has_resolved_performer,
                            isrc, isrc_source, isrc_source_url,
                            duration_source, duration_source_url,
                            performer_source, performer_source_url,
                            enrichment_notes_json
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        RETURNING id
                        """,
                        (
                            rel.discogs_release_id,
                            track.track_order,
                            track.disc_number,
                            track.disc_label,
                            track.position,
                            track.entry_type,
                            track.parent_position,
                            track.title,
                            track.duration_raw,
                            track.duration_sec,
                            Jsonb([a.model_dump() for a in track.performing_artists]),
                            track.performing_artist_display,
                            track.has_valid_duration,
                            track.has_resolved_performer,
                            track.isrc,
                            track.isrc_source,
                            track.isrc_source_url,
                            track.duration_source,
                            track.duration_source_url,
                            track.performer_source,
                            track.performer_source_url,
                            Jsonb(track.enrichment_notes),
                        ),
                    )
                    track_id = cur.fetchone()[0]
                    for credit in track.credits:
                        cur.execute(
                            """
                            INSERT INTO track_credits (
                                track_id, discogs_artist_id, artist_name, anv,
                                role_raw, role_category, credit_source, source_url
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (track_id, artist_name, role_raw, credit_source) DO NOTHING
                            """,
                            (
                                track_id,
                                credit.discogs_artist_id,
                                credit.artist_name,
                                credit.anv,
                                credit.role_raw,
                                credit.role_category,
                                credit.credit_source,
                                credit.source_url,
                            ),
                        )
            conn.commit()

    def get_extracted_release(self, discogs_release_id: int) -> ExtractedRelease | None:
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT * FROM releases WHERE discogs_release_id = %s",
                    (discogs_release_id,),
                )
                rel_row = cur.fetchone()
                if rel_row is None:
                    return None

                cur.execute(
                    "SELECT barcode_norm, barcode_raw FROM release_barcodes WHERE discogs_release_id = %s",
                    (discogs_release_id,),
                )
                barcode_rows = cur.fetchall()

                cur.execute(
                    "SELECT * FROM tracks WHERE discogs_release_id = %s ORDER BY track_order ASC",
                    (discogs_release_id,),
                )
                track_rows = cur.fetchall()

                track_ids = [r["id"] for r in track_rows]
                credits_by_track: dict[int, list[ExtractedCredit]] = {tid: [] for tid in track_ids}
                if track_ids:
                    cur.execute(
                        "SELECT * FROM track_credits WHERE track_id = ANY(%s) ORDER BY id ASC",
                        (track_ids,),
                    )
                    for cr in cur.fetchall():
                        credits_by_track[cr["track_id"]].append(
                            ExtractedCredit(
                                discogs_artist_id=cr["discogs_artist_id"],
                                artist_name=cr["artist_name"],
                                anv=cr["anv"],
                                role_raw=cr["role_raw"],
                                role_category=cr["role_category"],
                                credit_source=cr["credit_source"],
                                source_url=str(cr.get("source_url") or ""),
                            )
                        )

        tracks: list[ExtractedTrack] = []
        for tr in track_rows:
            artists = [
                TrackArtistOut(**a) for a in (tr["performing_artists_json"] or [])
            ]
            tracks.append(
                ExtractedTrack(
                    track_order=tr["track_order"],
                    disc_number=tr["disc_number"],
                    disc_label=tr["disc_label"],
                    position=tr["position"],
                    entry_type=tr["entry_type"],
                    parent_position=tr["parent_position"],
                    title=tr["title"],
                    duration_raw=tr["duration_raw"],
                    duration_sec=tr["duration_sec"],
                    performing_artists=artists,
                    performing_artist_display=tr["performing_artist_display"],
                    has_valid_duration=tr["has_valid_duration"],
                    has_resolved_performer=tr["has_resolved_performer"],
                    credits=credits_by_track.get(tr["id"], []),
                    isrc=str(tr.get("isrc") or ""),
                    isrc_source=str(tr.get("isrc_source") or ""),
                    isrc_source_url=str(tr.get("isrc_source_url") or ""),
                    duration_source=str(tr.get("duration_source") or ""),
                    duration_source_url=str(tr.get("duration_source_url") or ""),
                    performer_source=str(tr.get("performer_source") or ""),
                    performer_source_url=str(tr.get("performer_source_url") or ""),
                    enrichment_notes=list(tr.get("enrichment_notes_json") or []),
                )
            )

        return ExtractedRelease(
            discogs_release_id=int(rel_row["discogs_release_id"]),
            master_id=rel_row["master_id"],
            url=rel_row["url"],
            resource_url=rel_row["resource_url"],
            title=rel_row["title"],
            title_norm=rel_row["title_norm"],
            artist_display=rel_row["artist_display"],
            artist_norm=rel_row["artist_norm"],
            year=rel_row["year"],
            country=rel_row["country"],
            formats=rel_row["formats_json"] or [],
            identifiers=rel_row["identifiers_json"] or [],
            labels=rel_row["labels_json"] or [],
            data_quality=rel_row["data_quality"],
            community_status=rel_row["community_status"],
            community_have=int(rel_row["community_have"]),
            community_want=int(rel_row["community_want"]),
            notes=rel_row["notes"],
            track_count=int(rel_row["track_count"]),
            tracklist_hash=rel_row["tracklist_hash"],
            unscoped_credits=rel_row["unscoped_credits_json"] or [],
            raw_json=rel_row["raw_json"] or {},
            barcode_entries=[(b["barcode_norm"], b["barcode_raw"]) for b in barcode_rows],
            tracks=tracks,
            source_provider=str(rel_row.get("source_provider") or "discogs"),
        )

    def find_latest_run(
        self, input_id: int, statuses: tuple[str, ...] | None = None
    ) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                if statuses:
                    cur.execute(
                        """
                        SELECT * FROM lookup_runs
                        WHERE input_id = %s AND status = ANY(%s)
                        ORDER BY created_at DESC, id DESC
                        LIMIT 1
                        """,
                        (input_id, list(statuses)),
                    )
                else:
                    cur.execute(
                        """
                        SELECT * FROM lookup_runs
                        WHERE input_id = %s
                        ORDER BY created_at DESC, id DESC
                        LIMIT 1
                        """,
                        (input_id,),
                    )
                return cur.fetchone()

    def get_input_matches(self, input_id: int) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT * FROM input_matches
                    WHERE input_id = %s
                    ORDER BY is_selected DESC, match_rank ASC
                    """,
                    (input_id,),
                )
                return cur.fetchall()

    def find_release_ids_by_barcodes(self, variants: list[str]) -> list[int]:
        if not variants:
            return []
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT discogs_release_id
                    FROM release_barcodes
                    WHERE barcode_norm = ANY(%s)
                    """,
                    (variants,),
                )
                return [int(r[0]) for r in cur.fetchall()]

    def find_release_ids_by_artist_title(
        self, artist_norm: str, title_norm: str, is_compilation: bool = False
    ) -> list[int]:
        if not title_norm:
            return []
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                if is_compilation or not artist_norm:
                    cur.execute(
                        """
                        SELECT discogs_release_id
                        FROM releases
                        WHERE title_norm = %s
                        LIMIT 10
                        """,
                        (title_norm,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT discogs_release_id
                        FROM releases
                        WHERE title_norm = %s AND artist_norm = %s
                        LIMIT 10
                        """,
                        (title_norm, artist_norm),
                    )
                return [int(r[0]) for r in cur.fetchall()]

    def save_agent_fallback(
        self,
        input_id: int,
        model_name: str,
        sources: list[str],
        tracks: list[tuple[int, TrackOut]],
        tracklist_hash: str,
        raw_json: dict[str, Any],
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> AgentFallbackOut:
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    INSERT INTO agent_fallbacks (
                        input_id, model_name, sources_json, track_count, tracklist_hash,
                        input_tokens, output_tokens, cost_usd, raw_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, created_at
                    """,
                    (
                        input_id,
                        model_name,
                        Jsonb(sources),
                        len(tracks),
                        tracklist_hash,
                        input_tokens,
                        output_tokens,
                        cost_usd,
                        Jsonb(raw_json),
                    ),
                )
                fb_row = cur.fetchone()
                fb_id = int(fb_row["id"])
                for track_number, track in tracks:
                    credits_dump = {
                        bucket: [c.model_dump() for c in track.credits.get(bucket, [])]
                        for bucket in ("composers", "lyricists", "producers", "featured", "other")
                    }
                    cur.execute(
                        """
                        INSERT INTO agent_fallback_tracks (
                            agent_fallback_id, track_order, disc_number, track_number,
                            position, title, duration_raw, duration_sec,
                            performing_artists_json, performing_artist_display,
                            has_resolved_performer, isrc, isrc_source, isrc_source_url,
                            duration_source, duration_source_url,
                            performer_source, performer_source_url,
                            credits_json, enrichment_notes_json
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        """,
                        (
                            fb_id,
                            track.track_order,
                            track.disc_number,
                            track_number,
                            track.position,
                            track.title,
                            track.duration_raw,
                            track.duration_sec,
                            Jsonb([a.model_dump() for a in track.performing_artists]),
                            track.performing_artist_display,
                            track.has_resolved_performer,
                            track.isrc,
                            track.isrc_source,
                            track.isrc_source_url,
                            track.duration_source,
                            track.duration_source_url,
                            track.performer_source,
                            track.performer_source_url,
                            Jsonb(credits_dump),
                            Jsonb(track.enrichment_notes),
                        ),
                    )
            conn.commit()
        return AgentFallbackOut(
            agent_fallback_id=fb_id,
            model_name=model_name,
            sources=list(sources),
            track_count=len(tracks),
            tracklist_hash=tracklist_hash,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            created_at=fb_row["created_at"].isoformat(),
        )

    def get_agent_fallback(
        self, agent_fallback_id: int
    ) -> tuple[AgentFallbackOut, list[TrackOut]] | None:
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT * FROM agent_fallbacks WHERE id = %s",
                    (agent_fallback_id,),
                )
                fb_row = cur.fetchone()
                if fb_row is None:
                    return None
                cur.execute(
                    """
                    SELECT * FROM agent_fallback_tracks
                    WHERE agent_fallback_id = %s
                    ORDER BY track_order ASC
                    """,
                    (agent_fallback_id,),
                )
                track_rows = cur.fetchall()

        has_multi_disc = any(int(tr["disc_number"]) > 1 for tr in track_rows)
        tracks_out: list[TrackOut] = []
        for tr in track_rows:
            raw_credits = tr.get("credits_json") or {}
            credits_out: dict[str, list[TrackCreditOut]] = {}
            for bucket in ("composers", "lyricists", "producers", "featured", "other"):
                credits_out[bucket] = [
                    TrackCreditOut(**c) if isinstance(c, dict) else c
                    for c in (raw_credits.get(bucket) or [])
                ]
            dur_sec = tr["duration_sec"]
            has_valid_dur = dur_sec is not None and int(dur_sec) > 0
            artists = [
                TrackArtistOut(**a) for a in (tr.get("performing_artists_json") or [])
            ]
            tracks_out.append(
                TrackOut(
                    track_order=int(tr["track_order"]),
                    disc_number=int(tr["disc_number"]),
                    disc_label=f"Disc {tr['disc_number']}" if has_multi_disc else "",
                    position=tr["position"],
                    entry_type="track",
                    parent_position="",
                    title=tr["title"],
                    duration_raw=tr["duration_raw"] or (format_duration_raw(dur_sec) if has_valid_dur else ""),
                    duration_sec=dur_sec,
                    performing_artists=artists,
                    performing_artist_display=str(tr.get("performing_artist_display") or ""),
                    has_valid_duration=has_valid_dur,
                    has_resolved_performer=bool(tr.get("has_resolved_performer", False)),
                    isrc=str(tr.get("isrc") or ""),
                    isrc_source=str(tr.get("isrc_source") or ""),
                    isrc_source_url=str(tr.get("isrc_source_url") or ""),
                    duration_source=str(tr.get("duration_source") or ""),
                    duration_source_url=str(tr.get("duration_source_url") or ""),
                    performer_source=str(tr.get("performer_source") or ""),
                    performer_source_url=str(tr.get("performer_source_url") or ""),
                    enrichment_notes=list(tr.get("enrichment_notes_json") or []),
                    credits=credits_out,
                )
            )
        fb_out = AgentFallbackOut(
            agent_fallback_id=int(fb_row["id"]),
            model_name=fb_row["model_name"],
            sources=list(fb_row["sources_json"] or []),
            track_count=int(fb_row["track_count"]),
            tracklist_hash=fb_row["tracklist_hash"],
            input_tokens=int(fb_row.get("input_tokens") or 0),
            output_tokens=int(fb_row.get("output_tokens") or 0),
            cost_usd=float(fb_row.get("cost_usd") or 0.0),
            created_at=fb_row["created_at"].isoformat(),
        )
        return fb_out, tracks_out

    def save_matches_and_run(
        self,
        input_id: int,
        status: str,
        http_code: int,
        source: str,
        selected_release_id: int | None,
        queries_attempted: list[dict[str, Any]],
        candidates_checked: int,
        evaluation_summary: dict[str, Any],
        review_reasons: list[str],
        matches: list[dict[str, Any]],
        error_detail: str = "",
        replace_matches: bool = True,
        selected_agent_fallback_id: int | None = None,
    ) -> int:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                if replace_matches:
                    cur.execute("DELETE FROM input_matches WHERE input_id = %s", (input_id,))
                    for m in matches:
                        cur.execute(
                            """
                            INSERT INTO input_matches (
                                input_id, discogs_release_id, is_selected,
                                match_rank, match_method, score, evaluation_json
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (input_id, discogs_release_id) DO UPDATE SET
                                is_selected = EXCLUDED.is_selected,
                                match_rank = EXCLUDED.match_rank,
                                match_method = EXCLUDED.match_method,
                                score = EXCLUDED.score,
                                evaluation_json = EXCLUDED.evaluation_json
                            """,
                            (
                                input_id,
                                m["discogs_release_id"],
                                m["is_selected"],
                                m["match_rank"],
                                m["match_method"],
                                m["score"],
                                Jsonb(m.get("evaluation_json", {})),
                            ),
                        )

                cur.execute(
                    """
                    INSERT INTO lookup_runs (
                        input_id, status, http_code, source, selected_release_id,
                        selected_agent_fallback_id, queries_attempted_json, candidates_checked,
                        evaluation_summary_json, review_reasons_json, error_detail
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        input_id,
                        status,
                        http_code,
                        source,
                        selected_release_id,
                        selected_agent_fallback_id,
                        Jsonb(queries_attempted),
                        candidates_checked,
                        Jsonb(evaluation_summary),
                        Jsonb(review_reasons),
                        error_detail,
                    ),
                )
                run_id = int(cur.fetchone()[0])
            conn.commit()
        return run_id

    def list_records(self, limit: int = 100, status: str | None = None) -> list[RecordSummaryOut]:
        where_clause = "WHERE lr.status = %s" if status else ""
        params: list[Any] = [status, limit] if status else [limit]
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT
                        ir.id AS input_id,
                        ir.barcode_raw,
                        ir.album_title_raw,
                        ir.album_artist_raw,
                        ir.barcode_norm,
                        ir.title_norm,
                        ir.artist_norm,
                        ir.format_hint,
                        ir.is_compilation,
                        lr.status AS latest_status,
                        lr.http_code AS latest_http_code,
                        lr.source AS latest_source,
                        lr.selected_release_id,
                        r.url AS selected_release_url,
                        lr.selected_agent_fallback_id,
                        af.sources_json->>0 AS agent_source_url,
                        COALESCE(af.input_tokens, 0) + COALESCE(r.enrichment_input_tokens, 0) AS agent_input_tokens,
                        COALESCE(af.output_tokens, 0) + COALESCE(r.enrichment_output_tokens, 0) AS agent_output_tokens,
                        COALESCE(af.cost_usd, 0.0) + COALESCE(r.enrichment_cost_usd, 0.0) AS agent_cost_usd,
                        lr.created_at AS latest_checked_at
                    FROM input_records ir
                    LEFT JOIN LATERAL (
                        SELECT * FROM lookup_runs
                        WHERE input_id = ir.id
                        ORDER BY created_at DESC, id DESC
                        LIMIT 1
                    ) lr ON TRUE
                    LEFT JOIN releases r ON r.discogs_release_id = lr.selected_release_id
                    LEFT JOIN agent_fallbacks af ON af.id = lr.selected_agent_fallback_id
                    {where_clause}
                    ORDER BY ir.id DESC
                    LIMIT %s
                    """,
                    tuple(params),
                )
                rows = cur.fetchall()

        return [
            RecordSummaryOut(
                input_id=int(row["input_id"]),
                barcode_raw=row["barcode_raw"],
                album_title_raw=row["album_title_raw"],
                album_artist_raw=row["album_artist_raw"],
                barcode_norm=row["barcode_norm"],
                title_norm=row["title_norm"],
                artist_norm=row["artist_norm"],
                format_hint=row["format_hint"] or {},
                is_compilation=bool(row["is_compilation"]),
                latest_status=row["latest_status"],
                latest_http_code=row["latest_http_code"],
                latest_source=row["latest_source"],
                selected_release_id=row["selected_release_id"],
                selected_release_url=row["selected_release_url"],
                selected_agent_fallback_id=row["selected_agent_fallback_id"],
                agent_source_url=row["agent_source_url"],
                agent_input_tokens=int(row.get("agent_input_tokens") or 0),
                agent_output_tokens=int(row.get("agent_output_tokens") or 0),
                agent_cost_usd=float(row.get("agent_cost_usd") or 0.0),
                latest_checked_at=(
                    row["latest_checked_at"].isoformat() if row["latest_checked_at"] else None
                ),
            )
            for row in rows
        ]

    def get_record_detail(self, input_id: int) -> RecordDetailOut | None:
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT
                        ir.id AS input_id,
                        ir.barcode_raw,
                        ir.album_title_raw,
                        ir.album_artist_raw,
                        ir.barcode_norm,
                        ir.title_norm,
                        ir.artist_norm,
                        ir.format_hint,
                        ir.is_compilation,
                        lr.status AS latest_status,
                        lr.http_code AS latest_http_code,
                        lr.source AS latest_source,
                        lr.selected_release_id,
                        r.url AS selected_release_url,
                        lr.selected_agent_fallback_id,
                        af.sources_json->>0 AS agent_source_url,
                        COALESCE(af.input_tokens, 0) + COALESCE(r.enrichment_input_tokens, 0) AS agent_input_tokens,
                        COALESCE(af.output_tokens, 0) + COALESCE(r.enrichment_output_tokens, 0) AS agent_output_tokens,
                        COALESCE(af.cost_usd, 0.0) + COALESCE(r.enrichment_cost_usd, 0.0) AS agent_cost_usd,
                        lr.created_at AS latest_checked_at
                    FROM input_records ir
                    LEFT JOIN LATERAL (
                        SELECT * FROM lookup_runs
                        WHERE input_id = ir.id
                        ORDER BY created_at DESC, id DESC
                        LIMIT 1
                    ) lr ON TRUE
                    LEFT JOIN releases r ON r.discogs_release_id = lr.selected_release_id
                    LEFT JOIN agent_fallbacks af ON af.id = lr.selected_agent_fallback_id
                    WHERE ir.id = %s
                    """,
                    (input_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None

                cur.execute(
                    """
                    SELECT id, status, http_code, source, selected_release_id,
                           selected_agent_fallback_id, queries_attempted_json,
                           candidates_checked, evaluation_summary_json, review_reasons_json,
                           error_detail, created_at
                    FROM lookup_runs
                    WHERE input_id = %s
                    ORDER BY created_at DESC, id DESC
                    """,
                    (input_id,),
                )
                runs = [
                    {**r, "created_at": r["created_at"].isoformat()}
                    for r in cur.fetchall()
                ]

                cur.execute(
                    """
                    SELECT im.discogs_release_id, im.is_selected, im.match_rank,
                           im.match_method, im.score, im.evaluation_json,
                           r.url, r.title, r.artist_display
                    FROM input_matches im
                    JOIN releases r ON r.discogs_release_id = im.discogs_release_id
                    WHERE im.input_id = %s
                    ORDER BY im.is_selected DESC, im.match_rank ASC
                    """,
                    (input_id,),
                )
                matches = cur.fetchall()

                cur.execute(
                    """
                    SELECT id, input_id, decision, reviewer_type, reviewer_name,
                           selected_release_id, selected_agent_fallback_id, note, created_at
                    FROM record_reviews
                    WHERE input_id = %s
                    ORDER BY created_at DESC, id DESC
                    """,
                    (input_id,),
                )
                reviews = [
                    ReviewOut(
                        id=int(rv["id"]),
                        input_id=int(rv["input_id"]),
                        decision=rv["decision"],
                        reviewer_type=rv["reviewer_type"],
                        reviewer_name=rv["reviewer_name"],
                        selected_release_id=rv["selected_release_id"],
                        selected_agent_fallback_id=rv["selected_agent_fallback_id"],
                        note=rv["note"],
                        created_at=rv["created_at"].isoformat(),
                    )
                    for rv in cur.fetchall()
                ]

        selected_release_out: ReleaseOut | None = None
        agent_fallback_out: AgentFallbackOut | None = None
        tracks_out: list[TrackOut] = []
        if row["selected_release_id"] is not None:
            extracted = self.get_extracted_release(int(row["selected_release_id"]))
            if extracted is not None:
                selected_release_out = ReleaseOut(
                    discogs_release_id=extracted.discogs_release_id,
                    master_id=extracted.master_id,
                    url=extracted.url,
                    title=extracted.title,
                    artist_display=extracted.artist_display,
                    year=extracted.year,
                    country=extracted.country,
                    formats=extracted.formats,
                    identifiers=extracted.identifiers,
                    labels=extracted.labels,
                    data_quality=extracted.data_quality,
                    community_status=extracted.community_status,
                    community_have=extracted.community_have,
                    tracklist_hash=extracted.tracklist_hash,
                    unscoped_credits=extracted.unscoped_credits,
                )
                tracks_out = [extracted_track_to_out(t) for t in extracted.tracks]
        elif row["selected_agent_fallback_id"] is not None:
            fb_loaded = self.get_agent_fallback(int(row["selected_agent_fallback_id"]))
            if fb_loaded is not None:
                agent_fallback_out, tracks_out = fb_loaded

        return RecordDetailOut(
            record=RecordSummaryOut(
                input_id=int(row["input_id"]),
                barcode_raw=row["barcode_raw"],
                album_title_raw=row["album_title_raw"],
                album_artist_raw=row["album_artist_raw"],
                barcode_norm=row["barcode_norm"],
                title_norm=row["title_norm"],
                artist_norm=row["artist_norm"],
                format_hint=row["format_hint"] or {},
                is_compilation=bool(row["is_compilation"]),
                latest_status=row["latest_status"],
                latest_http_code=row["latest_http_code"],
                latest_source=row["latest_source"],
                selected_release_id=row["selected_release_id"],
                selected_release_url=row["selected_release_url"],
                selected_agent_fallback_id=row["selected_agent_fallback_id"],
                agent_source_url=row["agent_source_url"],
                agent_input_tokens=int(row.get("agent_input_tokens") or 0),
                agent_output_tokens=int(row.get("agent_output_tokens") or 0),
                agent_cost_usd=float(row.get("agent_cost_usd") or 0.0),
                latest_checked_at=(
                    row["latest_checked_at"].isoformat() if row["latest_checked_at"] else None
                ),
            ),
            runs=runs,
            matches=matches,
            reviews=reviews,
            selected_release=selected_release_out,
            agent_fallback=agent_fallback_out,
            tracks=tracks_out,
        )

    def record_review(self, input_id: int, req: ReviewRequest) -> ReviewOut | None:
        if req.selected_release_id is not None and req.selected_agent_fallback_id is not None:
            raise ValueError(
                "Specify either selected_release_id or selected_agent_fallback_id, not both."
            )

        decision_to_status = {
            "confirm": "matched",
            "deny": "rejected",
            "request_review": "needs_review",
        }
        new_status = decision_to_status[req.decision]

        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT id FROM input_records WHERE id = %s", (input_id,))
                if cur.fetchone() is None:
                    return None

                cur.execute(
                    """
                    SELECT selected_release_id, selected_agent_fallback_id, candidates_checked
                    FROM lookup_runs
                    WHERE input_id = %s
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """,
                    (input_id,),
                )
                latest_run = cur.fetchone()
                prev_release_id = (
                    int(latest_run["selected_release_id"])
                    if latest_run and latest_run["selected_release_id"] is not None
                    else None
                )
                prev_agent_id = (
                    int(latest_run["selected_agent_fallback_id"])
                    if latest_run and latest_run["selected_agent_fallback_id"] is not None
                    else None
                )
                candidates_checked = (
                    int(latest_run["candidates_checked"]) if latest_run else 0
                )

                if req.selected_release_id is not None:
                    cur.execute(
                        "SELECT discogs_release_id FROM releases WHERE discogs_release_id = %s",
                        (req.selected_release_id,),
                    )
                    if cur.fetchone() is None:
                        raise ValueError(
                            f"Release {req.selected_release_id} not found in database."
                        )

                if req.selected_agent_fallback_id is not None:
                    cur.execute(
                        "SELECT id FROM agent_fallbacks WHERE id = %s AND input_id = %s",
                        (req.selected_agent_fallback_id, input_id),
                    )
                    if cur.fetchone() is None:
                        raise ValueError(
                            f"Agent fallback {req.selected_agent_fallback_id} not found for input {input_id}."
                        )

                target_release_id: int | None = None
                target_agent_id: int | None = None

                if req.decision == "confirm":
                    if req.selected_release_id is not None:
                        target_release_id = req.selected_release_id
                    elif req.selected_agent_fallback_id is not None:
                        target_agent_id = req.selected_agent_fallback_id
                    elif prev_release_id is not None:
                        target_release_id = prev_release_id
                    elif prev_agent_id is not None:
                        target_agent_id = prev_agent_id
                    else:
                        raise ValueError(
                            "Cannot confirm a record without a selected release or agent fallback."
                        )

                    if target_release_id is not None:
                        cur.execute(
                            """
                            UPDATE input_matches
                            SET is_selected = (discogs_release_id = %s)
                            WHERE input_id = %s
                            """,
                            (target_release_id, input_id),
                        )
                    else:
                        cur.execute(
                            "UPDATE input_matches SET is_selected = FALSE WHERE input_id = %s",
                            (input_id,),
                        )
                elif req.decision == "deny":
                    cur.execute(
                        "UPDATE input_matches SET is_selected = FALSE WHERE input_id = %s",
                        (input_id,),
                    )
                else:
                    if req.selected_release_id is not None:
                        target_release_id = req.selected_release_id
                    elif req.selected_agent_fallback_id is not None:
                        target_agent_id = req.selected_agent_fallback_id
                    else:
                        target_release_id = prev_release_id
                        target_agent_id = prev_agent_id

                review_reasons = (
                    [req.note.strip() or "Flagged for review"]
                    if req.decision == "request_review"
                    else []
                )

                cur.execute(
                    """
                    INSERT INTO record_reviews (
                        input_id, decision, reviewer_type, reviewer_name,
                        selected_release_id, selected_agent_fallback_id, note
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, input_id, decision, reviewer_type, reviewer_name,
                              selected_release_id, selected_agent_fallback_id, note, created_at
                    """,
                    (
                        input_id,
                        req.decision,
                        req.reviewer_type,
                        req.reviewer_name.strip(),
                        target_release_id if req.decision != "deny" else req.selected_release_id,
                        target_agent_id if req.decision != "deny" else req.selected_agent_fallback_id,
                        req.note.strip(),
                    ),
                )
                rv = cur.fetchone()

                cur.execute(
                    """
                    INSERT INTO lookup_runs (
                        input_id, status, http_code, source, selected_release_id,
                        selected_agent_fallback_id, queries_attempted_json, candidates_checked,
                        evaluation_summary_json, review_reasons_json, error_detail
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        input_id,
                        new_status,
                        200,
                        f"review:{req.reviewer_type}",
                        target_release_id,
                        target_agent_id,
                        Jsonb([]),
                        candidates_checked,
                        Jsonb(
                            {
                                "review_decision": req.decision,
                                "reviewer_name": req.reviewer_name.strip(),
                            }
                        ),
                        Jsonb(review_reasons),
                        req.note.strip(),
                    ),
                )
            conn.commit()

        return ReviewOut(
            id=int(rv["id"]),
            input_id=int(rv["input_id"]),
            decision=rv["decision"],
            reviewer_type=rv["reviewer_type"],
            reviewer_name=rv["reviewer_name"],
            selected_release_id=rv["selected_release_id"],
            selected_agent_fallback_id=rv["selected_agent_fallback_id"],
            note=rv["note"],
            created_at=rv["created_at"].isoformat(),
        )

    # ------------------------------------------------------------------ #
    # Track enrichment persistence & coverage refresh
    # ------------------------------------------------------------------ #

    def update_release_track_enrichments(
        self,
        discogs_release_id: int,
        tracks: list[TrackOut],
        mark_enriched: bool = True,
        add_input_tokens: int = 0,
        add_output_tokens: int = 0,
        add_cost_usd: float = 0.0,
    ) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                for track in tracks:
                    cur.execute(
                        """
                        UPDATE tracks
                        SET duration_raw = %s,
                            duration_sec = %s,
                            has_valid_duration = %s,
                            performing_artists_json = %s,
                            performing_artist_display = %s,
                            has_resolved_performer = %s,
                            isrc = %s,
                            isrc_source = %s,
                            isrc_source_url = %s,
                            duration_source = %s,
                            duration_source_url = %s,
                            performer_source = %s,
                            performer_source_url = %s,
                            enrichment_notes_json = %s
                        WHERE discogs_release_id = %s AND track_order = %s
                        RETURNING id
                        """,
                        (
                            track.duration_raw,
                            track.duration_sec,
                            track.has_valid_duration,
                            Jsonb([a.model_dump() for a in track.performing_artists]),
                            track.performing_artist_display,
                            track.has_resolved_performer,
                            track.isrc,
                            track.isrc_source,
                            track.isrc_source_url,
                            track.duration_source,
                            track.duration_source_url,
                            track.performer_source,
                            track.performer_source_url,
                            Jsonb(track.enrichment_notes),
                            discogs_release_id,
                            track.track_order,
                        ),
                    )
                    row = cur.fetchone()
                    if row is None:
                        continue
                    track_id = int(row[0])
                    for bucket_credits in track.credits.values():
                        for credit in bucket_credits:
                            cur.execute(
                                """
                                INSERT INTO track_credits (
                                    track_id, discogs_artist_id, artist_name, anv,
                                    role_raw, role_category, credit_source, source_url
                                )
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (track_id, artist_name, role_raw, credit_source) DO NOTHING
                                """,
                                (
                                    track_id,
                                    credit.discogs_artist_id,
                                    credit.artist_name,
                                    credit.anv,
                                    credit.role_raw,
                                    credit.role_category,
                                    credit.credit_source,
                                    credit.source_url,
                                ),
                            )
                if mark_enriched or add_input_tokens or add_output_tokens or add_cost_usd:
                    cur.execute(
                        """
                        UPDATE releases
                        SET enriched_at = CASE WHEN %s THEN NOW() ELSE enriched_at END,
                            enrichment_input_tokens = enrichment_input_tokens + %s,
                            enrichment_output_tokens = enrichment_output_tokens + %s,
                            enrichment_cost_usd = enrichment_cost_usd + %s
                        WHERE discogs_release_id = %s
                        """,
                        (
                            mark_enriched,
                            add_input_tokens,
                            add_output_tokens,
                            add_cost_usd,
                            discogs_release_id,
                        ),
                    )
            conn.commit()

    def update_agent_fallback_track_enrichments(
        self,
        agent_fallback_id: int,
        tracks: list[TrackOut],
        mark_enriched: bool = True,
        add_input_tokens: int = 0,
        add_output_tokens: int = 0,
        add_cost_usd: float = 0.0,
    ) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                for track in tracks:
                    credits_dump = {
                        bucket: [c.model_dump() for c in track.credits.get(bucket, [])]
                        for bucket in ("composers", "lyricists", "producers", "featured", "other")
                    }
                    cur.execute(
                        """
                        UPDATE agent_fallback_tracks
                        SET duration_raw = %s,
                            duration_sec = %s,
                            performing_artists_json = %s,
                            performing_artist_display = %s,
                            has_resolved_performer = %s,
                            isrc = %s,
                            isrc_source = %s,
                            isrc_source_url = %s,
                            duration_source = %s,
                            duration_source_url = %s,
                            performer_source = %s,
                            performer_source_url = %s,
                            credits_json = %s,
                            enrichment_notes_json = %s
                        WHERE agent_fallback_id = %s AND track_order = %s
                        """,
                        (
                            track.duration_raw,
                            track.duration_sec,
                            Jsonb([a.model_dump() for a in track.performing_artists]),
                            track.performing_artist_display,
                            track.has_resolved_performer,
                            track.isrc,
                            track.isrc_source,
                            track.isrc_source_url,
                            track.duration_source,
                            track.duration_source_url,
                            track.performer_source,
                            track.performer_source_url,
                            Jsonb(credits_dump),
                            Jsonb(track.enrichment_notes),
                            agent_fallback_id,
                            track.track_order,
                        ),
                    )
                if mark_enriched or add_input_tokens or add_output_tokens or add_cost_usd:
                    cur.execute(
                        """
                        UPDATE agent_fallbacks
                        SET enriched_at = CASE WHEN %s THEN NOW() ELSE enriched_at END,
                            input_tokens = input_tokens + %s,
                            output_tokens = output_tokens + %s,
                            cost_usd = cost_usd + %s
                        WHERE id = %s
                        """,
                        (
                            mark_enriched,
                            add_input_tokens,
                            add_output_tokens,
                            add_cost_usd,
                            agent_fallback_id,
                        ),
                    )
            conn.commit()

    def is_target_already_enriched(
        self,
        selected_release_id: int | None,
        selected_agent_fallback_id: int | None,
    ) -> bool:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                if selected_release_id is not None:
                    cur.execute(
                        "SELECT enriched_at IS NOT NULL FROM releases WHERE discogs_release_id = %s",
                        (selected_release_id,),
                    )
                    row = cur.fetchone()
                    return bool(row and row[0])
                if selected_agent_fallback_id is not None:
                    cur.execute(
                        "SELECT enriched_at IS NOT NULL FROM agent_fallbacks WHERE id = %s",
                        (selected_agent_fallback_id,),
                    )
                    row = cur.fetchone()
                    return bool(row and row[0])
        return False

    def find_sibling_releases(
        self, input_id: int, selected_release_id: int, master_id: int | None
    ) -> list[ExtractedRelease]:
        sibling_ids: list[int] = []
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT discogs_release_id
                    FROM input_matches
                    WHERE input_id = %s AND discogs_release_id <> %s
                    ORDER BY match_rank ASC
                    LIMIT 5
                    """,
                    (input_id, selected_release_id),
                )
                for (rid,) in cur.fetchall():
                    if int(rid) not in sibling_ids:
                        sibling_ids.append(int(rid))

                if master_id is not None:
                    cur.execute(
                        """
                        SELECT discogs_release_id
                        FROM releases
                        WHERE master_id = %s AND discogs_release_id <> %s
                        LIMIT 5
                        """,
                        (master_id, selected_release_id),
                    )
                    for (rid,) in cur.fetchall():
                        if int(rid) not in sibling_ids:
                            sibling_ids.append(int(rid))

        releases: list[ExtractedRelease] = []
        for rid in sibling_ids[:6]:
            rel = self.get_extracted_release(rid)
            if rel is not None:
                releases.append(rel)
        return releases

    def list_matched_inputs_with_missing_track_data(
        self, limit: int = 100, force_refresh: bool = False
    ) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT
                        ir.*,
                        lr.selected_release_id,
                        lr.selected_agent_fallback_id,
                        lr.status AS latest_status,
                        lr.source AS latest_source
                    FROM input_records ir
                    JOIN LATERAL (
                        SELECT status, source, selected_release_id, selected_agent_fallback_id
                        FROM lookup_runs
                        WHERE input_id = ir.id
                        ORDER BY created_at DESC, id DESC
                        LIMIT 1
                    ) lr ON TRUE
                    LEFT JOIN releases r ON r.discogs_release_id = lr.selected_release_id
                    LEFT JOIN agent_fallbacks af ON af.id = lr.selected_agent_fallback_id
                    WHERE lr.status IN ('matched', 'needs_review')
                      AND NOT EXISTS (
                          SELECT 1 FROM queue_items qi
                          WHERE qi.input_id = ir.id
                            AND qi.stage = ANY(%s)
                      )
                      AND (
                          (
                              lr.selected_release_id IS NOT NULL
                              AND r.discogs_release_id IS NOT NULL
                              AND (%s OR r.enriched_at IS NULL)
                              AND EXISTS (
                                  SELECT 1 FROM tracks t
                                  WHERE t.discogs_release_id = lr.selected_release_id
                                    AND (
                                        NOT t.has_valid_duration
                                        OR NOT t.has_resolved_performer
                                        OR t.isrc = ''
                                        OR NOT EXISTS (
                                            SELECT 1 FROM track_credits tc
                                            WHERE tc.track_id = t.id
                                              AND tc.role_category IN ('composer', 'lyricist')
                                        )
                                        OR NOT EXISTS (
                                            SELECT 1 FROM track_credits tc
                                            WHERE tc.track_id = t.id
                                              AND tc.role_category = 'producer'
                                        )
                                    )
                              )
                          )
                          OR (
                              lr.selected_agent_fallback_id IS NOT NULL
                              AND af.id IS NOT NULL
                              AND (%s OR af.enriched_at IS NULL)
                              AND EXISTS (
                                  SELECT 1 FROM agent_fallback_tracks aft
                                  WHERE aft.agent_fallback_id = lr.selected_agent_fallback_id
                                    AND (
                                        aft.duration_sec IS NULL
                                        OR aft.duration_sec <= 0
                                        OR NOT aft.has_resolved_performer
                                        OR aft.isrc = ''
                                        OR COALESCE(jsonb_array_length(aft.credits_json->'composers'), 0)
                                         + COALESCE(jsonb_array_length(aft.credits_json->'lyricists'), 0) = 0
                                        OR COALESCE(jsonb_array_length(aft.credits_json->'producers'), 0) = 0
                                    )
                              )
                          )
                      )
                    ORDER BY ir.id ASC
                    LIMIT %s
                    """,
                    (list(ACTIVE_QUEUE_STAGES), force_refresh, force_refresh, limit),
                )
                return cur.fetchall()

    def refresh_latest_run_coverage(
        self,
        input_id: int,
        duration_cov: float,
        performer_cov: float,
        writer_cov: float,
        producer_cov: float,
    ) -> str | None:
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT id, status, source, evaluation_summary_json, review_reasons_json
                    FROM lookup_runs
                    WHERE input_id = %s
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    FOR UPDATE
                    """,
                    (input_id,),
                )
                run = cur.fetchone()
                if run is None:
                    conn.rollback()
                    return None

                summary = dict(run.get("evaluation_summary_json") or {})
                summary["duration_coverage"] = duration_cov
                summary["performer_coverage"] = performer_cov
                summary["writer_credit_coverage"] = writer_cov
                summary["producer_credit_coverage"] = producer_cov

                reasons = list(run.get("review_reasons_json") or [])
                if performer_cov >= 1.0:
                    reasons = [
                        r
                        for r in reasons
                        if not str(r).startswith("Performing artists missing/unresolved")
                    ]
                    if "review_reasons" in summary:
                        summary["review_reasons"] = reasons

                new_status = str(run["status"])
                if (
                    new_status == "needs_review"
                    and not reasons
                    and not str(run.get("source") or "").startswith("gemini_agent")
                ):
                    new_status = "matched"

                cur.execute(
                    """
                    UPDATE lookup_runs
                    SET status = %s,
                        evaluation_summary_json = %s,
                        review_reasons_json = %s
                    WHERE id = %s
                    """,
                    (new_status, Jsonb(summary), Jsonb(reasons), run["id"]),
                )
            conn.commit()
        return new_status

    # ------------------------------------------------------------------ #
    # Queue operations (PostgreSQL FOR UPDATE SKIP LOCKED)
    # ------------------------------------------------------------------ #

    def get_input_record(self, input_id: int) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT * FROM input_records WHERE id = %s", (input_id,))
                return cur.fetchone()

    def has_active_queue_item(self, input_id: int) -> bool:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1 FROM queue_items
                    WHERE input_id = %s
                      AND stage = ANY(%s)
                    LIMIT 1
                    """,
                    (input_id, list(ACTIVE_QUEUE_STAGES)),
                )
                return cur.fetchone() is not None

    def is_queue_idle(self) -> bool:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1 FROM queue_items
                    WHERE stage = ANY(%s)
                    LIMIT 1
                    """,
                    (list(ACTIVE_QUEUE_STAGES),),
                )
                return cur.fetchone() is None

    def list_not_found_records_for_agent_retry(self) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT ir.*
                    FROM input_records ir
                    JOIN LATERAL (
                        SELECT status
                        FROM lookup_runs
                        WHERE input_id = ir.id
                        ORDER BY created_at DESC, id DESC
                        LIMIT 1
                    ) lr ON TRUE
                    WHERE lr.status = 'not_found'
                      AND NOT EXISTS (
                          SELECT 1 FROM queue_items qi
                          WHERE qi.input_id = ir.id
                            AND qi.stage = ANY(%s)
                      )
                    ORDER BY ir.id ASC
                    """,
                    (list(ACTIVE_QUEUE_STAGES),),
                )
                return cur.fetchall()

    @staticmethod
    def _recompute_job_summary(cur: Any, job_id: int) -> tuple[dict[str, Any], bool]:
        cur.execute(
            """
            SELECT status, started_at, finished_at, total_items,
                   completed_items, output_path, output_json_path
            FROM queue_jobs
            WHERE id = %s
            FOR UPDATE
            """,
            (job_id,),
        )
        job_row = cur.fetchone()
        assert job_row is not None
        prev_status = str(job_row["status"])
        prev_started_at = job_row["started_at"]
        prev_finished_at = job_row["finished_at"]
        prev_completed_items = int(job_row["completed_items"])
        total_items = int(job_row["total_items"])
        has_pending_report = bool(job_row["output_path"]) and not bool(
            job_row["output_json_path"]
        )

        cur.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE stage = 'done') AS done_count,
                COUNT(*) FILTER (WHERE stage = 'cancelled') AS cancelled_count,
                COUNT(*) FILTER (WHERE stage IN ('checking_discogs', 'checking_agent', 'checking_enrich')) AS active_count,
                COUNT(*) FILTER (WHERE stage IN ('queued_discogs', 'queued_agent', 'queued_enrich')) AS queued_count,
                COUNT(*) FILTER (WHERE stage = 'done' AND result_status = 'matched') AS matched_count,
                COUNT(*) FILTER (WHERE stage = 'done' AND result_status = 'needs_review') AS needs_review_count,
                COUNT(*) FILTER (
                    WHERE stage = 'done'
                      AND result_source IN ('gemini_agent', 'gemini_agent_fallback', 'discogs_via_agent')
                ) AS agent_recovered_count,
                COUNT(*) FILTER (WHERE stage = 'done' AND result_status = 'not_found') AS not_found_count,
                COUNT(*) FILTER (
                    WHERE stage = 'done'
                      AND result_status IN ('upstream_error', 'validation_error')
                ) AS error_count
            FROM queue_items
            WHERE job_id = %s
            """,
            (job_id,),
        )
        agg = cur.fetchone()
        done_count = int(agg["done_count"])
        cancelled_count = int(agg["cancelled_count"])
        active_count = int(agg["active_count"])
        queued_count = int(agg["queued_count"])
        completed_items = done_count + cancelled_count

        needs_report_finalize = False
        if prev_status == "cancelled":
            new_status = "cancelled"
            mark_finished = active_count == 0 and queued_count == 0
        else:
            if total_items == 0 or (
                active_count == 0 and queued_count == 0 and completed_items >= total_items
            ):
                if total_items > 0 and has_pending_report:
                    new_status = "running"
                    mark_finished = False
                    needs_report_finalize = (
                        prev_completed_items < total_items and prev_finished_at is None
                    )
                else:
                    new_status = "completed"
                    mark_finished = True
            elif active_count > 0 or completed_items > 0 or prev_started_at is not None:
                new_status = "running"
                mark_finished = False
            else:
                new_status = "queued"
                mark_finished = False

        cur.execute(
            """
            UPDATE queue_jobs
            SET status = %s,
                completed_items = %s,
                matched_count = %s,
                needs_review_count = %s,
                agent_recovered_count = %s,
                not_found_count = %s,
                error_count = %s,
                cancelled_count = %s,
                started_at = CASE
                    WHEN started_at IS NOT NULL THEN started_at
                    WHEN %s IN ('running', 'completed', 'cancelled') THEN NOW()
                    ELSE NULL
                END,
                finished_at = CASE
                    WHEN %s THEN COALESCE(finished_at, NOW())
                    ELSE NULL
                END,
                updated_at = NOW()
            WHERE id = %s
            RETURNING *
            """,
            (
                new_status,
                completed_items,
                int(agg["matched_count"]),
                int(agg["needs_review_count"]),
                int(agg["agent_recovered_count"]),
                int(agg["not_found_count"]),
                int(agg["error_count"]),
                cancelled_count,
                new_status,
                mark_finished,
                job_id,
            ),
        )
        updated = cur.fetchone()
        just_finished = needs_report_finalize or (
            prev_finished_at is None and updated["finished_at"] is not None
        )
        return updated, just_finished

    def create_queue_job(
        self,
        job_type: str,
        label: str,
        items: list[dict[str, Any]],
        csv_path: str = "",
        output_path: str = "",
        force_refresh: bool = False,
    ) -> QueueJobOut:
        total = len(items)
        initial_status = "completed" if total == 0 else "queued"
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    INSERT INTO queue_jobs (
                        job_type, label, status, csv_path, output_path, force_refresh,
                        total_items, completed_items, started_at, finished_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, 0,
                        CASE WHEN %s = 0 THEN NOW() ELSE NULL END,
                        CASE WHEN %s = 0 THEN NOW() ELSE NULL END
                    )
                    RETURNING *
                    """,
                    (
                        job_type,
                        label,
                        initial_status,
                        csv_path,
                        output_path,
                        force_refresh,
                        total,
                        total,
                        total,
                    ),
                )
                job_row = cur.fetchone()
                job_id = int(job_row["id"])

                for idx, item in enumerate(items, start=1):
                    cur.execute(
                        """
                        INSERT INTO queue_items (
                            job_id, row_number, input_id, barcode, album_title, album_artist,
                            force_refresh, mode, stage, status_message, context_json
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            job_id,
                            int(item.get("row_number", idx)),
                            item.get("input_id"),
                            str(item.get("barcode") or ""),
                            str(item.get("album_title") or ""),
                            str(item.get("album_artist") or ""),
                            bool(item.get("force_refresh", force_refresh)),
                            str(item.get("mode", "full")),
                            str(item.get("stage", "queued_discogs")),
                            str(item.get("status_message") or "Waiting in queue"),
                            Jsonb(item.get("context_json") or {}),
                        ),
                    )
            conn.commit()
        return _row_to_queue_job(job_row)

    def claim_next_queue_item(
        self,
        from_stage: str,
        to_stage: str,
        worker_name: str,
        status_message: str,
    ) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT qi.id, qi.job_id
                    FROM queue_items qi
                    JOIN queue_jobs qj ON qj.id = qi.job_id
                    WHERE qi.stage = %s
                      AND qj.status <> 'cancelled'
                    ORDER BY qi.id ASC
                    FOR UPDATE OF qi SKIP LOCKED
                    LIMIT 1
                    """,
                    (from_stage,),
                )
                candidate = cur.fetchone()
                if candidate is None:
                    conn.rollback()
                    return None

                cur.execute(
                    """
                    UPDATE queue_items
                    SET stage = %s,
                        worker_name = %s,
                        status_message = %s,
                        started_at = COALESCE(started_at, NOW()),
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING *
                    """,
                    (to_stage, worker_name, status_message, candidate["id"]),
                )
                claimed = cur.fetchone()

                cur.execute(
                    """
                    UPDATE queue_jobs
                    SET status = 'running',
                        started_at = COALESCE(started_at, NOW()),
                        updated_at = NOW()
                    WHERE id = %s AND status = 'queued'
                    """,
                    (candidate["job_id"],),
                )
            conn.commit()
        return dict(claimed) if claimed is not None else None

    def update_queue_item_progress(
        self,
        item_id: int,
        status_message: str,
        attempt: int | None = None,
        input_id: int | None = None,
    ) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE queue_items
                    SET status_message = %s,
                        attempt = COALESCE(%s, attempt),
                        input_id = COALESCE(%s, input_id),
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (status_message, attempt, input_id, item_id),
                )
            conn.commit()

    def handoff_item_to_agent(
        self,
        item_id: int,
        job_id: int,
        input_id: int,
        context: dict[str, Any],
        status_message: str,
    ) -> tuple[bool, bool]:
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT status FROM queue_jobs WHERE id = %s FOR UPDATE",
                    (job_id,),
                )
                job_row = cur.fetchone()
                if job_row is not None and job_row["status"] == "cancelled":
                    cur.execute(
                        """
                        UPDATE queue_items
                        SET stage = 'cancelled',
                            worker_name = '',
                            input_id = %s,
                            status_message = 'Cancelled before agent stage',
                            finished_at = NOW(),
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (input_id, item_id),
                    )
                    _, just_finished = self._recompute_job_summary(cur, job_id)
                    conn.commit()
                    return False, just_finished

                cur.execute(
                    """
                    UPDATE queue_items
                    SET stage = 'queued_agent',
                        worker_name = '',
                        input_id = %s,
                        context_json = %s,
                        status_message = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (input_id, Jsonb(context), status_message, item_id),
                )
            conn.commit()
        return True, False

    def handoff_item_to_enrich(
        self,
        item_id: int,
        job_id: int,
        input_id: int,
        selected_release_id: int | None,
        selected_agent_fallback_id: int | None,
        track_count: int,
        context: dict[str, Any],
        status_message: str,
    ) -> tuple[bool, bool]:
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT status FROM queue_jobs WHERE id = %s FOR UPDATE",
                    (job_id,),
                )
                job_row = cur.fetchone()
                if job_row is not None and job_row["status"] == "cancelled":
                    cur.execute(
                        """
                        UPDATE queue_items
                        SET stage = 'done',
                            worker_name = '',
                            input_id = %s,
                            result_status = %s,
                            result_http_code = 200,
                            result_source = %s,
                            selected_release_id = %s,
                            selected_agent_fallback_id = %s,
                            track_count = %s,
                            status_message = 'Completed before enrich stage (job cancelled)',
                            finished_at = NOW(),
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (
                            input_id,
                            str(context.get("result_status") or "matched"),
                            str(context.get("result_source") or ""),
                            selected_release_id,
                            selected_agent_fallback_id,
                            track_count,
                            item_id,
                        ),
                    )
                    _, just_finished = self._recompute_job_summary(cur, job_id)
                    conn.commit()
                    return False, just_finished

                cur.execute(
                    """
                    UPDATE queue_items
                    SET stage = 'queued_enrich',
                        worker_name = '',
                        input_id = %s,
                        selected_release_id = %s,
                        selected_agent_fallback_id = %s,
                        track_count = %s,
                        context_json = %s,
                        status_message = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (
                        input_id,
                        selected_release_id,
                        selected_agent_fallback_id,
                        track_count,
                        Jsonb(context),
                        status_message,
                        item_id,
                    ),
                )
            conn.commit()
        return True, False

    def complete_queue_item(
        self,
        item_id: int,
        job_id: int,
        input_id: int | None,
        result_status: str,
        result_http_code: int,
        result_source: str,
        selected_release_id: int | None = None,
        selected_agent_fallback_id: int | None = None,
        track_count: int = 0,
        status_message: str = "",
        error_detail: str = "",
    ) -> bool:
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    UPDATE queue_items
                    SET stage = 'done',
                        input_id = COALESCE(%s, input_id),
                        result_status = %s,
                        result_http_code = %s,
                        result_source = %s,
                        selected_release_id = %s,
                        selected_agent_fallback_id = %s,
                        track_count = %s,
                        status_message = %s,
                        error_detail = %s,
                        finished_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (
                        input_id,
                        result_status,
                        result_http_code,
                        result_source,
                        selected_release_id,
                        selected_agent_fallback_id,
                        track_count,
                        status_message or f"Finished ({result_status})",
                        error_detail,
                        item_id,
                    ),
                )
                _, just_finished = self._recompute_job_summary(cur, job_id)
            conn.commit()
        return just_finished

    def cancel_queue_job(self, job_id: int) -> QueueJobOut | None:
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT id, status FROM queue_jobs WHERE id = %s FOR UPDATE",
                    (job_id,),
                )
                job_row = cur.fetchone()
                if job_row is None:
                    conn.rollback()
                    return None

                if job_row["status"] != "completed":
                    cur.execute(
                        "UPDATE queue_jobs SET status = 'cancelled', updated_at = NOW() WHERE id = %s",
                        (job_id,),
                    )
                cur.execute(
                    """
                    UPDATE queue_items
                    SET stage = 'cancelled',
                        status_message = 'Cancelled',
                        finished_at = NOW(),
                        updated_at = NOW()
                    WHERE job_id = %s
                      AND stage IN ('queued_discogs', 'queued_agent', 'queued_enrich')
                    """,
                    (job_id,),
                )
                updated_job, _ = self._recompute_job_summary(cur, job_id)
            conn.commit()
        return _row_to_queue_job(updated_job)

    def recover_stale_queue_items(self) -> int:
        total_reset = 0
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE queue_items
                    SET stage = 'queued_discogs',
                        worker_name = '',
                        status_message = 'Recovered after restart',
                        updated_at = NOW()
                    WHERE stage = 'checking_discogs'
                    """
                )
                total_reset += cur.rowcount or 0
                cur.execute(
                    """
                    UPDATE queue_items
                    SET stage = 'queued_agent',
                        worker_name = '',
                        status_message = 'Recovered after restart',
                        updated_at = NOW()
                    WHERE stage = 'checking_agent'
                    """
                )
                total_reset += cur.rowcount or 0
                cur.execute(
                    """
                    UPDATE queue_items
                    SET stage = 'queued_enrich',
                        worker_name = '',
                        status_message = 'Recovered after restart',
                        updated_at = NOW()
                    WHERE stage = 'checking_enrich'
                    """
                )
                total_reset += cur.rowcount or 0
            conn.commit()
        return total_reset

    def set_queue_job_report_paths(
        self,
        job_id: int,
        output_json_path: str,
        output_markdown_path: str,
        mark_completed: bool = True,
    ) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                if mark_completed:
                    cur.execute(
                        """
                        UPDATE queue_jobs
                        SET output_json_path = %s,
                            output_markdown_path = %s,
                            status = CASE WHEN status <> 'cancelled' THEN 'completed' ELSE status END,
                            finished_at = COALESCE(finished_at, NOW()),
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (output_json_path, output_markdown_path, job_id),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE queue_jobs
                        SET output_json_path = %s,
                            output_markdown_path = %s,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (output_json_path, output_markdown_path, job_id),
                    )
            conn.commit()

    def get_queue_job(self, job_id: int) -> QueueJobDetailOut | None:
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT * FROM queue_jobs WHERE id = %s", (job_id,))
                job_row = cur.fetchone()
                if job_row is None:
                    return None
                cur.execute(
                    """
                    SELECT * FROM queue_items
                    WHERE job_id = %s
                    ORDER BY row_number ASC, id ASC
                    """,
                    (job_id,),
                )
                item_rows = cur.fetchall()
        return QueueJobDetailOut(
            job=_row_to_queue_job(job_row),
            items=[_row_to_queue_item(r) for r in item_rows],
        )

    def list_queue_jobs(
        self, limit: int = 20, status: str | None = None
    ) -> list[QueueJobOut]:
        where_clause = "WHERE status = %s" if status else ""
        params: tuple[Any, ...] = (status, limit) if status else (limit,)
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT * FROM queue_jobs
                    {where_clause}
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    params,
                )
                rows = cur.fetchall()
        return [_row_to_queue_job(r) for r in rows]

    def get_queue_status_snapshot(
        self, pending_limit: int = 50, jobs_limit: int = 20
    ) -> tuple[QueueStageCountsOut, list[QueueItemOut], list[QueueItemOut], list[QueueJobOut]]:
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE stage = 'queued_discogs') AS queued_discogs,
                        COUNT(*) FILTER (WHERE stage = 'checking_discogs') AS checking_discogs,
                        COUNT(*) FILTER (WHERE stage = 'queued_agent') AS queued_agent,
                        COUNT(*) FILTER (WHERE stage = 'checking_agent') AS checking_agent,
                        COUNT(*) FILTER (WHERE stage = 'queued_enrich') AS queued_enrich,
                        COUNT(*) FILTER (WHERE stage = 'checking_enrich') AS checking_enrich
                    FROM queue_items
                    """
                )
                cnt = cur.fetchone()

                cur.execute(
                    """
                    SELECT * FROM queue_items
                    WHERE stage IN ('checking_discogs', 'checking_agent', 'checking_enrich')
                    ORDER BY started_at ASC NULLS LAST, id ASC
                    """
                )
                active_rows = cur.fetchall()

                cur.execute(
                    """
                    SELECT * FROM queue_items
                    WHERE stage IN ('queued_discogs', 'queued_agent', 'queued_enrich')
                    ORDER BY id ASC
                    LIMIT %s
                    """,
                    (pending_limit,),
                )
                pending_rows = cur.fetchall()

                cur.execute(
                    """
                    SELECT * FROM queue_jobs
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (jobs_limit,),
                )
                job_rows = cur.fetchall()

        counts = QueueStageCountsOut(
            queued_discogs=int(cnt["queued_discogs"]),
            checking_discogs=int(cnt["checking_discogs"]),
            queued_agent=int(cnt["queued_agent"]),
            checking_agent=int(cnt["checking_agent"]),
            queued_enrich=int(cnt["queued_enrich"]),
            checking_enrich=int(cnt["checking_enrich"]),
        )
        return (
            counts,
            [_row_to_queue_item(r) for r in active_rows],
            [_row_to_queue_item(r) for r in pending_rows],
            [_row_to_queue_job(r) for r in job_rows],
        )
