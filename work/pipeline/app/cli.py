import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from psycopg_pool import ConnectionPool

from app.agent_client import (
    AgentFallbackProvider,
    AzureOpenAIAgentClient,
    GeminiAgentClient,
    build_agent_client,
)
from app.config import Settings, get_settings
from app.discogs_client import CatalogProvider, DiscogsClient
from app.enricher import TrackEnricher, build_default_track_enricher
from app.fallback_providers import FallbackProvider, build_default_fallback_providers
from app.models import BatchRequest, LookupNotFoundResponse, LookupRequest
from app.repository import PipelineRepository
from app.service import LookupService, UpstreamServiceError


def _parse_rows(rows_arg: str | None) -> list[int] | None:
    if not rows_arg:
        return None
    return [int(part.strip()) for part in rows_arg.split(",") if part.strip()]


def main(
    argv: list[str] | None = None,
    provider: CatalogProvider | None = None,
    agent_provider: AgentFallbackProvider | None = None,
    settings: Settings | None = None,
    fallback_providers: list[FallbackProvider] | None = None,
    enricher: TrackEnricher | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Discogs tracklist discovery CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    lookup_cmd = sub.add_parser("lookup", help="Look up a single release")
    lookup_cmd.add_argument("--barcode", default="", help="EAN/UPC barcode")
    lookup_cmd.add_argument("--title", default="", help="Album title")
    lookup_cmd.add_argument("--artist", default="", help="Album artist")
    lookup_cmd.add_argument("--refresh", action="store_true", help="Bypass fresh DB cache")

    batch_cmd = sub.add_parser("batch", help="Process rows from a CSV file")
    batch_cmd.add_argument(
        "--csv", default="/app/data/development.csv", help="Path to CSV input"
    )
    batch_cmd.add_argument(
        "--rows",
        default="",
        help="Comma-separated 1-indexed data row numbers (e.g. 1,3,7,13,16)",
    )
    batch_cmd.add_argument(
        "--output",
        default="/app/outputs/batch_report.json",
        help="Output JSON report path (also writes companion .md)",
    )
    batch_cmd.add_argument("--refresh", action="store_true", help="Bypass fresh DB cache")

    enq_batch = sub.add_parser(
        "enqueue-batch",
        help="Enqueue a CSV batch into PostgreSQL queue_jobs for background API workers",
    )
    enq_batch.add_argument(
        "--csv", default="/app/data/development.csv", help="Path to CSV input"
    )
    enq_batch.add_argument(
        "--rows",
        default="",
        help="Comma-separated 1-indexed data row numbers (e.g. 1,3,7,13,16)",
    )
    enq_batch.add_argument(
        "--output",
        default="/app/outputs/batch_report.json",
        help="Output JSON report path (also writes companion .md)",
    )
    enq_batch.add_argument("--refresh", action="store_true", help="Bypass fresh DB cache")

    sub.add_parser(
        "retry-not-found",
        help="Enqueue all eligible not_found records for Gemini agent retry",
    )

    enrich_cmd = sub.add_parser(
        "enrich",
        help="Synchronously enrich missing track metadata for matched records",
    )
    enrich_cmd.add_argument("--input-id", type=int, default=None, help="Specific input_id to enrich")
    enrich_cmd.add_argument(
        "--all-missing",
        action="store_true",
        help="Enrich all matched/needs_review records with missing track data",
    )
    enrich_cmd.add_argument("--limit", type=int, default=100, help="Max records for --all-missing")
    enrich_cmd.add_argument("--use-agent", action="store_true", help="Allow Tier 3 Gemini search")
    enrich_cmd.add_argument("--refresh", action="store_true", help="Bypass api_cache & enriched_at")

    enq_enrich = sub.add_parser(
        "enqueue-enrich",
        help="Enqueue async track enrichment job(s) into PostgreSQL queue_jobs",
    )
    enq_enrich.add_argument("--input-id", type=int, default=None, help="Specific input_id to enrich")
    enq_enrich.add_argument(
        "--all-missing",
        action="store_true",
        help="Enqueue all matched/needs_review records with missing track data",
    )
    enq_enrich.add_argument("--limit", type=int, default=100, help="Max records for --all-missing")
    enq_enrich.add_argument("--use-agent", action="store_true", help="Allow Tier 3 Gemini search")
    enq_enrich.add_argument("--refresh", action="store_true", help="Bypass api_cache & enriched_at")

    args = parser.parse_args(argv)
    cfg = settings or get_settings()
    pool = ConnectionPool(conninfo=cfg.database_url, min_size=1, max_size=5, open=True)
    repo = PipelineRepository(pool)
    repo.ensure_schema()
    own_client: DiscogsClient | None = None
    active_provider = provider
    if active_provider is None:
        own_client = DiscogsClient(cfg, cache_store=repo)
        active_provider = own_client

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

    try:
        if args.command == "lookup":
            req = LookupRequest(
                barcode=args.barcode,
                album_title=args.title,
                album_artist=args.artist,
                force_refresh=args.refresh,
            )
            try:
                res = service.lookup(req, source_tag="cli")
            except UpstreamServiceError as exc:
                print(
                    json.dumps(
                        {
                            "status": "upstream_error",
                            "http_code": exc.status_code,
                            "detail": exc.detail,
                        },
                        indent=2,
                    )
                )
                return 2
            print(json.dumps(res.model_dump(), indent=2, ensure_ascii=False))
            return 1 if isinstance(res, LookupNotFoundResponse) else 0

        if args.command == "batch":
            req = BatchRequest(
                csv_path=args.csv,
                rows=_parse_rows(args.rows),
                force_refresh=args.refresh,
                output_path=args.output,
            )
            summary = service.run_batch(req)
            print(json.dumps(summary.model_dump(), indent=2, ensure_ascii=False))
            return 0

        if args.command == "enqueue-batch":
            csv_file = Path(args.csv)
            if not csv_file.exists():
                print(
                    json.dumps({"error": f"CSV file not found: {csv_file}"}),
                    file=sys.stderr,
                )
                return 2
            with csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
                all_rows = list(csv.DictReader(handle))
            selected_rows = _parse_rows(args.rows)
            selected_indices = (
                set(selected_rows) if selected_rows else set(range(1, len(all_rows) + 1))
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
                        "force_refresh": args.refresh,
                        "mode": "full",
                        "stage": "queued_discogs",
                        "status_message": "Queued for Discogs",
                    }
                )
            job = repo.create_queue_job(
                job_type="batch_csv",
                label=f"Batch: {csv_file.name}",
                items=items,
                csv_path=args.csv,
                output_path=args.output,
                force_refresh=args.refresh,
            )
            print(json.dumps(job.model_dump(), indent=2, ensure_ascii=False))
            return 0

        if args.command == "retry-not-found":
            if not service.has_agent_fallback():
                print(
                    json.dumps(
                        {"error": "Gemini agent fallback is not configured or disabled."}
                    ),
                    file=sys.stderr,
                )
                return 2
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
            print(json.dumps(job.model_dump(), indent=2, ensure_ascii=False))
            return 0

        if args.command == "enrich":
            if args.input_id is None and not args.all_missing:
                print(
                    json.dumps({"error": "Specify --input-id <ID> or --all-missing."}),
                    file=sys.stderr,
                )
                return 2
            if args.use_agent and not service.has_agent_fallback():
                print(
                    json.dumps({"error": "Gemini agent fallback is not configured or disabled."}),
                    file=sys.stderr,
                )
                return 2
            if args.input_id is not None:
                try:
                    res = service.enrich_record(
                        args.input_id,
                        use_agent=args.use_agent,
                        force_refresh=args.refresh,
                    )
                except ValueError as exc:
                    print(json.dumps({"error": str(exc)}), file=sys.stderr)
                    return 2
                if res is None:
                    print(
                        json.dumps({"error": f"Input record {args.input_id} not found."}),
                        file=sys.stderr,
                    )
                    return 1
                print(json.dumps(res.model_dump(), indent=2, ensure_ascii=False))
                return 0
            batch_res = service.enrich_missing_records(
                limit=args.limit,
                use_agent=args.use_agent,
                force_refresh=args.refresh,
            )
            print(json.dumps(batch_res.model_dump(), indent=2, ensure_ascii=False))
            return 0

        if args.command == "enqueue-enrich":
            if args.input_id is None and not args.all_missing:
                print(
                    json.dumps({"error": "Specify --input-id <ID> or --all-missing."}),
                    file=sys.stderr,
                )
                return 2
            if args.use_agent and not service.has_agent_fallback():
                print(
                    json.dumps({"error": "Gemini agent fallback is not configured or disabled."}),
                    file=sys.stderr,
                )
                return 2
            if args.input_id is not None:
                rec = repo.get_input_record(args.input_id)
                if rec is None:
                    print(
                        json.dumps({"error": f"Input record {args.input_id} not found."}),
                        file=sys.stderr,
                    )
                    return 1
                latest = repo.find_latest_run(args.input_id)
                if latest is None or latest["status"] not in ("matched", "needs_review"):
                    print(
                        json.dumps(
                            {
                                "error": f"Input record {args.input_id} has no matched/needs_review result."
                            }
                        ),
                        file=sys.stderr,
                    )
                    return 2
                items = [
                    {
                        "row_number": 1,
                        "input_id": args.input_id,
                        "barcode": rec["barcode_raw"],
                        "album_title": rec["album_title_raw"],
                        "album_artist": rec["album_artist_raw"],
                        "force_refresh": args.refresh,
                        "mode": "enrich",
                        "stage": "queued_enrich",
                        "status_message": "Queued for track enrichment",
                        "context_json": {
                            "use_agent": args.use_agent,
                            "force_refresh": args.refresh,
                            "result_status": latest["status"],
                            "result_source": latest["source"],
                        },
                    }
                ]
                label = (
                    f"Enrich #{args.input_id}: {rec['album_artist_raw']} - {rec['album_title_raw']}"
                ).strip()
            else:
                records = repo.list_matched_inputs_with_missing_track_data(
                    limit=args.limit, force_refresh=args.refresh
                )
                items = [
                    {
                        "row_number": idx,
                        "input_id": int(rec["id"]),
                        "barcode": rec["barcode_raw"],
                        "album_title": rec["album_title_raw"],
                        "album_artist": rec["album_artist_raw"],
                        "force_refresh": args.refresh,
                        "mode": "enrich",
                        "stage": "queued_enrich",
                        "status_message": "Queued for track enrichment",
                        "context_json": {
                            "use_agent": args.use_agent,
                            "force_refresh": args.refresh,
                            "result_status": rec["latest_status"],
                            "result_source": rec["latest_source"],
                        },
                    }
                    for idx, rec in enumerate(records, start=1)
                ]
                label = f"Enrich missing ({len(items)} records)"
            job = repo.create_queue_job(
                job_type="track_enrich",
                label=label,
                items=items,
                force_refresh=args.refresh,
            )
            print(json.dumps(job.model_dump(), indent=2, ensure_ascii=False))
            return 0
    finally:
        if own_enricher is not None:
            own_enricher.close()
        for fb in own_fallbacks:
            fb.close()
        if own_agent_client is not None:
            own_agent_client.close()
        if own_client is not None:
            own_client.close()
        pool.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
