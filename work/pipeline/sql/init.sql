CREATE TABLE IF NOT EXISTS input_records (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    barcode_raw TEXT NOT NULL DEFAULT '',
    album_title_raw TEXT NOT NULL DEFAULT '',
    album_artist_raw TEXT NOT NULL DEFAULT '',
    barcode_norm TEXT NOT NULL DEFAULT '',
    title_norm TEXT NOT NULL DEFAULT '',
    artist_norm TEXT NOT NULL DEFAULT '',
    format_hint JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_compilation BOOLEAN NOT NULL DEFAULT FALSE,
    input_hash TEXT NOT NULL UNIQUE,
    source_tag TEXT NOT NULL DEFAULT 'api',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_input_records_barcode_norm ON input_records (barcode_norm);
CREATE INDEX IF NOT EXISTS idx_input_records_artist_title ON input_records (artist_norm, title_norm);

CREATE TABLE IF NOT EXISTS releases (
    discogs_release_id BIGINT PRIMARY KEY,
    master_id BIGINT,
    url TEXT NOT NULL,
    resource_url TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL,
    title_norm TEXT NOT NULL,
    artist_display TEXT NOT NULL DEFAULT '',
    artist_norm TEXT NOT NULL DEFAULT '',
    year INTEGER,
    country TEXT NOT NULL DEFAULT '',
    formats_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    identifiers_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    labels_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    data_quality TEXT NOT NULL DEFAULT '',
    community_status TEXT NOT NULL DEFAULT '',
    community_have INTEGER NOT NULL DEFAULT 0,
    community_want INTEGER NOT NULL DEFAULT 0,
    notes TEXT NOT NULL DEFAULT '',
    track_count INTEGER NOT NULL DEFAULT 0,
    tracklist_hash TEXT NOT NULL DEFAULT '',
    unscoped_credits_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    raw_json JSONB NOT NULL,
    source_provider TEXT NOT NULL DEFAULT 'discogs',
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE releases
    ADD COLUMN IF NOT EXISTS source_provider TEXT NOT NULL DEFAULT 'discogs';

CREATE INDEX IF NOT EXISTS idx_releases_artist_title ON releases (artist_norm, title_norm);
CREATE INDEX IF NOT EXISTS idx_releases_tracklist_hash ON releases (tracklist_hash);

CREATE TABLE IF NOT EXISTS release_barcodes (
    discogs_release_id BIGINT NOT NULL REFERENCES releases(discogs_release_id) ON DELETE CASCADE,
    barcode_norm TEXT NOT NULL,
    barcode_raw TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (discogs_release_id, barcode_norm)
);

CREATE INDEX IF NOT EXISTS idx_release_barcodes_norm ON release_barcodes (barcode_norm);

CREATE TABLE IF NOT EXISTS tracks (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    discogs_release_id BIGINT NOT NULL REFERENCES releases(discogs_release_id) ON DELETE CASCADE,
    track_order INTEGER NOT NULL,
    disc_number INTEGER NOT NULL DEFAULT 1,
    disc_label TEXT NOT NULL DEFAULT '',
    position TEXT NOT NULL DEFAULT '',
    entry_type TEXT NOT NULL DEFAULT 'track',
    parent_position TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL,
    duration_raw TEXT NOT NULL DEFAULT '',
    duration_sec INTEGER,
    performing_artists_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    performing_artist_display TEXT NOT NULL DEFAULT '',
    has_valid_duration BOOLEAN NOT NULL DEFAULT FALSE,
    has_resolved_performer BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (discogs_release_id, track_order)
);

CREATE TABLE IF NOT EXISTS track_credits (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    track_id BIGINT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    discogs_artist_id BIGINT,
    artist_name TEXT NOT NULL,
    anv TEXT NOT NULL DEFAULT '',
    role_raw TEXT NOT NULL,
    role_category TEXT NOT NULL,
    credit_source TEXT NOT NULL,
    UNIQUE (track_id, artist_name, role_raw, credit_source)
);

CREATE TABLE IF NOT EXISTS input_matches (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    input_id BIGINT NOT NULL REFERENCES input_records(id) ON DELETE CASCADE,
    discogs_release_id BIGINT NOT NULL REFERENCES releases(discogs_release_id) ON DELETE CASCADE,
    is_selected BOOLEAN NOT NULL DEFAULT FALSE,
    match_rank INTEGER NOT NULL,
    match_method TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    evaluation_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (input_id, discogs_release_id)
);

CREATE TABLE IF NOT EXISTS agent_fallbacks (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    input_id BIGINT NOT NULL REFERENCES input_records(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL DEFAULT '',
    sources_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    track_count INTEGER NOT NULL DEFAULT 0,
    tracklist_hash TEXT NOT NULL DEFAULT '',
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    raw_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE agent_fallbacks
    ADD COLUMN IF NOT EXISTS input_tokens INTEGER NOT NULL DEFAULT 0;
ALTER TABLE agent_fallbacks
    ADD COLUMN IF NOT EXISTS output_tokens INTEGER NOT NULL DEFAULT 0;
ALTER TABLE agent_fallbacks
    ADD COLUMN IF NOT EXISTS cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0.0;

CREATE INDEX IF NOT EXISTS idx_agent_fallbacks_input_created ON agent_fallbacks (input_id, created_at DESC);

CREATE TABLE IF NOT EXISTS agent_fallback_tracks (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    agent_fallback_id BIGINT NOT NULL REFERENCES agent_fallbacks(id) ON DELETE CASCADE,
    track_order INTEGER NOT NULL,
    disc_number INTEGER NOT NULL DEFAULT 1,
    track_number INTEGER NOT NULL,
    position TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL,
    duration_raw TEXT NOT NULL DEFAULT '',
    duration_sec INTEGER,
    UNIQUE (agent_fallback_id, track_order)
);

CREATE TABLE IF NOT EXISTS lookup_runs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    input_id BIGINT NOT NULL REFERENCES input_records(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    http_code INTEGER NOT NULL,
    source TEXT NOT NULL,
    selected_release_id BIGINT REFERENCES releases(discogs_release_id) ON DELETE SET NULL,
    selected_agent_fallback_id BIGINT REFERENCES agent_fallbacks(id) ON DELETE SET NULL,
    queries_attempted_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    candidates_checked INTEGER NOT NULL DEFAULT 0,
    evaluation_summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    review_reasons_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    error_detail TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_lookup_runs_input_created ON lookup_runs (input_id, created_at DESC);

CREATE TABLE IF NOT EXISTS api_cache (
    cache_key TEXT PRIMARY KEY,
    endpoint TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    payload_json JSONB NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS record_reviews (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    input_id BIGINT NOT NULL REFERENCES input_records(id) ON DELETE CASCADE,
    decision TEXT NOT NULL,
    reviewer_type TEXT NOT NULL DEFAULT 'human',
    reviewer_name TEXT NOT NULL DEFAULT '',
    selected_release_id BIGINT REFERENCES releases(discogs_release_id) ON DELETE SET NULL,
    selected_agent_fallback_id BIGINT REFERENCES agent_fallbacks(id) ON DELETE SET NULL,
    note TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_record_reviews_input_created ON record_reviews (input_id, created_at DESC);

ALTER TABLE lookup_runs
    ADD COLUMN IF NOT EXISTS selected_agent_fallback_id BIGINT REFERENCES agent_fallbacks(id) ON DELETE SET NULL;

ALTER TABLE record_reviews
    ADD COLUMN IF NOT EXISTS selected_agent_fallback_id BIGINT REFERENCES agent_fallbacks(id) ON DELETE SET NULL;

CREATE TABLE IF NOT EXISTS queue_jobs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_type TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'queued',
    csv_path TEXT NOT NULL DEFAULT '',
    output_path TEXT NOT NULL DEFAULT '',
    force_refresh BOOLEAN NOT NULL DEFAULT FALSE,
    total_items INTEGER NOT NULL DEFAULT 0,
    completed_items INTEGER NOT NULL DEFAULT 0,
    matched_count INTEGER NOT NULL DEFAULT 0,
    needs_review_count INTEGER NOT NULL DEFAULT 0,
    agent_recovered_count INTEGER NOT NULL DEFAULT 0,
    not_found_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    cancelled_count INTEGER NOT NULL DEFAULT 0,
    output_json_path TEXT,
    output_markdown_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS queue_items (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id BIGINT NOT NULL REFERENCES queue_jobs(id) ON DELETE CASCADE,
    row_number INTEGER NOT NULL DEFAULT 1,
    input_id BIGINT REFERENCES input_records(id) ON DELETE SET NULL,
    barcode TEXT NOT NULL DEFAULT '',
    album_title TEXT NOT NULL DEFAULT '',
    album_artist TEXT NOT NULL DEFAULT '',
    force_refresh BOOLEAN NOT NULL DEFAULT FALSE,
    mode TEXT NOT NULL DEFAULT 'full',
    stage TEXT NOT NULL DEFAULT 'queued_discogs',
    worker_name TEXT NOT NULL DEFAULT '',
    attempt INTEGER NOT NULL DEFAULT 0,
    status_message TEXT NOT NULL DEFAULT '',
    context_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    result_status TEXT,
    result_http_code INTEGER,
    result_source TEXT,
    selected_release_id BIGINT REFERENCES releases(discogs_release_id) ON DELETE SET NULL,
    selected_agent_fallback_id BIGINT REFERENCES agent_fallbacks(id) ON DELETE SET NULL,
    track_count INTEGER NOT NULL DEFAULT 0,
    error_detail TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_queue_items_stage_id ON queue_items (stage, id ASC);
CREATE INDEX IF NOT EXISTS idx_queue_items_job_row ON queue_items (job_id, row_number ASC);
CREATE INDEX IF NOT EXISTS idx_queue_items_input_stage ON queue_items (input_id, stage);

-- Track enrichment & provenance migrations (idempotent for existing pgdata volumes)
ALTER TABLE releases
    ADD COLUMN IF NOT EXISTS enriched_at TIMESTAMPTZ;

ALTER TABLE tracks
    ADD COLUMN IF NOT EXISTS isrc TEXT NOT NULL DEFAULT '';
ALTER TABLE tracks
    ADD COLUMN IF NOT EXISTS isrc_source TEXT NOT NULL DEFAULT '';
ALTER TABLE tracks
    ADD COLUMN IF NOT EXISTS isrc_source_url TEXT NOT NULL DEFAULT '';
ALTER TABLE tracks
    ADD COLUMN IF NOT EXISTS duration_source TEXT NOT NULL DEFAULT '';
ALTER TABLE tracks
    ADD COLUMN IF NOT EXISTS duration_source_url TEXT NOT NULL DEFAULT '';
ALTER TABLE tracks
    ADD COLUMN IF NOT EXISTS performer_source TEXT NOT NULL DEFAULT '';
ALTER TABLE tracks
    ADD COLUMN IF NOT EXISTS performer_source_url TEXT NOT NULL DEFAULT '';
ALTER TABLE tracks
    ADD COLUMN IF NOT EXISTS enrichment_notes_json JSONB NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE track_credits
    ADD COLUMN IF NOT EXISTS source_url TEXT NOT NULL DEFAULT '';

ALTER TABLE agent_fallbacks
    ADD COLUMN IF NOT EXISTS enriched_at TIMESTAMPTZ;

ALTER TABLE agent_fallback_tracks
    ADD COLUMN IF NOT EXISTS performing_artists_json JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE agent_fallback_tracks
    ADD COLUMN IF NOT EXISTS performing_artist_display TEXT NOT NULL DEFAULT '';
ALTER TABLE agent_fallback_tracks
    ADD COLUMN IF NOT EXISTS has_resolved_performer BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE agent_fallback_tracks
    ADD COLUMN IF NOT EXISTS isrc TEXT NOT NULL DEFAULT '';
ALTER TABLE agent_fallback_tracks
    ADD COLUMN IF NOT EXISTS isrc_source TEXT NOT NULL DEFAULT '';
ALTER TABLE agent_fallback_tracks
    ADD COLUMN IF NOT EXISTS isrc_source_url TEXT NOT NULL DEFAULT '';
ALTER TABLE agent_fallback_tracks
    ADD COLUMN IF NOT EXISTS duration_source TEXT NOT NULL DEFAULT '';
ALTER TABLE agent_fallback_tracks
    ADD COLUMN IF NOT EXISTS duration_source_url TEXT NOT NULL DEFAULT '';
ALTER TABLE agent_fallback_tracks
    ADD COLUMN IF NOT EXISTS performer_source TEXT NOT NULL DEFAULT '';
ALTER TABLE agent_fallback_tracks
    ADD COLUMN IF NOT EXISTS performer_source_url TEXT NOT NULL DEFAULT '';
ALTER TABLE agent_fallback_tracks
    ADD COLUMN IF NOT EXISTS credits_json JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE agent_fallback_tracks
    ADD COLUMN IF NOT EXISTS enrichment_notes_json JSONB NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE releases
    ADD COLUMN IF NOT EXISTS enrichment_input_tokens INTEGER NOT NULL DEFAULT 0;
ALTER TABLE releases
    ADD COLUMN IF NOT EXISTS enrichment_output_tokens INTEGER NOT NULL DEFAULT 0;
ALTER TABLE releases
    ADD COLUMN IF NOT EXISTS enrichment_cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0.0;

