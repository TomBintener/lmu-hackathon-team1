import json
from html import escape
from typing import Any

from app.models import QueueStatusOut, RecordDetailOut, RecordSummaryOut, TrackOut

_BASE_CSS = """
:root {
  --bg: #f8fafc;
  --card: #ffffff;
  --border: #e2e8f0;
  --text: #0f172a;
  --muted: #64748b;
  --accent: #2563eb;
  --ok-bg: #dcfce7;
  --ok-fg: #166534;
  --warn-bg: #fef3c7;
  --warn-fg: #92400e;
  --err-bg: #fee2e2;
  --err-fg: #991b1b;
  --neutral-bg: #f1f5f9;
  --neutral-fg: #334155;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 24px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.45;
}
.container { max-width: 1280px; margin: 0 auto; }
header {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
}
h1 { font-size: 1.35rem; margin: 0; }
h2 { font-size: 1.05rem; margin: 24px 0 10px; }
.nav { display: flex; gap: 12px; font-size: 0.9rem; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.stats {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 16px;
}
.stat {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 14px;
  min-width: 130px;
}
.stat-label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
.stat-val { font-size: 1.25rem; font-weight: 600; margin-top: 2px; }
.toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  margin-bottom: 12px;
}
input[type="search"], input[type="text"], select {
  padding: 7px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--card);
  font-size: 0.9rem;
}
input[type="search"] { min-width: 260px; flex: 1; max-width: 420px; }
.btn {
  padding: 7px 12px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--card);
  color: var(--text);
  font-size: 0.85rem;
  font-weight: 500;
  cursor: pointer;
}
.btn:hover:not(:disabled) { background: #f1f5f9; }
.btn:disabled { opacity: 0.45; cursor: not-allowed; }
.btn-sm { padding: 4px 9px; font-size: 0.78rem; }
.btn-confirm { background: var(--ok-bg); color: var(--ok-fg); border-color: #bbf7d0; }
.btn-confirm:hover:not(:disabled) { background: #bbf7d0; }
.btn-deny { background: var(--err-bg); color: var(--err-fg); border-color: #fecaca; }
.btn-deny:hover:not(:disabled) { background: #fecaca; }
.btn-review { background: var(--warn-bg); color: var(--warn-fg); border-color: #fde68a; }
.btn-review:hover:not(:disabled) { background: #fde68a; }
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow-x: auto;
}
.review-panel {
  padding: 14px;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
}
.review-error {
  color: var(--err-fg);
  font-size: 0.85rem;
  margin-top: 6px;
  display: none;
}
table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
th, td {
  text-align: left;
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
th {
  background: #f8fafc;
  color: var(--muted);
  font-weight: 600;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  white-space: nowrap;
}
tr:last-child td { border-bottom: none; }
tbody tr:hover { background: #f8fafc; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.83rem; }
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 0.76rem;
  font-weight: 600;
  white-space: nowrap;
}
.badge-matched, .badge-done, .badge-completed { background: var(--ok-bg); color: var(--ok-fg); }
.badge-needs_review, .badge-queued_discogs, .badge-queued_agent, .badge-queued_enrich, .badge-queued { background: var(--warn-bg); color: var(--warn-fg); }
.badge-checking_discogs, .badge-checking_agent, .badge-checking_enrich, .badge-running { background: #dbeafe; color: #1e40af; }
.badge-not_found, .badge-rejected, .badge-cancelled { background: var(--err-bg); color: var(--err-fg); }
.badge-other { background: var(--neutral-bg); color: var(--neutral-fg); }
.src-tag { font-size: 0.72rem; color: var(--muted); margin-left: 4px; white-space: nowrap; }
.note-sub { font-size: 0.75rem; color: var(--warn-fg); margin-top: 2px; }
.queue-banner {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 14px;
  margin-bottom: 14px;
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  font-size: 0.88rem;
}
.queue-banner.active {
  background: #eff6ff;
  border-color: #bfdbfe;
}
.progress-wrap {
  width: 120px;
  height: 8px;
  background: #e2e8f0;
  border-radius: 999px;
  overflow: hidden;
  display: inline-block;
  vertical-align: middle;
  margin-right: 6px;
}
.progress-bar {
  height: 100%;
  background: var(--accent);
}
.form-inline {
  padding: 14px;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
}
.muted-note { color: var(--muted); font-size: 0.84rem; }
.empty { padding: 36px 20px; text-align: center; color: var(--muted); }
.meta-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
  padding: 14px;
}
.meta-item .label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; }
.meta-item .val { font-size: 0.92rem; margin-top: 2px; word-break: break-word; }
.actions-cell { display: flex; gap: 8px; align-items: center; white-space: nowrap; }
"""


def _status_badge(status: str | None) -> str:
    s = status or "unknown"
    cls = s if s in {"matched", "needs_review", "not_found", "rejected"} else "other"
    return f'<span class="badge badge-{cls}">{escape(s)}</span>'


def _is_safe_http_url(url: str | None) -> bool:
    if not url:
        return False
    lowered = url.strip().lower()
    return lowered.startswith("http://") or lowered.startswith("https://")


def _format_release_label(
    release_id: int, source: str | None = None, url: str | None = None
) -> str:
    if release_id > 0:
        return f"#{release_id}"
    src = (source or "").lower()
    u = (url or "").lower()
    if "jpc" in src or "jpc.de" in u:
        return "JPC"
    if "itunes" in src or "apple.com" in u:
        return "iTunes"
    return "External"


def _provenance_tag(source: str, url: str = "", primary_provider: str = "") -> str:
    src = (source or "").strip()
    if not src:
        return ""
    if primary_provider and src == primary_provider:
        return ""
    if _is_safe_http_url(url):
        return f'<a class="src-tag" href="{escape(url)}" target="_blank" rel="noopener">[{escape(src)}]</a>'
    return f'<span class="src-tag">[{escape(src)}]</span>'


def _format_duration(track: TrackOut, primary_provider: str = "") -> str:
    if track.duration_raw:
        base = escape(track.duration_raw)
    elif track.duration_sec is not None:
        mins, secs = divmod(track.duration_sec, 60)
        base = f"{mins}:{secs:02d}"
    else:
        return "—"
    return base + _provenance_tag(track.duration_source, track.duration_source_url, primary_provider)


def _format_credits(credits: list[Any], primary_provider: str = "") -> str:
    if not credits:
        return "—"
    parts = []
    for c in credits:
        if isinstance(c, dict):
            name = escape(str(c.get("artist_name") or ""))
            role = escape(str(c.get("role_raw") or ""))
            c_src = str(c.get("credit_source") or "")
            c_url = str(c.get("source_url") or "")
        else:
            name = escape(str(getattr(c, "artist_name", "") or ""))
            role = escape(str(getattr(c, "role_raw", "") or ""))
            c_src = str(getattr(c, "credit_source", "") or "")
            c_url = str(getattr(c, "source_url", "") or "")
        text = f"{name} ({role})" if role else name
        if c_src and c_src not in {"track", "release_all", "release_range", primary_provider}:
            text += _provenance_tag(c_src, c_url, primary_provider)
        parts.append(text)
    return ", ".join(parts)


def render_results_table(records: list[RecordSummaryOut]) -> str:
    total = len(records)
    matched = sum(1 for r in records if r.latest_status == "matched")
    needs_review = sum(1 for r in records if r.latest_status == "needs_review")
    not_found = sum(1 for r in records if r.latest_status == "not_found")
    rejected = sum(1 for r in records if r.latest_status == "rejected")
    total_agent_cost = sum(r.agent_cost_usd for r in records)

    rows_html = []
    for r in records:
        status_str = r.latest_status or "unknown"
        if r.selected_release_id and r.selected_release_url:
            rel_label = _format_release_label(
                r.selected_release_id, r.latest_source, r.selected_release_url
            )
            release_cell = (
                f'<a href="{escape(r.selected_release_url)}" target="_blank" rel="noopener">{escape(rel_label)}</a>'
            )
        elif r.selected_release_id:
            release_cell = escape(
                _format_release_label(r.selected_release_id, r.latest_source)
            )
        elif r.selected_agent_fallback_id:
            if _is_safe_http_url(r.agent_source_url):
                release_cell = (
                    f'<a href="{escape(r.agent_source_url or "")}" target="_blank" rel="noopener">Agent #{r.selected_agent_fallback_id}</a>'
                )
            else:
                release_cell = f"Agent #{r.selected_agent_fallback_id}"
        else:
            release_cell = "—"
        checked_at = escape((r.latest_checked_at or "—")[:19].replace("T", " "))
        flag_btn = (
            f'<button class="btn btn-sm btn-review" onclick="triggerReview({r.input_id})">Trigger Review</button>'
            if status_str != "needs_review"
            else f'<a class="btn btn-sm btn-review" href="/results/{r.input_id}">Verify →</a>'
        )
        enqueue_btn = (
            f'<button class="btn btn-sm" onclick="enqueueAgent({r.input_id})">Enqueue Agent</button>'
            if status_str in {"not_found", "rejected"}
            else ""
        )
        cost_cell = (
            f"${r.agent_cost_usd:.6f}"
            if (r.agent_input_tokens > 0 or r.agent_output_tokens > 0 or r.agent_cost_usd > 0)
            else "—"
        )
        rows_html.append(
            f'<tr data-status="{escape(status_str)}">'
            f'<td class="mono">{r.input_id}</td>'
            f'<td class="mono">{escape(r.barcode_raw or "—")}</td>'
            f"<td>{escape(r.album_title_raw or '—')}</td>"
            f"<td>{escape(r.album_artist_raw or '—')}</td>"
            f"<td>{_status_badge(r.latest_status)}</td>"
            f'<td class="mono">{release_cell}</td>'
            f"<td>{escape(r.latest_source or '—')}</td>"
            f'<td class="mono">{cost_cell}</td>'
            f'<td class="mono">{checked_at}</td>'
            f'<td><div class="actions-cell"><a href="/results/{r.input_id}">Inspect</a>{flag_btn}{enqueue_btn}</div></td>'
            f"</tr>"
        )

    table_body = (
        "\n".join(rows_html)
        if rows_html
        else '<tr><td colspan="10" class="empty">No records in database yet. Run a lookup or batch first.</td></tr>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tracklist Discovery Results</title>
  <style>{_BASE_CSS}</style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Tracklist Discovery Results</h1>
      <div class="nav">
        <a href="/queue">Live Queue (/queue)</a>
        <a href="/records?status=needs_review" target="_blank">Review Queue JSON</a>
        <a href="/records" target="_blank">All JSON (/records)</a>
        <a href="/docs" target="_blank">API Docs (/docs)</a>
      </div>
    </header>

    <div id="queueBanner" class="queue-banner">
      <span id="queueBannerText">Checking queue activity...</span>
      <a href="/queue">Open Live Queue →</a>
    </div>

    <div class="stats">
      <div class="stat"><div class="stat-label">Total</div><div class="stat-val">{total}</div></div>
      <div class="stat"><div class="stat-label">Matched</div><div class="stat-val">{matched}</div></div>
      <div class="stat"><div class="stat-label">Needs Review</div><div class="stat-val">{needs_review}</div></div>
      <div class="stat"><div class="stat-label">Not Found</div><div class="stat-val">{not_found}</div></div>
      <div class="stat"><div class="stat-label">Rejected</div><div class="stat-val">{rejected}</div></div>
      <div class="stat"><div class="stat-label">Agent Cost</div><div class="stat-val mono">${total_agent_cost:.6f}</div></div>
    </div>

    <div class="toolbar">
      <input id="search" type="search" placeholder="Filter by barcode, title, artist..." oninput="filterRows()" />
      <select id="statusFilter" onchange="filterRows()">
        <option value="">All statuses</option>
        <option value="matched">matched</option>
        <option value="needs_review">needs_review</option>
        <option value="not_found">not_found</option>
        <option value="rejected">rejected</option>
      </select>
      <button class="btn" onclick="location.reload()">Refresh</button>
      <button class="btn btn-confirm" onclick="enqueueEnrichMissing()">Enrich Missing Data</button>
      <label style="font-size:0.84rem; display:inline-flex; align-items:center; gap:0.25rem;"><input id="enrichMissingUseAgent" type="checkbox" checked /> Allow Gemini</label>
      <button class="btn btn-deny" id="restartDemoBtn" onclick="restartDemo()">Restart 25 Songs</button>
      <label style="font-size:0.84rem; display:inline-flex; align-items:center; gap:0.25rem;" title="Also wipe cached HTTP API responses (slower live network calls)"><input id="restartClearCache" type="checkbox" /> Clear API Cache</label>
      <span id="toolbarMsg" class="muted-note"></span>
    </div>

    <div class="card">
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Barcode</th>
            <th>Album Title</th>
            <th>Album Artist</th>
            <th>Status</th>
            <th>Discogs Release</th>
            <th>Source</th>
            <th>Agent Cost</th>
            <th>Checked At (UTC)</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody id="resultsBody">
          {table_body}
        </tbody>
      </table>
    </div>
  </div>
  <script>
    function filterRows() {{
      const q = document.getElementById("search").value.toLowerCase();
      const st = document.getElementById("statusFilter").value;
      const rows = document.querySelectorAll("#resultsBody tr[data-status]");
      for (const row of rows) {{
        const text = row.textContent.toLowerCase();
        const rowStatus = row.getAttribute("data-status");
        const show = (!q || text.includes(q)) && (!st || rowStatus === st);
        row.style.display = show ? "" : "none";
      }}
    }}

    async function triggerReview(inputId) {{
      const note = prompt("Optional review note (leave blank if none):", "") ?? "";
      const res = await fetch(`/records/${{inputId}}/review`, {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{
          decision: "request_review",
          reviewer_type: "human",
          note: note
        }})
      }});
      if (res.ok) {{
        location.reload();
      }} else {{
        const err = await res.json().catch(() => ({{}}));
        alert(err.detail || "Failed to trigger review.");
      }}
    }}

    function escHtml(v) {{
      return String(v ?? "").replace(/[&<>"']/g, (ch) => ({{
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }}[ch]));
    }}

    async function enqueueAgent(inputId) {{
      const res = await fetch(`/jobs/agent-retry/${{inputId}}`, {{ method: "POST" }});
      if (res.ok) {{
        pollQueueBanner();
      }} else {{
        const err = await res.json().catch(() => ({{}}));
        alert(err.detail || "Failed to enqueue agent retry.");
      }}
    }}

    async function enqueueEnrichMissing() {{
      const msgEl = document.getElementById("toolbarMsg");
      const useAgent = Boolean(document.getElementById("enrichMissingUseAgent")?.checked);
      const res = await fetch("/jobs/enrich-missing", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ use_agent: useAgent, force_refresh: true }})
      }});
      const data = await res.json().catch(() => ({{}}));
      if (res.ok) {{
        msgEl.textContent = data.total_items
          ? `Enqueued enrichment job #${{data.id}} (${{data.total_items}} records)`
          : "No un-enriched records with missing fields.";
        pollQueueBanner();
      }} else {{
        msgEl.textContent = data.detail || "Failed to enqueue enrichment.";
      }}
    }}

    async function restartDemo() {{
      const clearCache = Boolean(document.getElementById("restartClearCache")?.checked);
      const cacheNote = clearCache ? " AND clear cached HTTP API responses" : " (keeping cached API responses for smooth live replay)";
      if (!confirm(`Reset all current records/jobs${{cacheNote}} and reload all 25 songs into the live queue?`)) return;
      const btn = document.getElementById("restartDemoBtn");
      const msgEl = document.getElementById("toolbarMsg");
      if (btn) btn.disabled = true;
      msgEl.textContent = "Resetting DB & queuing 25 songs...";
      try {{
        const res = await fetch("/jobs/restart-demo", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ clear_api_cache: clearCache }})
        }});
        const data = await res.json().catch(() => ({{}}));
        if (res.ok) {{
          window.location.href = `/queue?job=${{data.id}}`;
        }} else {{
          msgEl.textContent = data.detail || "Failed to restart 25 songs.";
          if (btn) btn.disabled = false;
        }}
      }} catch (_e) {{
        msgEl.textContent = "Network error while restarting.";
        if (btn) btn.disabled = false;
      }}
    }}

    async function pollQueueBanner() {{
      const banner = document.getElementById("queueBanner");
      const textEl = document.getElementById("queueBannerText");
      try {{
        const res = await fetch("/jobs/status");
        if (!res.ok) return;
        const data = await res.json();
        const c = data.counts || {{}};
        const activeCount = (c.checking_discogs || 0) + (c.checking_agent || 0) + (c.checking_enrich || 0);
        const queuedCount = (c.queued_discogs || 0) + (c.queued_agent || 0) + (c.queued_enrich || 0);
        if (activeCount > 0 || queuedCount > 0) {{
          banner.classList.add("active");
          const currentNames = (data.active_items || []).slice(0, 3).map(
            (i) => `${{escHtml(i.worker_name || i.stage)}}: ${{escHtml(i.album_artist)}} - ${{escHtml(i.album_title)}}`
          );
          const suffix = currentNames.length ? ` (${{currentNames.join(" | ")}})` : "";
          textEl.innerHTML = `<strong>Queue Active:</strong> ${{c.checking_discogs || 0}} Discogs, ${{c.checking_agent || 0}} Agent, ${{c.checking_enrich || 0}} Enrich, ${{queuedCount}} queued${{suffix}}`;
        }} else {{
          banner.classList.remove("active");
          textEl.textContent = `Queue idle (D:${{data.discogs_workers || 0}} A:${{data.agent_workers || 0}} E:${{data.enrich_workers || 0}})`;
        }}
      }} catch (_e) {{}}
    }}

    pollQueueBanner();
    setInterval(pollQueueBanner, 2000);
  </script>
</body>
</html>"""


def render_record_detail(detail: RecordDetailOut) -> str:
    rec = detail.record
    rel = detail.selected_release
    agent_fb = detail.agent_fallback
    has_selected = (rel is not None) or (agent_fb is not None)

    if rel is not None:
        formats_str = ", ".join(
            str(f.get("name", "")) for f in rel.formats if f.get("name")
        ) or "—"
        rel_label = _format_release_label(rel.discogs_release_id, rec.latest_source, rel.url)
        release_html = f"""
        <div class="card meta-grid">
          <div class="meta-item"><div class="label">Release</div><div class="val"><a href="{escape(rel.url)}" target="_blank" rel="noopener">{escape(rel_label)} — {escape(rel.title)}</a></div></div>
          <div class="meta-item"><div class="label">Artist</div><div class="val">{escape(rel.artist_display or '—')}</div></div>
          <div class="meta-item"><div class="label">Year / Country</div><div class="val">{escape(str(rel.year or '—'))} / {escape(rel.country or '—')}</div></div>
          <div class="meta-item"><div class="label">Formats</div><div class="val">{escape(formats_str)}</div></div>
          <div class="meta-item"><div class="label">Data Quality</div><div class="val">{escape(rel.data_quality or '—')}</div></div>
        </div>"""
    elif agent_fb is not None:
        sources_items = []
        for src in agent_fb.sources:
            if _is_safe_http_url(src):
                sources_items.append(
                    f'<a href="{escape(src)}" target="_blank" rel="noopener">{escape(src)}</a>'
                )
            else:
                sources_items.append(f"<span>{escape(src)}</span>")
        sources_html = "<br />".join(sources_items) if sources_items else "—"
        release_html = f"""
        <div class="card meta-grid">
          <div class="meta-item"><div class="label">Agent Fallback</div><div class="val mono">Agent #{agent_fb.agent_fallback_id}</div></div>
          <div class="meta-item"><div class="label">Model</div><div class="val mono">{escape(agent_fb.model_name or '—')}</div></div>
          <div class="meta-item"><div class="label">Track Count</div><div class="val">{agent_fb.track_count}</div></div>
          <div class="meta-item"><div class="label">Tokens (in / out)</div><div class="val mono">{agent_fb.input_tokens:,} / {agent_fb.output_tokens:,}</div></div>
          <div class="meta-item"><div class="label">Agent Cost (USD)</div><div class="val mono">${agent_fb.cost_usd:.6f}</div></div>
          <div class="meta-item"><div class="label">Tracklist Hash</div><div class="val mono">{escape(agent_fb.tracklist_hash or '—')}</div></div>
          <div class="meta-item"><div class="label">Sources</div><div class="val">{sources_html}</div></div>
        </div>"""
    else:
        release_html = '<div class="card empty">No release selected.</div>'

    primary_provider = ""
    if rel is not None:
        if rel.discogs_release_id > 0:
            primary_provider = "discogs"
        elif "jpc" in (rec.latest_source or "").lower() or "jpc.de" in rel.url.lower():
            primary_provider = "jpc"
        elif "itunes" in (rec.latest_source or "").lower() or "apple.com" in rel.url.lower():
            primary_provider = "itunes"
    elif agent_fb is not None:
        primary_provider = "gemini_agent"

    track_rows = []
    for t in detail.tracks:
        writers_list = list(t.credits.get("composers", [])) + list(t.credits.get("lyricists", []))
        writers = _format_credits(writers_list, primary_provider=primary_provider)
        producers = _format_credits(t.credits.get("producers", []), primary_provider=primary_provider)
        featured = _format_credits(t.credits.get("featured", []), primary_provider=primary_provider)
        perf_cell = (
            escape(t.performing_artist_display)
            + _provenance_tag(t.performer_source, t.performer_source_url, primary_provider)
            if t.performing_artist_display
            else "—"
        )
        isrc_cell = (
            f'<span class="mono">{escape(t.isrc)}</span>'
            + _provenance_tag(t.isrc_source, t.isrc_source_url, primary_provider)
            if t.isrc
            else "—"
        )
        notes_html = (
            "".join(f'<div class="note-sub">{escape(n)}</div>' for n in t.enrichment_notes)
            if t.enrichment_notes
            else ""
        )
        track_rows.append(
            f"<tr>"
            f'<td class="mono">{t.disc_number}</td>'
            f'<td class="mono">{escape(t.position or str(t.track_order))}</td>'
            f"<td>{escape(t.title)}{notes_html}</td>"
            f"<td>{perf_cell}</td>"
            f'<td class="mono">{_format_duration(t, primary_provider=primary_provider)}</td>'
            f"<td>{isrc_cell}</td>"
            f"<td>{writers}</td>"
            f"<td>{producers}</td>"
            f"<td>{featured}</td>"
            f"</tr>"
        )
    tracks_body = (
        "\n".join(track_rows)
        if track_rows
        else '<tr><td colspan="9" class="empty">No tracks extracted.</td></tr>'
    )

    match_rows = []
    for m in detail.matches:
        sel_mark = "✓" if m.get("is_selected") else ""
        rid = int(m["discogs_release_id"])
        m_label = _format_release_label(rid, str(m.get("match_method") or ""), str(m.get("url") or ""))
        match_rows.append(
            f"<tr>"
            f'<td class="mono">{escape(str(m.get("match_rank", "")))}</td>'
            f"<td>{sel_mark}</td>"
            f'<td class="mono"><a href="{escape(str(m.get("url", "")))}" target="_blank" rel="noopener">{escape(m_label)}</a></td>'
            f"<td>{escape(str(m.get('title', '')))}</td>"
            f"<td>{escape(str(m.get('artist_display', '')))}</td>"
            f'<td class="mono">{escape(str(m.get("match_method", "")))}</td>'
            f'<td class="mono">{escape(str(m.get("score", "")))}</td>'
            f'<td><button class="btn btn-sm btn-confirm" onclick="submitReview(\'confirm\', {rid})">Confirm This</button></td>'
            f"</tr>"
        )
    matches_body = (
        "\n".join(match_rows)
        if match_rows
        else '<tr><td colspan="8" class="empty">No candidate matches recorded.</td></tr>'
    )

    review_rows = []
    for rv in detail.reviews:
        if rv.selected_release_id:
            rel_str = _format_release_label(rv.selected_release_id, rec.latest_source)
        elif rv.selected_agent_fallback_id:
            rel_str = f"Agent #{rv.selected_agent_fallback_id}"
        else:
            rel_str = "—"
        rv_time = escape((rv.created_at or "—")[:19].replace("T", " "))
        review_rows.append(
            f"<tr>"
            f'<td class="mono">{rv_time}</td>'
            f"<td>{escape(rv.reviewer_type)}</td>"
            f"<td>{escape(rv.reviewer_name or '—')}</td>"
            f"<td>{escape(rv.decision)}</td>"
            f'<td class="mono">{rel_str}</td>'
            f"<td>{escape(rv.note or '—')}</td>"
            f"</tr>"
        )
    reviews_body = (
        "\n".join(review_rows)
        if review_rows
        else '<tr><td colspan="6" class="empty">No reviews recorded yet.</td></tr>'
    )

    confirm_disabled = "" if has_selected else "disabled"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Record #{rec.input_id} — {escape(rec.album_title_raw or rec.barcode_raw or 'Details')}</title>
  <style>{_BASE_CSS}</style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Record #{rec.input_id} {_status_badge(rec.latest_status)}</h1>
      <div class="nav">
        <a href="/results">← All Results</a>
        <a href="/queue">Live Queue</a>
        <a href="/records/{rec.input_id}" target="_blank">JSON</a>
        <a href="/docs" target="_blank">API Docs</a>
      </div>
    </header>

    <div class="card meta-grid">
      <div class="meta-item"><div class="label">Barcode (raw / norm)</div><div class="val mono">{escape(rec.barcode_raw or '—')} / {escape(rec.barcode_norm or '—')}</div></div>
      <div class="meta-item"><div class="label">Album Title</div><div class="val">{escape(rec.album_title_raw or '—')}</div></div>
      <div class="meta-item"><div class="label">Album Artist</div><div class="val">{escape(rec.album_artist_raw or '—')}</div></div>
      <div class="meta-item"><div class="label">Source / Checked At</div><div class="val">{escape(rec.latest_source or '—')} / <span class="mono">{escape((rec.latest_checked_at or '—')[:19].replace('T', ' '))}</span></div></div>
    </div>

    <h2>Verification &amp; Review</h2>
    <div class="card review-panel">
      <select id="reviewerType">
        <option value="human">human</option>
        <option value="agent">agent</option>
      </select>
      <input id="reviewerName" type="text" placeholder="Reviewer name (optional)" />
      <input id="reviewNote" type="text" placeholder="Review note / reason (optional)" style="flex:1; min-width:220px;" />
      <button class="btn btn-confirm" onclick="submitReview('confirm', null)" {confirm_disabled}>Confirm Match</button>
      <button class="btn btn-deny" onclick="submitReview('deny', null)">Deny / Reject</button>
      <button class="btn btn-review" onclick="submitReview('request_review', null)">Trigger Review</button>
      {"" if not has_selected else '<button class="btn" onclick="enqueueEnrich()">Enrich Missing Tracks</button><label style="font-size:0.84rem;"><input id="enrichUseAgent" type="checkbox" checked /> Allow Gemini</label>'}
      {"" if (rec.latest_status not in {"not_found", "rejected"}) else '<button class="btn" onclick="enqueueAgentFallback()">Enqueue Agent Fallback</button>'}
      <div id="reviewError" class="review-error"></div>
    </div>

    <h2>Selected Release</h2>
    {release_html}

    <h2>Tracklist ({len(detail.tracks)})</h2>
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Disc</th>
            <th>Pos</th>
            <th>Title</th>
            <th>Performing Artists</th>
            <th>Duration</th>
            <th>ISRC</th>
            <th>Writers</th>
            <th>Producers</th>
            <th>Featured</th>
          </tr>
        </thead>
        <tbody>{tracks_body}</tbody>
      </table>
    </div>

    <h2>Candidate Matches ({len(detail.matches)})</h2>
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Selected</th>
            <th>Release</th>
            <th>Title</th>
            <th>Artist</th>
            <th>Method</th>
            <th>Score</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>{matches_body}</tbody>
      </table>
    </div>

    <h2>Review History ({len(detail.reviews)})</h2>
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Time (UTC)</th>
            <th>Type</th>
            <th>Reviewer</th>
            <th>Decision</th>
            <th>Release</th>
            <th>Note</th>
          </tr>
        </thead>
        <tbody>{reviews_body}</tbody>
      </table>
    </div>
  </div>
  <script>
    async function submitReview(decision, releaseId) {{
      const errBox = document.getElementById("reviewError");
      errBox.style.display = "none";
      const payload = {{
        decision: decision,
        reviewer_type: document.getElementById("reviewerType").value,
        reviewer_name: document.getElementById("reviewerName").value,
        selected_release_id: releaseId,
        note: document.getElementById("reviewNote").value
      }};
      const res = await fetch("/records/{rec.input_id}/review", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(payload)
      }});
      if (res.ok) {{
        location.reload();
      }} else {{
        const data = await res.json().catch(() => ({{}}));
        errBox.textContent = data.detail || "Review request failed.";
        errBox.style.display = "block";
      }}
    }}

    async function enqueueAgentFallback() {{
      const errBox = document.getElementById("reviewError");
      errBox.style.display = "none";
      const res = await fetch("/jobs/agent-retry/{rec.input_id}", {{ method: "POST" }});
      if (res.ok) {{
        window.location.href = "/queue";
      }} else {{
        const data = await res.json().catch(() => ({{}}));
        errBox.textContent = data.detail || "Failed to enqueue agent fallback.";
        errBox.style.display = "block";
      }}
    }}

    async function enqueueEnrich() {{
      const errBox = document.getElementById("reviewError");
      errBox.style.display = "none";
      const useAgent = Boolean(document.getElementById("enrichUseAgent")?.checked);
      const res = await fetch("/jobs/enrich/{rec.input_id}", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ use_agent: useAgent, force_refresh: true }})
      }});
      if (res.ok) {{
        window.location.href = "/queue";
      }} else {{
        const data = await res.json().catch(() => ({{}}));
        errBox.textContent = data.detail || "Failed to enqueue track enrichment.";
        errBox.style.display = "block";
      }}
    }}
  </script>
</body>
</html>"""


_QUEUE_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Live Pipeline &amp; Agent Queue</title>
  <style>__BASE_CSS__</style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Live Pipeline &amp; Agent Queue</h1>
      <div class="nav">
        <a href="/results">&larr; Results (/results)</a>
        <a href="/jobs/status" target="_blank">Queue Status JSON (/jobs/status)</a>
        <a href="/docs" target="_blank">API Docs (/docs)</a>
      </div>
    </header>

    <div class="stats">
      <div class="stat"><div class="stat-label">Active Discogs</div><div class="stat-val" id="statActiveDiscogs">0</div></div>
      <div class="stat"><div class="stat-label">Queued Discogs</div><div class="stat-val" id="statQueuedDiscogs">0</div></div>
      <div class="stat"><div class="stat-label">Active Agent</div><div class="stat-val" id="statActiveAgent">0</div></div>
      <div class="stat"><div class="stat-label">Queued Agent</div><div class="stat-val" id="statQueuedAgent">0</div></div>
      <div class="stat"><div class="stat-label">Active Enrich</div><div class="stat-val" id="statActiveEnrich">0</div></div>
      <div class="stat"><div class="stat-label">Queued Enrich</div><div class="stat-val" id="statQueuedEnrich">0</div></div>
      <div class="stat"><div class="stat-label">Worker Pool</div><div class="stat-val" id="statWorkers" style="font-size:0.95rem;">—</div></div>
    </div>

    <div class="toolbar">
      <button class="btn" id="pollToggleBtn" onclick="togglePolling()">Pause Auto-Poll (1s)</button>
      <button class="btn" onclick="pollStatus()">Refresh Now</button>
      <button class="btn btn-review" id="retryAllBtn" onclick="retryNotFound()">Retry All not_found with Agent</button>
      <button class="btn btn-confirm" id="enrichMissingBtn" onclick="enqueueEnrichMissing()">Enrich Missing Tracks</button>
      <label style="font-size:0.84rem; display:inline-flex; align-items:center; gap:0.25rem;"><input id="enrichMissingUseAgent" type="checkbox" checked /> Allow Gemini</label>
      <button class="btn btn-deny" id="restartDemoBtn" onclick="restartDemo()">Restart 25 Songs</button>
      <label style="font-size:0.84rem; display:inline-flex; align-items:center; gap:0.25rem;" title="Also wipe cached HTTP API responses (slower live network calls)"><input id="restartClearCache" type="checkbox" /> Clear API Cache</label>
      <span class="muted-note" id="pollMeta">Auto-polling every 1s</span>
      <span class="muted-note" id="actionMsg"></span>
    </div>

    <div class="card" style="margin-bottom: 16px;">
      <form class="form-inline" onsubmit="enqueueSingle(event)">
        <strong>Enqueue Single Lookup:</strong>
        <input id="inBarcode" type="text" placeholder="Barcode (optional)" />
        <input id="inTitle" type="text" placeholder="Album title" />
        <input id="inArtist" type="text" placeholder="Album artist" />
        <select id="inMode">
          <option value="full">full (Discogs &rarr; Agent)</option>
          <option value="agent_only">agent_only (Gemini directly)</option>
        </select>
        <button class="btn btn-confirm" type="submit">Enqueue</button>
      </form>
    </div>

    <h2>Active Workers</h2>
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Worker</th>
            <th>Job / Row</th>
            <th>Album &amp; Artist</th>
            <th>Stage</th>
            <th>Status Message</th>
            <th>Attempt</th>
            <th>Elapsed</th>
          </tr>
        </thead>
        <tbody id="activeWorkersBody">
          <tr><td colspan="7" class="empty">No active workers right now.</td></tr>
        </tbody>
      </table>
    </div>

    <h2>Pending Queue</h2>
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Item ID</th>
            <th>Job / Row</th>
            <th>Stage</th>
            <th>Album &amp; Artist</th>
            <th>Mode</th>
            <th>Status Message</th>
          </tr>
        </thead>
        <tbody id="pendingQueueBody">
          <tr><td colspan="6" class="empty">Queue is empty.</td></tr>
        </tbody>
      </table>
    </div>

    <h2>Recent Jobs</h2>
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Job</th>
            <th>Type / Label</th>
            <th>Status</th>
            <th>Progress</th>
            <th>Breakdown</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody id="jobsBody">
          <tr><td colspan="6" class="empty">No jobs queued yet.</td></tr>
        </tbody>
      </table>
    </div>

    <div id="jobDetailPanel" style="display:none; margin-top: 20px;">
      <h2 id="jobDetailHeading">Job Details</h2>
      <div class="card">
        <table>
          <thead>
            <tr>
              <th>Row</th>
              <th>Album &amp; Artist</th>
              <th>Stage</th>
              <th>Worker</th>
              <th>Result</th>
              <th>Source</th>
              <th>Tracks</th>
              <th>Status Message</th>
            </tr>
          </thead>
          <tbody id="jobDetailBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const INITIAL_STATUS = __INITIAL_STATUS_JSON__;
    let paused = false;
    let selectedJobId = null;

    function escHtml(v) {
      return String(v ?? "").replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[ch]));
    }

    function stageBadge(stage) {
      const safe = escHtml(stage || "other");
      return `<span class="badge badge-${safe}">${safe}</span>`;
    }

    function renderStatus(data) {
      const c = data.counts || {};
      document.getElementById("statActiveDiscogs").textContent = c.checking_discogs ?? 0;
      document.getElementById("statQueuedDiscogs").textContent = c.queued_discogs ?? 0;
      document.getElementById("statActiveAgent").textContent = c.checking_agent ?? 0;
      document.getElementById("statQueuedAgent").textContent = c.queued_agent ?? 0;
      document.getElementById("statActiveEnrich").textContent = c.checking_enrich ?? 0;
      document.getElementById("statQueuedEnrich").textContent = c.queued_enrich ?? 0;
      const poolText = `${data.workers_running ? "running" : "stopped"} | D:${data.discogs_workers} A:${data.agent_workers} E:${data.enrich_workers ?? 0} | Agent:${data.agent_configured ? "on" : "off"} | Idle:${data.idle_auto_enrichment_enabled ? "on" : "off"}`;
      document.getElementById("statWorkers").textContent = poolText;
      document.getElementById("retryAllBtn").disabled = !data.agent_configured;

      const active = data.active_items || [];
      const activeBody = document.getElementById("activeWorkersBody");
      if (!active.length) {
        activeBody.innerHTML = '<tr><td colspan="7" class="empty">No active workers right now.</td></tr>';
      } else {
        activeBody.innerHTML = active.map((item) => `
          <tr>
            <td class="mono">${escHtml(item.worker_name || "—")}</td>
            <td class="mono">#${item.job_id} / row ${item.row_number}</td>
            <td><strong>${escHtml(item.album_title || item.barcode || "—")}</strong><br/><span class="muted-note">${escHtml(item.album_artist || "")}</span></td>
            <td>${stageBadge(item.stage)}</td>
            <td>${escHtml(item.status_message || "—")}</td>
            <td class="mono">${item.attempt || 0}</td>
            <td class="mono">${item.elapsed_sec != null ? item.elapsed_sec + "s" : "—"}</td>
          </tr>
        `).join("");
      }

      const pending = data.pending_items || [];
      const pendingBody = document.getElementById("pendingQueueBody");
      if (!pending.length) {
        pendingBody.innerHTML = '<tr><td colspan="6" class="empty">Queue is empty.</td></tr>';
      } else {
        pendingBody.innerHTML = pending.map((item) => `
          <tr>
            <td class="mono">#${item.id}</td>
            <td class="mono">#${item.job_id} / row ${item.row_number}</td>
            <td>${stageBadge(item.stage)}</td>
            <td><strong>${escHtml(item.album_title || item.barcode || "—")}</strong><br/><span class="muted-note">${escHtml(item.album_artist || "")}</span></td>
            <td class="mono">${escHtml(item.mode)}</td>
            <td>${escHtml(item.status_message || "—")}</td>
          </tr>
        `).join("");
      }

      const jobs = data.recent_jobs || [];
      const jobsBody = document.getElementById("jobsBody");
      if (!jobs.length) {
        jobsBody.innerHTML = '<tr><td colspan="6" class="empty">No jobs queued yet.</td></tr>';
      } else {
        jobsBody.innerHTML = jobs.map((job) => {
          const canCancel = job.status === "queued" || job.status === "running";
          const cancelBtn = canCancel
            ? `<button class="btn btn-sm btn-deny" onclick="cancelJob(${job.id})">Cancel</button>`
            : "";
          const breakdown = `matched:${job.matched_count} review:${job.needs_review_count} agent:${job.agent_recovered_count} 404:${job.not_found_count} err:${job.error_count} cancel:${job.cancelled_count}`;
          return `
            <tr>
              <td class="mono">#${job.id}</td>
              <td><strong>${escHtml(job.label || job.job_type)}</strong><br/><span class="muted-note mono">${escHtml(job.job_type)}</span></td>
              <td>${stageBadge(job.status)}</td>
              <td>
                <div class="progress-wrap"><div class="progress-bar" style="width:${job.progress_pct}%"></div></div>
                <span class="mono">${job.completed_items}/${job.total_items} (${job.progress_pct}%)</span>
              </td>
              <td class="mono">${escHtml(breakdown)}</td>
              <td><div class="actions-cell"><button class="btn btn-sm" onclick="inspectJob(${job.id})">Inspect Items</button>${cancelBtn}</div></td>
            </tr>
          `;
        }).join("");
      }

      document.getElementById("pollMeta").textContent =
        (paused ? "Paused — " : "Live — ") + "Updated " + new Date().toISOString().slice(11, 19) + " UTC";
    }

    async function pollStatus() {
      try {
        const res = await fetch("/jobs/status");
        if (res.ok) {
          const data = await res.json();
          renderStatus(data);
        }
        if (selectedJobId !== null) {
          await loadJobDetail(selectedJobId);
        }
      } catch (_e) {}
    }

    function togglePolling() {
      paused = !paused;
      document.getElementById("pollToggleBtn").textContent = paused
        ? "Resume Auto-Poll"
        : "Pause Auto-Poll (1s)";
      if (!paused) pollStatus();
    }

    async function loadJobDetail(jobId) {
      const res = await fetch(`/jobs/${jobId}`);
      if (!res.ok) {
        if (res.status === 404) {
          selectedJobId = null;
          document.getElementById("jobDetailPanel").style.display = "none";
        }
        return;
      }
      const detail = await res.json();
      document.getElementById("jobDetailPanel").style.display = "block";
      document.getElementById("jobDetailHeading").textContent =
        `Job #${detail.job.id} Items (${detail.items.length})`;
      document.getElementById("jobDetailBody").innerHTML = detail.items.map((item) => `
        <tr>
          <td class="mono">${item.row_number}</td>
          <td><strong>${escHtml(item.album_title || item.barcode || "—")}</strong><br/><span class="muted-note">${escHtml(item.album_artist || "")}</span></td>
          <td>${stageBadge(item.stage)}</td>
          <td class="mono">${escHtml(item.worker_name || "—")}</td>
          <td>${item.result_status ? stageBadge(item.result_status) : "—"}</td>
          <td class="mono">${escHtml(item.result_source || "—")}</td>
          <td class="mono">${item.track_count}</td>
          <td>${escHtml(item.status_message || item.error_detail || "—")}</td>
        </tr>
      `).join("");
    }

    async function inspectJob(jobId) {
      selectedJobId = jobId;
      await loadJobDetail(jobId);
    }

    async function cancelJob(jobId) {
      const res = await fetch(`/jobs/${jobId}/cancel`, { method: "POST" });
      if (res.ok) {
        await pollStatus();
      }
    }

    async function retryNotFound() {
      const msgEl = document.getElementById("actionMsg");
      const res = await fetch("/jobs/retry-not-found", { method: "POST" });
      const data = await res.json().catch(() => ({}));
      if (res.ok) {
        msgEl.textContent = `Enqueued Job #${data.id} (${data.total_items} records)`;
        await pollStatus();
      } else {
        msgEl.textContent = data.detail || "Failed to enqueue retry job";
      }
    }

    async function enqueueEnrichMissing() {
      const msgEl = document.getElementById("actionMsg");
      const useAgent = Boolean(document.getElementById("enrichMissingUseAgent")?.checked);
      const res = await fetch("/jobs/enrich-missing", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ use_agent: useAgent, force_refresh: true })
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok) {
        msgEl.textContent = data.total_items
          ? `Enqueued Enrich Job #${data.id} (${data.total_items} records)`
          : "No un-enriched records need backfill";
        await pollStatus();
      } else {
        msgEl.textContent = data.detail || "Failed to enqueue enrichment job";
      }
    }

    async function enqueueSingle(ev) {
      ev.preventDefault();
      const msgEl = document.getElementById("actionMsg");
      const payload = {
        barcode: document.getElementById("inBarcode").value,
        album_title: document.getElementById("inTitle").value,
        album_artist: document.getElementById("inArtist").value,
        mode: document.getElementById("inMode").value,
      };
      const res = await fetch("/jobs/lookup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok) {
        msgEl.textContent = `Enqueued Job #${data.job.id}`;
        document.getElementById("inBarcode").value = "";
        document.getElementById("inTitle").value = "";
        document.getElementById("inArtist").value = "";
        await pollStatus();
      } else {
        msgEl.textContent = data.detail || "Failed to enqueue lookup";
      }
    }

    async function restartDemo() {
      const clearCache = Boolean(document.getElementById("restartClearCache")?.checked);
      const cacheNote = clearCache ? " AND clear cached HTTP API responses" : " (keeping cached API responses for smooth live replay)";
      if (!confirm(`Reset all current records/jobs${cacheNote} and reload all 25 songs into the live queue?`)) return;
      const btn = document.getElementById("restartDemoBtn");
      const msgEl = document.getElementById("actionMsg");
      if (btn) btn.disabled = true;
      msgEl.textContent = "Resetting DB & queuing 25 songs...";
      try {
        const res = await fetch("/jobs/restart-demo", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ clear_api_cache: clearCache })
        });
        const data = await res.json().catch(() => ({}));
        if (res.ok) {
          selectedJobId = data.id;
          if (paused) togglePolling();
          msgEl.textContent = `Restarted! Queued Job #${data.id} (${data.total_items} songs)`;
          await pollStatus();
        } else {
          msgEl.textContent = data.detail || "Failed to restart 25 songs";
        }
      } catch (_e) {
        msgEl.textContent = "Network error while restarting";
      } finally {
        if (btn) btn.disabled = false;
      }
    }

    renderStatus(INITIAL_STATUS);
    const jobParam = new URLSearchParams(window.location.search).get("job");
    if (jobParam && /^\\d+$/.test(jobParam) && Number(jobParam) > 0) {
      selectedJobId = Number(jobParam);
      loadJobDetail(selectedJobId);
    }
    setInterval(() => { if (!paused) pollStatus(); }, 1000);
  </script>
</body>
</html>"""


def render_queue_dashboard(status: QueueStatusOut) -> str:
    initial_json = json.dumps(status.model_dump(), ensure_ascii=False).replace("</", "<\\/")
    return (
        _QUEUE_HTML_TEMPLATE.replace("__BASE_CSS__", _BASE_CSS).replace(
            "__INITIAL_STATUS_JSON__", initial_json
        )
    )
