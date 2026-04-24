from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .models import BenchmarkResult


def _score_color(score: float | None) -> str:
    if score is None:
        return "#888888"
    if score >= 0.8:
        return "#2d6a4f"
    if score >= 0.5:
        return "#b5770d"
    return "#b91c1c"


def _score_bg(score: float | None) -> str:
    if score is None:
        return "#e5e5e5"
    if score >= 0.8:
        return "#d8f3dc"
    if score >= 0.5:
        return "#fef3c7"
    return "#fee2e2"


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _score_label(score: float | None) -> str:
    if score is None:
        return "N/A"
    return f"{score:.3f}"


def _bar_html(score: float | None) -> str:
    if score is None:
        pct = 0
        label = "N/A"
    else:
        pct = int(score * 100)
        label = f"{score:.3f}"
    color = _score_color(score)
    return (
        f'<div style="background:#2a2a3e;border-radius:4px;height:22px;width:160px;overflow:hidden;display:inline-block;vertical-align:middle">'
        f'<div style="background:{color};width:{pct}%;height:100%;display:flex;align-items:center;padding-left:6px;'
        f'font-size:12px;color:#fff;font-weight:600;min-width:36px">{label}</div>'
        f"</div>"
    )


def _pill(score: float | None) -> str:
    label = _score_label(score)
    color = _score_color(score)
    bg = _score_bg(score)
    border = color
    return (
        f'<span style="display:inline-flex;align-items:center;justify-content:center;'
        f"background:{bg};color:{color};border:1px solid {border};border-radius:2px;"
        f"padding:2px 7px;font-size:.72rem;font-weight:700;min-width:42px;"
        f'font-family:monospace">{label}</span>'
    )


def _detail_block(label: str, content: str, mono: bool = False) -> str:
    safe = _escape_html(content)
    style = (
        "white-space:pre-wrap;font-family:'SFMono Regular','Consolas',monospace;font-size:.74rem;line-height:1.5"
        if mono
        else "color:var(--text);line-height:1.55"
    )
    return (
        f'<div style="background:var(--bg);border-left:2px solid var(--accent);'
        f'padding:9px 12px;margin-bottom:7px">'
        f'<strong style="font-size:.63rem;text-transform:uppercase;letter-spacing:.07em;'
        f'color:var(--muted);display:block;margin-bottom:4px;font-family:monospace">{label}</strong>'
        f'<p style="{style}">{safe}</p>'
        f"</div>"
    )


def _stat_table_html(stats: dict[str, float], unit: str, keys: list[str]) -> str:
    if not stats:
        return '<p style="color:var(--muted);font-style:italic;font-size:.84rem">No data.</p>'
    max_val = max((stats.get(k, 0) for k in keys), default=1) or 1
    rows = ""
    for key in keys:
        val = stats.get(key)
        if val is None:
            continue
        bar_w = int((val / max_val) * 100)
        rows += (
            f"<tr>"
            f'<td style="padding:8px 14px;width:50px;text-align:right;font-family:monospace;'
            f'font-size:.78rem;color:var(--muted)">{key}</td>'
            f'<td style="padding:8px 14px;width:100%">'
            f'<div style="background:var(--surface2);height:12px;border:1px solid var(--border);border-radius:1px;overflow:hidden">'
            f'<div style="width:{bar_w}%;height:100%;background:var(--accent);opacity:.7"></div></div></td>'
            f'<td style="padding:8px 14px;text-align:right;font-family:monospace;font-size:.78rem;'
            f'color:var(--muted);white-space:nowrap">{val:,.0f} {unit}</td>'
            f"</tr>"
        )
    return f'<table style="border-collapse:collapse;width:100%"><tbody>{rows}</tbody></table>'


def _generate_csv(result: BenchmarkResult) -> str:
    """Generate CSV by flattening the JSON structure — auto-discovers all metric columns."""
    if not result.per_sample:
        return ""

    # Flatten each SampleScore into a plain dict then let the union of all keys
    # drive the columns — no hardcoding needed when metrics change.
    flat_rows: list[dict] = []
    for ss in result.per_sample:
        row: dict = {
            "run_id": result.run_id,
            "sample_id": ss.sample_id,
            "question": ss.question,
            "answer": ss.answer,
            "reference_answer": ss.reference_answer,
            "context_string": ss.context_string,
            "context_tokens": ss.context_tokens,
            "latency_ms": round(ss.latency_ms, 1),
        }
        # Flatten scores and reasons — one column per metric, one per reason
        for m, score in ss.scores.items():
            row[m] = score if score is not None else ""
        for m, reason in ss.reasons.items():
            row[f"{m}_reason"] = reason or ""
        flat_rows.append(row)

    # Column order: fixed fields first, then metrics (scores then reasons) in discovery order
    fixed = [
        "run_id",
        "sample_id",
        "question",
        "answer",
        "reference_answer",
        "context_string",
        "context_tokens",
        "latency_ms",
    ]
    metrics = list(result.per_sample[0].scores.keys())
    fieldnames = fixed + metrics + [f"{m}_reason" for m in metrics]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(flat_rows)
    return buf.getvalue()


def _generate_html(result: BenchmarkResult) -> str:
    # ── Aggregate score bars ──────────────────────────────────────────────────
    agg_rows = ""
    for metric, score in result.aggregate_scores.items():
        color = _score_color(score)
        pct = int((score or 0) * 100)
        cls = "high" if (score or 0) >= 0.8 else ("mid" if (score or 0) >= 0.5 else "low")
        agg_rows += (
            f"<tr>"
            f'<td style="padding:10px 14px;font-size:.84rem;width:200px;white-space:nowrap">'
            f"{metric.replace('_', ' ').title()}</td>"
            f'<td style="padding:10px 14px;width:100%">'
            f'<div style="background:var(--surface2);height:16px;border:1px solid var(--border);border-radius:2px;overflow:hidden">'
            f'<div class="st-fill {cls}" style="width:{pct}%;height:100%"></div></div></td>'
            f'<td style="padding:10px 14px;text-align:right;font-weight:700;font-family:monospace;'
            f'font-size:.84rem;color:{color};white-space:nowrap">{_score_label(score)}</td>'
            f"</tr>"
        )

    # ── Per-sample rows ───────────────────────────────────────────────────────
    if result.per_sample:
        all_metrics = list(result.per_sample[0].scores.keys())
    else:
        all_metrics = list(result.aggregate_scores.keys())

    metric_headers = "".join(
        f'<th onclick="sortTable({i + 2})">{m.replace("_", " ").title()}</th>' for i, m in enumerate(all_metrics)
    )

    sample_rows = ""
    for idx, ss in enumerate(result.per_sample):
        # Metric pills for the collapsed row
        metric_cells = "".join(f"<td>{_pill(ss.scores.get(m))}</td>" for m in all_metrics)

        # Expandable detail section
        detail_blocks = _detail_block("Answer", ss.answer or "—")
        if ss.context_string:
            detail_blocks += _detail_block("Retrieved Context (build_context_string)", ss.context_string, mono=True)
        detail_blocks += _detail_block("Reference Answer", ss.reference_answer or "—")
        for m in all_metrics:
            reason = ss.reasons.get(m)
            if reason:
                detail_blocks += _detail_block(f"{m.replace('_', ' ').title()} — Reason", reason)

        row_id = f"row-{idx}"
        sample_rows += (
            f"<tr>"
            f'<td style="color:var(--muted);font-size:.72rem;font-family:monospace">{idx + 1}</td>'
            f'<td class="q-cell">'
            f'<div class="q-summary" onclick="toggleDetail(\'{row_id}\')">'
            f'<span class="chevron" id="chev-{row_id}">▶</span>'
            f"{_escape_html(ss.sample_id)} — {_escape_html(ss.question[:80])}{'…' if len(ss.question) > 80 else ''}"
            f"</div>"
            f'<div class="q-detail" id="{row_id}">{detail_blocks}</div>'
            f"</td>"
            f"{metric_cells}"
            f'<td style="font-size:.73rem;color:var(--muted);font-family:monospace;white-space:nowrap">'
            f"{ss.latency_ms:.0f} ms</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_escape_html(result.name)} — DeepEval Report</title>
<style>
  :root{{
    --bg:#fdf8f0;--surface:#fffef9;--surface2:#f5efe3;--border:#e2d9c8;
    --text:#1c1208;--muted:#7a6a52;--accent:#8b5e1a;--accent2:#c4831e;
    --green:#2d6a4f;--yellow:#b5770d;--red:#b91c1c;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);padding:48px 40px;max-width:1200px;margin:0 auto}}
  @media(max-width:700px){{body{{padding:24px 16px}}}}
  h1{{font-size:1.9rem;font-weight:700;letter-spacing:-.01em;border-bottom:2px solid var(--text);padding-bottom:16px;margin-bottom:8px}}
  .byline{{font-size:.82rem;color:var(--muted);margin-bottom:36px;padding-top:8px;border-top:1px solid var(--border)}}
  .byline strong{{color:var(--text)}}
  .sh{{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin-bottom:14px;display:flex;align-items:center;gap:10px;font-family:monospace}}
  .sh::after{{content:'';flex:1;height:1px;background:var(--border)}}
  .section{{margin-bottom:44px}}
  .stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;margin-bottom:40px}}
  .stat-box{{background:var(--surface);border:1px solid var(--border);padding:16px 20px}}
  .stat-val{{font-size:1.8rem;font-weight:700;color:var(--accent);font-family:monospace}}
  .stat-lbl{{font-size:.65rem;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-top:4px;font-family:monospace}}
  .score-table{{width:100%;border-collapse:collapse;background:var(--surface);border:1px solid var(--border)}}
  .score-table tr{{border-bottom:1px solid var(--border)}}
  .score-table tr:last-child{{border-bottom:none}}
  .st-fill.high{{background:#4a7c59}}
  .st-fill.mid{{background:#b5770d}}
  .st-fill.low{{background:#b91c1c}}
  .table-wrap{{background:var(--surface);border:1px solid var(--border);overflow:hidden}}
  .table-toolbar{{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid var(--border);background:var(--surface2);flex-wrap:wrap;gap:10px}}
  .search-box{{background:var(--surface);border:1px solid var(--border);padding:6px 11px;color:var(--text);font-size:.82rem;width:260px;outline:none}}
  .search-box:focus{{border-color:var(--accent2)}}
  .search-box::placeholder{{color:var(--muted)}}
  .row-count{{font-size:.75rem;color:var(--muted);font-family:monospace}}
  table{{width:100%;border-collapse:collapse}}
  thead th{{background:var(--surface2);font-size:.64rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);padding:10px 12px;text-align:left;border-bottom:2px solid var(--border);white-space:nowrap;cursor:pointer;user-select:none;font-family:monospace}}
  thead th:hover{{color:var(--text)}}
  thead th.sorted{{color:var(--accent2)}}
  tbody tr{{border-bottom:1px solid var(--border)}}
  tbody tr:last-child{{border-bottom:none}}
  tbody tr:nth-child(even){{background:var(--surface2)}}
  tbody tr:hover{{background:var(--border)}}
  td{{padding:10px 12px;font-size:.81rem;vertical-align:middle}}
  .q-cell{{min-width:280px}}
  .q-summary{{cursor:pointer;color:var(--text);font-size:.81rem;line-height:1.45;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}}
  .q-detail{{display:none;margin-top:10px;font-size:.78rem}}
  .q-detail.open{{display:block}}
  .chevron{{display:inline-block;transition:transform .18s;margin-right:5px;font-size:.65rem;color:var(--muted)}}
  .chevron.open{{transform:rotate(90deg)}}
  .no-results{{padding:40px;text-align:center;color:var(--muted);font-size:.88rem;font-style:italic}}
  .footer{{margin-top:56px;padding-top:16px;border-top:1px solid var(--border);font-size:.75rem;color:var(--muted);text-align:center;font-family:monospace}}
</style>
</head>
<body>

<h1>{_escape_html(result.name)}</h1>
<div class="byline">
  Run ID: <strong>{result.run_id}</strong> &nbsp;|&nbsp; {result.timestamp}
</div>

<div class="stat-grid">
  <div class="stat-box"><div class="stat-val">{result.total_samples}</div><div class="stat-lbl">Total Samples</div></div>
  <div class="stat-box"><div class="stat-val">{result.error_count}</div><div class="stat-lbl">Errors</div></div>
  <div class="stat-box"><div class="stat-val">{result.latency_p50_ms:.0f} ms</div><div class="stat-lbl">Latency p50</div></div>
  <div class="stat-box"><div class="stat-val">{result.latency_p95_ms:.0f} ms</div><div class="stat-lbl">Latency p95</div></div>
</div>

<div class="section">
  <div class="sh">Aggregate Scores</div>
  <table class="score-table">
    <tbody>{agg_rows}</tbody>
  </table>
</div>

<div class="section">
  <div class="sh">Performance</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:start">
    <div>
      <div style="font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:10px;font-family:monospace">Query Latency — E2E (ms)</div>
      <div style="background:var(--surface);border:1px solid var(--border);padding:16px">
        {_stat_table_html(result.latency_stats, "ms", ["min", "mean", "p50", "p75", "p95", "p99", "max"])}
      </div>
    </div>
    <div>
      <div style="font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:10px;font-family:monospace">Context Tokens per Query (build_context_string)</div>
      <div style="background:var(--surface);border:1px solid var(--border);padding:16px">
        {_stat_table_html(result.context_token_stats, "tok", ["min", "mean", "max"])}
      </div>
    </div>
  </div>
</div>

<div class="section">
  <div class="sh">Per-Sample Results</div>
  <p style="font-size:.8rem;color:var(--muted);margin-bottom:12px">Click any row to expand answer, retrieved context, reference answer, and metric reasons.</p>
  <div class="table-wrap">
    <div class="table-toolbar">
      <input class="search-box" type="text" id="tableSearch" placeholder="Search questions…" oninput="filterTable()">
      <div class="row-count" id="rowCount"></div>
    </div>
    <div style="overflow-x:auto">
    <table id="resultsTable">
      <thead>
        <tr>
          <th style="width:36px">#</th>
          <th onclick="sortTable(1)" style="min-width:280px">Question</th>
          {metric_headers}
          <th onclick="sortTable({len(all_metrics) + 2})">Latency</th>
        </tr>
      </thead>
      <tbody id="tableBody">
      {sample_rows}
      </tbody>
    </table>
    <div class="no-results" id="noResults" style="display:none">No matching questions found.</div>
    </div>
  </div>
</div>

<div class="footer">HydraDB DeepEval Benchmark &nbsp;·&nbsp; {result.run_id} &nbsp;·&nbsp; {result.timestamp}</div>

<script>
function toggleDetail(id) {{
  const detail = document.getElementById(id);
  const chevron = document.getElementById('chev-' + id);
  detail.classList.toggle('open');
  chevron.classList.toggle('open');
}}

function filterTable() {{
  const q = document.getElementById('tableSearch').value.toLowerCase();
  const rows = document.querySelectorAll('#tableBody tr');
  let visible = 0;
  rows.forEach(row => {{
    const text = row.querySelector('.q-summary').textContent.toLowerCase();
    const show = text.includes(q);
    row.style.display = show ? '' : 'none';
    if (show) visible++;
  }});
  document.getElementById('rowCount').textContent = visible + ' of ' + rows.length + ' rows';
  document.getElementById('noResults').style.display = visible === 0 ? 'block' : 'none';
}}

function sortTable(col) {{
  const tbody = document.getElementById('tableBody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const th = document.querySelectorAll('thead th')[col];
  const asc = th.dataset.sort !== 'asc';
  document.querySelectorAll('thead th').forEach(t => {{ t.classList.remove('sorted'); delete t.dataset.sort; }});
  th.classList.add('sorted');
  th.dataset.sort = asc ? 'asc' : 'desc';
  rows.sort((a, b) => {{
    const av = a.cells[col]?.textContent.trim().replace('ms','') || '';
    const bv = b.cells[col]?.textContent.trim().replace('ms','') || '';
    const an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
  }});
  rows.forEach(r => tbody.appendChild(r));
}}

window.addEventListener('DOMContentLoaded', () => {{
  const rows = document.querySelectorAll('#tableBody tr');
  if (rows.length) document.getElementById('rowCount').textContent = rows.length + ' rows';
}});
</script>
</body>
</html>
"""
    return html


def _generate_comparison_html(hydra: BenchmarkResult, sm: BenchmarkResult) -> str:
    h_label = hydra.name
    s_label = sm.name
    title = f"{h_label} vs {s_label}"
    all_metrics = sorted(set(hydra.aggregate_scores) | set(sm.aggregate_scores))

    # ── Aggregate score rows ─────────────────────────────────────────────────
    agg_rows = ""
    for metric in all_metrics:
        h = hydra.aggregate_scores.get(metric)
        s = sm.aggregate_scores.get(metric)
        h_pct = int((h or 0) * 100)
        s_pct = int((s or 0) * 100)
        if h is not None and s is not None:
            if h > s + 0.005:
                winner = f'<span style="color:#2563eb;font-weight:700">▲ {_escape_html(h_label)}</span>'
            elif s > h + 0.005:
                winner = f'<span style="color:#7c3aed;font-weight:700">▲ {_escape_html(s_label)}</span>'
            else:
                winner = '<span style="color:var(--muted)">Tie</span>'
        else:
            winner = "—"
        h_color = _score_color(h)
        s_color = _score_color(s)
        agg_rows += f"""
        <tr>
          <td style="padding:12px 16px;font-size:.84rem;width:200px;white-space:nowrap">{metric.replace("_", " ").title()}</td>
          <td style="padding:12px 16px;width:220px">
            <div style="background:var(--surface2);height:14px;border-radius:2px;overflow:hidden;border:1px solid var(--border)">
              <div style="width:{h_pct}%;height:100%;background:#2563eb;opacity:.75"></div></div>
            <span style="font-family:monospace;font-size:.78rem;color:{h_color};font-weight:700">{_score_label(h)}</span>
          </td>
          <td style="padding:12px 16px;width:220px">
            <div style="background:var(--surface2);height:14px;border-radius:2px;overflow:hidden;border:1px solid var(--border)">
              <div style="width:{s_pct}%;height:100%;background:#7c3aed;opacity:.75"></div></div>
            <span style="font-family:monospace;font-size:.78rem;color:{s_color};font-weight:700">{_score_label(s)}</span>
          </td>
          <td style="padding:12px 16px;font-size:.82rem">{winner}</td>
        </tr>"""

    # ── Latency comparison ───────────────────────────────────────────────────
    lat_keys = ["min", "mean", "p50", "p75", "p95", "p99", "max"]
    lat_rows = ""
    for k in lat_keys:
        hv = hydra.latency_stats.get(k)
        sv = sm.latency_stats.get(k)
        if hv is None and sv is None:
            continue
        faster = ""
        if hv is not None and sv is not None:
            if hv < sv - 5:
                faster = '<span style="color:#2563eb;font-size:.7rem">faster</span>'
            elif sv < hv - 5:
                faster = '<span style="color:#7c3aed;font-size:.7rem">faster</span>'
        lat_rows += (
            f"<tr>"
            f'<td style="padding:7px 14px;font-family:monospace;font-size:.78rem;color:var(--muted)">{k}</td>'
            f'<td style="padding:7px 14px;font-family:monospace;font-size:.78rem;color:#2563eb">{f"{hv:,.0f} ms" if hv is not None else "—"}</td>'
            f'<td style="padding:7px 14px;font-family:monospace;font-size:.78rem;color:#7c3aed">{f"{sv:,.0f} ms" if sv is not None else "—"}</td>'
            f'<td style="padding:7px 14px;font-size:.72rem">{faster}</td>'
            f"</tr>"
        )

    # ── Per-sample table ─────────────────────────────────────────────────────
    hydra_by_id = {ss.sample_id: ss for ss in hydra.per_sample}
    sm_by_id = {ss.sample_id: ss for ss in sm.per_sample}
    all_ids = list(dict.fromkeys([ss.sample_id for ss in hydra.per_sample] + [ss.sample_id for ss in sm.per_sample]))

    metric_headers = "".join(
        f'<th colspan="2" style="text-align:center">{m.replace("_", " ").title()}</th>' for m in all_metrics
    )
    h_abbr = _escape_html(h_label[:6])
    s_abbr = _escape_html(s_label[:6])
    sub_headers = "".join(
        f'<th style="color:#2563eb;font-size:.6rem">{h_abbr}</th>'
        f'<th style="color:#7c3aed;font-size:.6rem">{s_abbr}</th>'
        for _ in all_metrics
    )

    sample_rows = ""
    for idx, sid in enumerate(all_ids):
        h_ss = hydra_by_id.get(sid)
        s_ss = sm_by_id.get(sid)
        question = (h_ss or s_ss).question
        h_lat = f"{h_ss.latency_ms:.0f} ms" if h_ss else "—"
        s_lat = f"{s_ss.latency_ms:.0f} ms" if s_ss else "—"

        metric_cells = ""
        for m in all_metrics:
            h_score = h_ss.scores.get(m) if h_ss else None
            s_score = s_ss.scores.get(m) if s_ss else None
            metric_cells += f"<td>{_pill(h_score)}</td><td>{_pill(s_score)}</td>"

        detail = ""
        if h_ss:
            detail += _detail_block(f"{h_label} Answer", h_ss.answer or "—")
        if s_ss:
            detail += _detail_block(f"{s_label} Answer", s_ss.answer or "—")
        if h_ss and h_ss.context_string:
            detail += _detail_block(f"{h_label} Context", h_ss.context_string, mono=True)
        if s_ss and s_ss.context_string:
            detail += _detail_block(f"{s_label} Context", s_ss.context_string, mono=True)
        ref = (h_ss or s_ss).reference_answer
        if ref:
            detail += _detail_block("Reference Answer", ref)

        row_id = f"crow-{idx}"
        sample_rows += (
            f"<tr>"
            f'<td style="color:var(--muted);font-size:.72rem;font-family:monospace">{idx + 1}</td>'
            f'<td class="q-cell">'
            f'<div class="q-summary" onclick="toggleDetail(\'{row_id}\')">'
            f'<span class="chevron" id="chev-{row_id}">▶</span>'
            f"{_escape_html(sid)} — {_escape_html(question[:80])}{'…' if len(question) > 80 else ''}"
            f"</div>"
            f'<div class="q-detail" id="{row_id}">{detail}</div>'
            f"</td>"
            f"{metric_cells}"
            f'<td style="font-family:monospace;font-size:.72rem;color:#2563eb">{h_lat}</td>'
            f'<td style="font-family:monospace;font-size:.72rem;color:#7c3aed">{s_lat}</td>'
            f"</tr>"
        )

    run_id = hydra.run_id
    timestamp = hydra.timestamp
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_escape_html(title)} — Comparison Report</title>
<style>
  :root{{
    --bg:#fdf8f0;--surface:#fffef9;--surface2:#f5efe3;--border:#e2d9c8;
    --text:#1c1208;--muted:#7a6a52;--accent:#8b5e1a;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);padding:48px 40px;max-width:1200px;margin:0 auto}}
  h1{{font-size:1.9rem;font-weight:700;border-bottom:2px solid var(--text);padding-bottom:16px;margin-bottom:8px}}
  .byline{{font-size:.82rem;color:var(--muted);margin-bottom:36px;padding-top:8px;border-top:1px solid var(--border)}}
  .byline strong{{color:var(--text)}}
  .legend{{display:flex;gap:24px;margin-bottom:28px;font-size:.82rem;font-weight:600}}
  .dot{{width:12px;height:12px;border-radius:50%;display:inline-block;margin-right:6px;vertical-align:middle}}
  .sh{{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin-bottom:14px;display:flex;align-items:center;gap:10px;font-family:monospace}}
  .sh::after{{content:'';flex:1;height:1px;background:var(--border)}}
  .section{{margin-bottom:44px}}
  .stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin-bottom:40px}}
  .stat-box{{background:var(--surface);border:1px solid var(--border);padding:16px 20px}}
  .stat-val{{font-size:1.6rem;font-weight:700;font-family:monospace}}
  .stat-lbl{{font-size:.65rem;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-top:4px;font-family:monospace}}
  .score-table{{width:100%;border-collapse:collapse;background:var(--surface);border:1px solid var(--border)}}
  .score-table tr{{border-bottom:1px solid var(--border)}}
  .score-table tr:last-child{{border-bottom:none}}
  .score-table thead th{{background:var(--surface2);font-size:.64rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);padding:10px 16px;text-align:left;border-bottom:2px solid var(--border);font-family:monospace}}
  .table-wrap{{background:var(--surface);border:1px solid var(--border);overflow:hidden}}
  .table-toolbar{{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid var(--border);background:var(--surface2);flex-wrap:wrap;gap:10px}}
  .search-box{{background:var(--surface);border:1px solid var(--border);padding:6px 11px;color:var(--text);font-size:.82rem;width:260px;outline:none}}
  .search-box:focus{{border-color:var(--accent)}}
  .search-box::placeholder{{color:var(--muted)}}
  .row-count{{font-size:.75rem;color:var(--muted);font-family:monospace}}
  table{{width:100%;border-collapse:collapse}}
  thead th{{background:var(--surface2);font-size:.64rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);padding:10px 12px;text-align:left;border-bottom:2px solid var(--border);white-space:nowrap;font-family:monospace}}
  tbody tr{{border-bottom:1px solid var(--border)}}
  tbody tr:last-child{{border-bottom:none}}
  tbody tr:nth-child(even){{background:var(--surface2)}}
  tbody tr:hover{{background:var(--border)}}
  td{{padding:10px 12px;font-size:.81rem;vertical-align:middle}}
  .q-cell{{min-width:260px}}
  .q-summary{{cursor:pointer;color:var(--text);font-size:.81rem;line-height:1.45}}
  .q-detail{{display:none;margin-top:10px;font-size:.78rem}}
  .q-detail.open{{display:block}}
  .chevron{{display:inline-block;transition:transform .18s;margin-right:5px;font-size:.65rem;color:var(--muted)}}
  .chevron.open{{transform:rotate(90deg)}}
  .footer{{margin-top:56px;padding-top:16px;border-top:1px solid var(--border);font-size:.75rem;color:var(--muted);text-align:center;font-family:monospace}}
</style>
</head>
<body>

<h1>{_escape_html(title)}</h1>
<div class="byline">Run ID: <strong>{run_id}</strong> &nbsp;|&nbsp; {timestamp}</div>

<div class="legend">
  <span><span class="dot" style="background:#2563eb"></span>{_escape_html(h_label)}</span>
  <span><span class="dot" style="background:#7c3aed"></span>{_escape_html(s_label)}</span>
</div>

<div class="stat-grid">
  <div class="stat-box"><div class="stat-val" style="color:#2563eb">{hydra.latency_p50_ms:.0f} ms</div><div class="stat-lbl">{_escape_html(h_label)} p50</div></div>
  <div class="stat-box"><div class="stat-val" style="color:#7c3aed">{sm.latency_p50_ms:.0f} ms</div><div class="stat-lbl">{_escape_html(s_label)} p50</div></div>
  <div class="stat-box"><div class="stat-val" style="color:#2563eb">{hydra.latency_p95_ms:.0f} ms</div><div class="stat-lbl">{_escape_html(h_label)} p95</div></div>
  <div class="stat-box"><div class="stat-val" style="color:#7c3aed">{sm.latency_p95_ms:.0f} ms</div><div class="stat-lbl">{_escape_html(s_label)} p95</div></div>
  <div class="stat-box"><div class="stat-val">{hydra.error_count}/{hydra.total_samples}</div><div class="stat-lbl">{_escape_html(h_label)} Errors</div></div>
  <div class="stat-box"><div class="stat-val">{sm.error_count}/{sm.total_samples}</div><div class="stat-lbl">{_escape_html(s_label)} Errors</div></div>
</div>

<div class="section">
  <div class="sh">Aggregate Scores</div>
  <table class="score-table">
    <thead><tr><th>Metric</th><th style="color:#2563eb">{_escape_html(h_label)}</th><th style="color:#7c3aed">{_escape_html(s_label)}</th><th>Winner</th></tr></thead>
    <tbody>{agg_rows}</tbody>
  </table>
</div>

<div class="section">
  <div class="sh">Latency Distribution</div>
  <table class="score-table">
    <thead><tr><th>Percentile</th><th style="color:#2563eb">{_escape_html(h_label)}</th><th style="color:#7c3aed">{_escape_html(s_label)}</th><th></th></tr></thead>
    <tbody>{lat_rows}</tbody>
  </table>
</div>

<div class="section">
  <div class="sh">Per-Sample Comparison</div>
  <p style="font-size:.8rem;color:var(--muted);margin-bottom:12px">Click any row to expand answers and retrieved contexts from both providers.</p>
  <div class="table-wrap">
    <div class="table-toolbar">
      <input class="search-box" type="text" id="tableSearch" placeholder="Search questions…" oninput="filterTable()">
      <div class="row-count" id="rowCount"></div>
    </div>
    <div style="overflow-x:auto">
    <table id="resultsTable">
      <thead>
        <tr>
          <th style="width:36px">#</th>
          <th style="min-width:260px">Question</th>
          {metric_headers}
          <th style="color:#2563eb">{h_abbr} Lat</th>
          <th style="color:#7c3aed">{s_abbr} Lat</th>
        </tr>
        <tr>
          <th></th><th></th>
          {sub_headers}
          <th></th><th></th>
        </tr>
      </thead>
      <tbody id="tableBody">{sample_rows}</tbody>
    </table>
    </div>
  </div>
</div>

<div class="footer">{_escape_html(title)} &nbsp;·&nbsp; {run_id}</div>

<script>
function toggleDetail(id){{
  const d=document.getElementById(id),c=document.getElementById('chev-'+id);
  d.classList.toggle('open');c.classList.toggle('open');
}}
function filterTable(){{
  const q=document.getElementById('tableSearch').value.toLowerCase();
  const rows=document.querySelectorAll('#tableBody tr');
  let v=0;
  rows.forEach(r=>{{
    const t=r.querySelector('.q-summary');
    const show=t&&t.textContent.toLowerCase().includes(q);
    r.style.display=show?'':'none';
    if(show)v++;
  }});
  document.getElementById('rowCount').textContent=v+' of '+rows.length+' rows';
}}
window.addEventListener('DOMContentLoaded',()=>{{
  const rows=document.querySelectorAll('#tableBody tr');
  if(rows.length)document.getElementById('rowCount').textContent=rows.length+' rows';
}});
</script>
</body>
</html>"""


class BenchmarkReporter:
    def __init__(self, output_dir: str, formats: list[str], include_per_sample: bool) -> None:
        self._output_dir = Path(output_dir)
        self._formats = formats
        self._include_per_sample = include_per_sample

    def save(self, result: BenchmarkResult) -> list[Path]:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        slug = result.run_id.replace(":", "-").replace(" ", "_")

        if "json" in self._formats:
            data = result.model_dump()
            if not self._include_per_sample:
                data.pop("per_sample", None)
            path = self._output_dir / f"{slug}.json"
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            saved.append(path)

        if "html" in self._formats:
            html = _generate_html(result)
            path = self._output_dir / f"{slug}.html"
            path.write_text(html, encoding="utf-8")
            saved.append(path)

        if "csv" in self._formats and self._include_per_sample and result.per_sample:
            path = self._output_dir / f"{slug}.csv"
            path.write_text(_generate_csv(result), encoding="utf-8", newline="")
            saved.append(path)

        return saved

    def save_comparison(
        self,
        hydra: BenchmarkResult,
        sm: BenchmarkResult,
        run_id: str,
    ) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        html = _generate_comparison_html(hydra, sm)
        path = self._output_dir / f"{run_id}_comparison.html"
        path.write_text(html, encoding="utf-8")
        return path
