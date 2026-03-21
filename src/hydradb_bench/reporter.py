"""Report generation: JSON, CSV, and self-contained HTML dashboard."""

from __future__ import annotations

import json
import logging
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import BenchmarkResult, ReportingConfig, TokenUsageResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTML template (self-contained — no CDN, no JS frameworks)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ benchmark_name }} — {{ run_id }}</title>
<style>
  :root{
    --green:#2d6a4f;--green-bg:#d8f3dc;--green-text:#1b4332;
    --yellow:#b5770d;--yellow-bg:#fef3c7;--yellow-text:#7c4f00;
    --red:#b91c1c;--red-bg:#fee2e2;--red-text:#7f1d1d;
    --blue:#1d4ed8;--blue-bg:#dbeafe;--blue-text:#1e3a8a;
    --purple:#6d28d9;--purple-bg:#ede9fe;--purple-text:#4c1d95;
    --bg:#fdf8f0;--surface:#fffef9;--surface2:#f5efe3;--border:#e2d9c8;
    --text:#1c1208;--muted:#7a6a52;--accent:#8b5e1a;--accent2:#c4831e;
  }
  *{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;}
  a{color:var(--accent);}

  /* ── Page wrapper ── */
  .page{max-width:1100px;margin:0 auto;padding:48px 40px;}
  @media(max-width:700px){.page{padding:24px 16px;}}

  /* ── Title block ── */
  .title-block{border-bottom:2px solid var(--text);padding-bottom:20px;margin-bottom:8px;}
  .title-block h1{font-size:1.9rem;font-weight:700;letter-spacing:-.01em;line-height:1.2;color:var(--text);}
  .title-block h1 em{font-style:normal;color:var(--accent2);}
  .byline{display:flex;flex-wrap:wrap;gap:0;margin-top:12px;font-size:.82rem;color:var(--muted);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;border-top:1px solid var(--border);padding-top:8px;}
  .byline span{margin-right:24px;}
  .byline strong{color:var(--text);}
  .run-mono{font-family:'SFMono Regular','Consolas','Liberation Mono',monospace;font-size:.75rem;background:var(--surface2);border:1px solid var(--border);padding:2px 8px;border-radius:3px;color:var(--muted);}

  /* ── Section heading ── */
  .section{margin-bottom:44px;}
  .sh{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin-bottom:14px;display:flex;align-items:center;gap:10px;font-family:'SFMono Regular','Consolas','Liberation Mono',monospace;}
  .sh::after{content:'';flex:1;height:1px;background:var(--border);}
  .tag{font-size:.62rem;padding:2px 7px;border:1px solid currentColor;border-radius:2px;font-weight:700;letter-spacing:.05em;font-family:'SFMono Regular','Consolas','Liberation Mono',monospace;}
  .tag-purple{color:var(--purple);border-color:var(--purple-bg);background:var(--purple-bg);}

  /* ── Abstract box (summary stats) ── */
  .abstract{background:var(--surface2);border-left:3px solid var(--accent2);padding:14px 20px;margin-bottom:20px;font-size:.85rem;line-height:1.7;font-style:italic;color:var(--muted);}
  .abstract strong{font-style:normal;color:var(--text);}

  /* ── Score bars ── */
  .score-table{width:100%;border-collapse:collapse;background:var(--surface);border:1px solid var(--border);}
  .score-table tr{border-bottom:1px solid var(--border);}
  .score-table tr:last-child{border-bottom:none;}
  .score-table td{padding:10px 14px;vertical-align:middle;}
  .sn{font-size:.84rem;width:200px;color:var(--text);white-space:nowrap;}
  .st{width:100%;}
  .st-track{background:var(--surface2);height:16px;border:1px solid var(--border);border-radius:2px;overflow:hidden;}
  .st-fill{height:100%;border-radius:1px;transition:width .5s ease;}
  .st-fill.high{background:#4a7c59;}
  .st-fill.mid{background:#b5770d;}
  .st-fill.low{background:#b91c1c;}
  .sv{width:52px;text-align:right;font-size:.84rem;font-weight:700;font-family:'SFMono Regular','Consolas','Liberation Mono',monospace;white-space:nowrap;}
  .sv.high{color:var(--green);}
  .sv.mid{color:var(--yellow);}
  .sv.low{color:var(--red);}

  /* ── Two-col ── */
  .two-col{display:grid;grid-template-columns:1fr 1fr;gap:28px;align-items:stretch;}
  .two-col > div{display:flex;flex-direction:column;}
  @media(max-width:800px){.two-col{grid-template-columns:1fr;}}

  /* ── Data cards ── */
  .data-card{background:var(--surface);border:1px solid var(--border);padding:20px;flex:1;}
  .data-card-title{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:14px;font-family:'SFMono Regular','Consolas','Liberation Mono',monospace;}

  /* ── Latency ── */
  .lat-row{display:flex;align-items:center;gap:10px;margin-bottom:7px;font-size:.82rem;}
  .lat-key{width:38px;text-align:right;color:var(--muted);font-family:'SFMono Regular','Consolas','Liberation Mono',monospace;font-size:.78rem;flex-shrink:0;}
  .lat-track{flex:1;background:var(--surface2);height:12px;border:1px solid var(--border);border-radius:1px;overflow:hidden;}
  .lat-fill{height:100%;background:#8b5e1a;opacity:.7;}
  .lat-val{width:68px;text-align:right;color:var(--muted);font-size:.76rem;font-family:'SFMono Regular','Consolas','Liberation Mono',monospace;flex-shrink:0;}

  /* ── Cost ── */
  .cost-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px;}
  .cost-cell{background:var(--surface2);border:1px solid var(--border);padding:12px;text-align:center;}
  .cost-num{font-size:1.15rem;font-weight:700;color:var(--accent);font-family:'SFMono Regular','Consolas','Liberation Mono',monospace;}
  .cost-lbl{font-size:.65rem;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-top:3px;font-family:'SFMono Regular','Consolas','Liberation Mono',monospace;}
  .cost-total-box{background:var(--surface2);border:1px solid var(--accent2);padding:14px;text-align:center;}
  .cost-total-num{font-size:1.6rem;font-weight:700;color:var(--accent2);font-family:'SFMono Regular','Consolas','Liberation Mono',monospace;}
  .model-tag{display:inline-block;margin-top:10px;font-family:'SFMono Regular','Consolas','Liberation Mono',monospace;font-size:.72rem;color:var(--muted);background:var(--surface2);border:1px solid var(--border);padding:3px 8px;}

  /* ── Per-sample table ── */
  .table-wrap{background:var(--surface);border:1px solid var(--border);overflow:hidden;}
  .table-toolbar{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid var(--border);gap:12px;flex-wrap:wrap;background:var(--surface2);}
  .search-box{background:var(--surface);border:1px solid var(--border);padding:6px 11px;color:var(--text);font-size:.82rem;width:240px;outline:none;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;}
  .search-box:focus{border-color:var(--accent2);}
  .search-box::placeholder{color:var(--muted);}
  .row-count{font-size:.75rem;color:var(--muted);font-family:'SFMono Regular','Consolas','Liberation Mono',monospace;}
  table{width:100%;border-collapse:collapse;}
  thead th{background:var(--surface2);font-size:.64rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);padding:10px 12px;text-align:left;border-bottom:2px solid var(--border);white-space:nowrap;cursor:pointer;user-select:none;font-family:'SFMono Regular','Consolas','Liberation Mono',monospace;}
  thead th:hover{color:var(--text);}
  thead th.sorted{color:var(--accent2);}
  tbody tr{border-bottom:1px solid var(--border);}
  tbody tr:last-child{border-bottom:none;}
  tbody tr:nth-child(even){background:var(--surface2);}
  tbody tr:hover{background:var(--border);}
  td{padding:10px 12px;font-size:.81rem;vertical-align:middle;}
  .q-cell{max-width:260px;}
  .q-summary{cursor:pointer;color:var(--text);font-size:.81rem;line-height:1.45;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;}
  .q-detail{display:none;margin-top:10px;font-size:.78rem;}
  .q-detail.open{display:block;}
  .q-detail-block{background:var(--bg);border-left:2px solid var(--accent2);padding:9px 12px;margin-bottom:7px;}
  .q-detail-block strong{font-size:.63rem;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);display:block;margin-bottom:4px;font-family:'SFMono Regular','Consolas','Liberation Mono',monospace;}
  .q-detail-block p{color:var(--text);line-height:1.55;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;}
  .pill{display:inline-flex;align-items:center;justify-content:center;border-radius:2px;padding:2px 7px;font-size:.72rem;font-weight:700;min-width:42px;font-family:'SFMono Regular','Consolas','Liberation Mono',monospace;border:1px solid transparent;}
  .pill-high{background:var(--green-bg);color:var(--green-text);border-color:#a7d7b5;}
  .pill-mid{background:var(--yellow-bg);color:var(--yellow-text);border-color:#f0cc82;}
  .pill-low{background:var(--red-bg);color:var(--red-text);border-color:#fca5a5;}
  .pill-na{background:var(--surface2);color:var(--muted);border-color:var(--border);}
  .lat-pill{font-size:.73rem;color:var(--muted);font-family:'SFMono Regular','Consolas','Liberation Mono',monospace;}
  .ctx-pill{display:inline-flex;align-items:center;justify-content:center;background:var(--blue-bg);color:var(--blue-text);border:1px solid #bfdbfe;border-radius:99px;width:20px;height:20px;font-size:.68rem;font-weight:700;font-family:'SFMono Regular','Consolas','Liberation Mono',monospace;}
  .idx{color:var(--muted);font-size:.72rem;font-family:'SFMono Regular','Consolas','Liberation Mono',monospace;}
  .no-results{padding:40px;text-align:center;color:var(--muted);font-size:.88rem;font-style:italic;}
  .chevron{display:inline-block;transition:transform .18s;margin-right:5px;font-size:.65rem;color:var(--muted);}
  .chevron.open{transform:rotate(90deg);}

  /* ── Footer ── */
  .footer{margin-top:56px;padding-top:16px;border-top:1px solid var(--border);font-size:.75rem;color:var(--muted);text-align:center;font-family:'SFMono Regular','Consolas','Liberation Mono',monospace;}
</style>
</head>
<body>
<div class="page">

<!-- ── Title block ── -->
<div class="title-block">
  <h1>{{ benchmark_name }}</h1>
</div>
<div class="byline">
  <span>{{ timestamp }}</span>
  <span>Tenant: <strong>{{ tenant_id }}{% if sub_tenant_id %} / {{ sub_tenant_id }}{% endif %}</strong></span>
  <span>Endpoint: <strong>{{ endpoint }}</strong></span>
  <span class="run-mono">{{ run_id }}</span>
</div>

<!-- ── Abstract ── -->
<div style="margin-top:24px;margin-bottom:40px;">
  <div class="abstract">
    {% if ragas_scores %}
    {% set avg = (ragas_scores.values()|list|sum / ragas_scores.values()|list|length * 100)|round(1) %}
    Evaluated <strong>{{ evaluated_count }}</strong> samples across <strong>{{ ragas_scores|length }}</strong> metrics.
    Mean aggregate score: <strong>{{ avg }}%</strong>.
    {% if error_count > 0 %}<strong>{{ error_count }}</strong> sample(s) returned errors and were excluded from scoring.{% endif %}
    {% if token_usage.total_tokens > 0 %}
    Total tokens consumed by RAGAS judge: <strong>{{ "{:,}".format(token_usage.total_tokens) }}</strong>
    (est. cost <strong>${{ "%.4f" | format(token_usage.estimated_cost_usd) }}</strong>).
    {% endif %}
    {% else %}
    No evaluation scores available for this run.
    {% endif %}
  </div>
</div>

<!-- ── RAGAS Scores ── -->
{% if ragas_scores %}
<div class="section">
  <div class="sh">RAGAS Evaluation Scores</div>
  <table class="score-table">
    {% for name, score in ragas_scores.items() %}
    {% set pct = (score * 100) | round(1) %}
    {% if score >= 0.8 %}{% set cls="high" %}{% elif score >= 0.5 %}{% set cls="mid" %}{% else %}{% set cls="low" %}{% endif %}
    <tr>
      <td class="sn">{{ name | replace("_"," ") | title }}</td>
      <td class="st"><div class="st-track"><div class="st-fill {{ cls }}" style="width:{{ pct }}%"></div></div></td>
      <td class="sv {{ cls }}">{{ pct }}%</td>
    </tr>
    {% endfor %}
  </table>
</div>
{% endif %}

{% if multi_turn_scores %}
<div class="section">
  <div class="sh">Multi-Turn Scores <span class="tag tag-purple" style="margin-left:8px">MULTI-TURN</span></div>
  <table class="score-table">
    {% for name, score in multi_turn_scores.items() %}
    {% set pct = (score * 100) | round(1) %}
    {% if score >= 0.8 %}{% set cls="high" %}{% elif score >= 0.5 %}{% set cls="mid" %}{% else %}{% set cls="low" %}{% endif %}
    <tr>
      <td class="sn">{{ name | replace("_"," ") | title }}</td>
      <td class="st"><div class="st-track"><div class="st-fill {{ cls }}" style="width:{{ pct }}%"></div></div></td>
      <td class="sv {{ cls }}">{{ pct }}%</td>
    </tr>
    {% endfor %}
  </table>
</div>
{% endif %}

<!-- ── Latency + Cost ── -->
<div class="two-col section">

<div>
  <div class="sh">Query Latency</div>
  <div class="data-card">
    {% if latency_stats %}
    {% for key, val in latency_stats.items() %}
    {% set bar_w = ((val / max_latency) * 100) | round(1) %}
    <div class="lat-row">
      <span class="lat-key">{{ key }}</span>
      <div class="lat-track"><div class="lat-fill" style="width:{{ bar_w }}%"></div></div>
      <span class="lat-val">{{ val | round(0) | int }} ms</span>
    </div>
    {% endfor %}
    {% else %}
    <p style="color:var(--muted);font-size:.84rem;font-style:italic">No latency data available.</p>
    {% endif %}
  </div>
</div>

<div>
  <div class="sh">Token Usage & Cost</div>
  <div class="data-card">
    {% if token_usage.total_tokens > 0 %}
    <div class="cost-grid">
      <div class="cost-cell">
        <div class="cost-num">{{ "{:,}".format(token_usage.input_tokens) }}</div>
        <div class="cost-lbl">Input Tokens</div>
      </div>
      <div class="cost-cell">
        <div class="cost-num">{{ "{:,}".format(token_usage.output_tokens) }}</div>
        <div class="cost-lbl">Output Tokens</div>
      </div>
    </div>
    <div class="cost-total-box">
      <div class="cost-lbl">Estimated Cost (USD)</div>
      <div class="cost-total-num">${{ "%.4f" | format(token_usage.estimated_cost_usd) }}</div>
      <div style="font-size:.72rem;color:var(--muted);margin-top:3px;font-family:'SFMono Regular','Consolas','Liberation Mono',monospace">{{ "{:,}".format(token_usage.total_tokens) }} total tokens</div>
    </div>
    <div style="text-align:center"><div class="model-tag">{{ token_usage.model or "—" }}</div></div>
    {% else %}
    <p style="color:var(--muted);font-size:.84rem;font-style:italic">Cost tracking disabled or no usage data.</p>
    {% endif %}
  </div>
</div>

</div>

{% if per_sample_scores %}
<!-- ── Per-sample table ── -->
<div class="section">
  <div class="sh">Per-Sample Results</div>
  <div class="table-wrap">
    <div class="table-toolbar">
      <input class="search-box" type="text" id="tableSearch" placeholder="Search questions…" oninput="filterTable()">
      <div class="row-count" id="rowCount"></div>
    </div>
    <div style="overflow-x:auto">
    <table id="resultsTable">
      <thead>
        <tr>
          <th onclick="sortTable(0)" style="width:36px">#</th>
          <th style="min-width:200px">Question</th>
          {% for col in metric_cols %}
          <th onclick="sortTable({{ loop.index + 1 }})">{{ col | replace("_"," ") | title }}</th>
          {% endfor %}
          <th onclick="sortTable({{ metric_cols|length + 2 }})">Latency</th>
          <th>Ctx</th>
        </tr>
      </thead>
      <tbody id="tableBody">
      {% for row in per_sample_scores %}
      <tr>
        <td><span class="idx">{{ loop.index }}</span></td>
        <td class="q-cell">
          <div class="q-summary" onclick="toggleDetail(this)">
            <span class="chevron">▶</span>{{ row.get("question","") | truncate(90) }}
          </div>
          <div class="q-detail">
            <div class="q-detail-block">
              <strong>HydraDB Answer</strong>
              <p>{{ row.get("hydra_answer","—") | truncate(500) }}</p>
            </div>
            <div class="q-detail-block">
              <strong>Reference Answer</strong>
              <p>{{ row.get("reference_answer","—") | truncate(500) }}</p>
            </div>
          </div>
        </td>
        {% for col in metric_cols %}
        {% set val = row.get(col) %}
        {% if val is not none %}
          {% if val >= 0.8 %}{% set p="pill-high" %}{% elif val >= 0.5 %}{% set p="pill-mid" %}{% else %}{% set p="pill-low" %}{% endif %}
          <td><span class="pill {{ p }}">{{ (val * 100) | round(0) | int }}%</span></td>
        {% else %}
          <td><span class="pill pill-na">N/A</span></td>
        {% endif %}
        {% endfor %}
        <td><span class="lat-pill">{{ row.get("latency_ms",0) | round(0) | int }}ms</span></td>
        <td><span class="ctx-pill">{{ row.get("contexts_retrieved",0) }}</span></td>
      </tr>
      {% endfor %}
      </tbody>
    </table>
    <div class="no-results" id="noResults" style="display:none">No matching questions found.</div>
    </div>
  </div>
</div>
{% endif %}

<div class="footer">HydraDB Benchmark Framework &nbsp;·&nbsp; {{ run_id }} &nbsp;·&nbsp; {{ timestamp }}</div>

</div><!-- /page -->

<script>
function toggleDetail(el) {
  const detail = el.nextElementSibling;
  const chevron = el.querySelector('.chevron');
  detail.classList.toggle('open');
  chevron.classList.toggle('open');
}

function filterTable() {
  const q = document.getElementById('tableSearch').value.toLowerCase();
  const rows = document.querySelectorAll('#tableBody tr');
  let visible = 0;
  rows.forEach(row => {
    const text = row.querySelector('.q-summary').textContent.toLowerCase();
    const show = text.includes(q);
    row.style.display = show ? '' : 'none';
    if (show) visible++;
  });
  document.getElementById('rowCount').textContent = visible + ' of ' + rows.length + ' rows';
  document.getElementById('noResults').style.display = visible === 0 ? 'block' : 'none';
}

function sortTable(col) {
  const tbody = document.getElementById('tableBody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const th = document.querySelectorAll('thead th')[col];
  const asc = th.dataset.sort !== 'asc';
  document.querySelectorAll('thead th').forEach(t => { t.classList.remove('sorted'); delete t.dataset.sort; });
  th.classList.add('sorted');
  th.dataset.sort = asc ? 'asc' : 'desc';
  rows.sort((a, b) => {
    const av = a.cells[col]?.textContent.trim().replace('%','').replace('ms','') || '';
    const bv = b.cells[col]?.textContent.trim().replace('%','').replace('ms','') || '';
    const an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
  });
  rows.forEach(r => tbody.appendChild(r));
}

window.addEventListener('DOMContentLoaded', () => {
  const rows = document.querySelectorAll('#tableBody tr');
  if (rows.length) document.getElementById('rowCount').textContent = rows.length + ' rows';
});
</script>
</body>
</html>
"""


def _compute_latency_stats(samples) -> dict[str, float]:
    latencies = [s.latency_ms for s in samples if not s.error and s.latency_ms > 0]
    if not latencies:
        return {}
    latencies.sort()
    n = len(latencies)
    return {
        "p50":  latencies[int(n * 0.50)],
        "p75":  latencies[int(n * 0.75)],
        "p95":  latencies[min(int(n * 0.95), n - 1)],
        "p99":  latencies[min(int(n * 0.99), n - 1)],
        "mean": statistics.mean(latencies),
        "min":  min(latencies),
        "max":  max(latencies),
    }


class BenchmarkReporter:
    """Generates JSON, CSV, and HTML reports from a BenchmarkResult."""

    def __init__(self, config: ReportingConfig) -> None:
        self.config = config

    def generate(self, result: BenchmarkResult, output_dir: str = "./reports") -> list[Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"bench_{result.run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        paths = []

        if "json" in self.config.formats:
            paths.append(self._write_json(result, out, stem))
        if "csv" in self.config.formats:
            paths.append(self._write_csv(result, out, stem))
        if "html" in self.config.formats:
            paths.append(self._write_html(result, out, stem))

        return paths

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def _write_json(self, result: BenchmarkResult, out: Path, stem: str) -> Path:
        path = out / f"{stem}.json"
        data: dict[str, Any] = {
            "run_id":          result.run_id,
            "timestamp":       result.timestamp,
            "benchmark_name":  result.benchmark_name,
            "ragas_scores":    result.ragas_scores,
            "multi_turn_scores": result.multi_turn_scores,
            "latency_stats":   result.latency_stats,
            "token_usage":     result.token_usage.model_dump(),
            "summary": {
                "total_samples":    len(result.samples),
                "evaluated_count":  result.evaluated_count,
                "error_count":      result.error_count,
                "multi_turn_conversations": len(result.multi_turn_samples),
            },
        }
        if self.config.include_per_sample_scores:
            data["per_sample_scores"] = result.per_sample_scores
        data["config_snapshot"] = result.config_snapshot

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("JSON report: %s", path)
        return path

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def _write_csv(self, result: BenchmarkResult, out: Path, stem: str) -> Path:
        path = out / f"{stem}.csv"
        rows = result.per_sample_scores
        if not rows:
            path.write_text("No data\n", encoding="utf-8")
            return path

        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            df.to_csv(path, index=False)
        except ImportError:
            if rows:
                headers = list(rows[0].keys())
                lines = [",".join(str(h) for h in headers)]
                for row in rows:
                    lines.append(",".join(str(row.get(h, "")) for h in headers))
                path.write_text("\n".join(lines), encoding="utf-8")

        logger.info("CSV report: %s", path)
        return path

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    def _write_html(self, result: BenchmarkResult, out: Path, stem: str) -> Path:
        path = out / f"{stem}.html"
        try:
            from jinja2 import Environment, Undefined
            env = Environment(undefined=Undefined)
            template = env.from_string(_HTML_TEMPLATE)

            skip = {"sample_id", "question", "reference_answer", "hydra_answer",
                    "latency_ms", "contexts_retrieved", "error"}
            metric_cols = (
                [k for k in result.per_sample_scores[0] if k not in skip]
                if result.per_sample_scores else []
            )
            max_latency = max(result.latency_stats.values(), default=1.0) or 1.0
            hydra_cfg = result.config_snapshot.get("hydradb", {})

            html = template.render(
                benchmark_name=result.benchmark_name,
                run_id=result.run_id,
                timestamp=result.timestamp,
                tenant_id=hydra_cfg.get("tenant_id", ""),
                sub_tenant_id=hydra_cfg.get("sub_tenant_id", ""),
                endpoint=result.config_snapshot.get("evaluation", {}).get("search_endpoint", ""),
                ragas_scores=result.ragas_scores,
                multi_turn_scores=result.multi_turn_scores,
                evaluated_count=result.evaluated_count,
                error_count=result.error_count,
                latency_stats=result.latency_stats,
                max_latency=max_latency,
                token_usage=result.token_usage,
                per_sample_scores=(
                    result.per_sample_scores if self.config.include_per_sample_scores else []
                ),
                metric_cols=metric_cols,
            )
            path.write_text(html, encoding="utf-8")
        except Exception as e:
            logger.warning("HTML report generation failed: %s", e)
            path.write_text(f"<pre>Report generation failed: {e}</pre>", encoding="utf-8")

        logger.info("HTML report: %s", path)
        return path
