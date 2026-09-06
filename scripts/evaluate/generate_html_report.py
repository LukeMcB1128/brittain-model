"""Compile universal benchmark JSON results into a standalone HTML leaderboard report.

Every number on the page (track list, task counts, table columns, summary stats)
is derived from the results JSON — nothing about the suite is hardcoded here.

Usage:
    python3 scripts/evaluate/generate_html_report.py \
        --input benchmarks/results/universal_benchmark_results.json \
        --output benchmarks/results/leaderboard.html

`universal_bench.py` calls `write_report()` directly so the leaderboard refreshes
on every benchmark run.
"""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

TRACK_LABELS = {"python": "Python", "javascript": "JavaScript", "typescript": "TypeScript",
                "json": "JSON", "prose": "Prose"}
TRACK_ABBREV = {"python": "Py", "javascript": "JS", "typescript": "TS", "json": "JSON", "prose": "Prose"}

CODE_TRACKS = ("python", "javascript", "typescript")
SCORED_TRACKS = CODE_TRACKS + ("json", "prose")
HARNESS_VERSION = 2  # keep in step with brittain.universal_bench.HARNESS_VERSION

DEFAULT_RESULTS_DIR = Path("benchmarks/results")
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "leaderboard.html"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HTML Report for Universal Benchmark")
    parser.add_argument("--input", default=None,
                        help="Path to JSON results file (defaults to latest report in benchmarks/results/)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help=f"Output HTML file path (default {DEFAULT_OUTPUT})")
    return parser.parse_args()


def looks_like_reports(data: Any) -> bool:
    """True if `data` is a benchmark report list (and not some other JSON in the dir)."""
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not data:
        return False
    return all(isinstance(r, dict) and "checkpoint" in r and "tracks" in r for r in data)


def load_reports(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    if not looks_like_reports(data):
        raise ValueError(f"{path} does not contain universal benchmark reports")
    return data


def find_latest_results_file() -> Path | None:
    """Newest JSON in benchmarks/results/ that actually parses as a report list."""
    if not DEFAULT_RESULTS_DIR.exists():
        return None
    files = sorted(DEFAULT_RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        try:
            if looks_like_reports(json.loads(path.read_text(encoding="utf-8"))):
                return path
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
    return None


# --------------------------------------------------------------------------- #
# Suite shape, derived from the data
# --------------------------------------------------------------------------- #

def collect_tracks(reports: Sequence[Dict[str, Any]]) -> List[str]:
    """Tracks present in the data, canonical ones first, then any unknown extras."""
    seen = {t for rep in reports for t in rep.get("tracks", {})}
    ordered = [t for t in SCORED_TRACKS if t in seen]
    ordered += sorted(t for t in seen if t not in SCORED_TRACKS)
    return ordered


def track_task_counts(reports: Sequence[Dict[str, Any]], tracks: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for track in tracks:
        counts[track] = max(
            (int(rep.get("tracks", {}).get(track, {}).get("tasks", 0)) for rep in reports),
            default=0,
        )
    return counts


def get_float(rep: Dict[str, Any], key: str) -> float:
    val = rep.get(key)
    if val is None:
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def fmt_bpb(val: float) -> str:
    return "—" if math.isnan(val) else f"{val:.3f}"


def calculate_composite_score(rep: Dict[str, Any], track: str = "all") -> float:
    """0-100 composite capability score for a model on a given track (or overall)."""
    tracks = rep.get("tracks", {})
    if track != "all":
        td = tracks.get(track, {})
        if not td:
            return 0.0
        collapse = td.get("repetition_collapse", 0.0) * 100.0
        if track in CODE_TRACKS:
            syntax = td.get("syntax_validity", 0.0) * 100.0
            p1 = td.get("pass@1", 0.0) * 100.0
            return max(0.0, (syntax * 0.4 + p1 * 0.6) - collapse * 0.2)
        if track == "json":
            syntax = td.get("syntax_validity", 0.0) * 100.0
            schema = td.get("schema_validity", 0.0) * 100.0
            return max(0.0, (syntax * 0.4 + schema * 0.6) - collapse * 0.2)
        if track == "prose":
            d1 = td.get("distinct_1", 0.0) * 100.0
            d2 = td.get("distinct_2", 0.0) * 100.0
            cls = td.get("sentence_closed_rate", 0.0) * 100.0
            return max(0.0, (d1 * 0.3 + d2 * 0.3 + cls * 0.4) - collapse * 0.5)
        # Unknown track: fall back to syntax validity minus collapse.
        return max(0.0, td.get("syntax_validity", 0.0) * 100.0 - collapse * 0.2)

    scores = [calculate_composite_score(rep, t) for t in tracks if t in SCORED_TRACKS]
    return sum(scores) / len(scores) if scores else 0.0


def render_svg_chart(reports: Sequence[Dict[str, Any]], track: str, metric: str) -> str:
    """Render an SVG horizontal bar chart for the given track and metric."""
    items = []
    for rep in reports:
        name = rep.get("checkpoint", "Unknown")
        tracks = rep.get("tracks", {})
        val = 0.0
        meta = ""

        if metric == "composite":
            val = calculate_composite_score(rep, track)
            meta = f"composite score: {val:.1f}/100"
        elif metric == "syntax":
            if track == "all":
                syn = [tracks[t].get("syntax_validity", 0.0) * 100.0 for t in tracks]
                val = sum(syn) / len(syn) if syn else 0.0
            else:
                val = tracks.get(track, {}).get("syntax_validity", 0.0) * 100.0
            meta = f"syntax validity: {val:.1f}%"
        elif metric == "pass1":
            if track == "all":
                p1 = [tracks[t].get("pass@1", 0.0) * 100.0 for t in tracks if "pass@1" in tracks[t]]
                val = sum(p1) / len(p1) if p1 else 0.0
            elif track == "json":
                val = tracks.get("json", {}).get("schema_validity", 0.0) * 100.0
            elif track == "prose":
                val = tracks.get("prose", {}).get("sentence_closed_rate", 0.0) * 100.0
            else:
                val = tracks.get(track, {}).get("pass@1", 0.0) * 100.0
            meta = f"pass / schema / closure: {val:.1f}%"
        elif metric == "bpb":
            bpb_c = get_float(rep, "bpb_code")
            bpb_p = get_float(rep, "bpb_prose")
            valid = [b for b in (bpb_c, bpb_p) if not math.isnan(b)]
            avg_bpb = sum(valid) / len(valid) if valid else float("nan")
            val = 0.0 if math.isnan(avg_bpb) else max(0.0, 100.0 - (avg_bpb * 40.0))
            meta = f"code BPB: {fmt_bpb(bpb_c)} · prose BPB: {fmt_bpb(bpb_p)}"

        items.append({"name": name, "val": round(val, 1), "meta": meta})

    items.sort(key=lambda x: x["val"], reverse=True)

    chart_width = 1140
    row_height = 54
    top_offset = 32
    bottom_offset = 20
    chart_height = max(180, top_offset + len(items) * row_height + bottom_offset)

    x_start = 360
    bar_max_width = 680

    svg = [f'<svg class="score-chart" viewBox="0 0 {chart_width} {chart_height}" role="img" aria-label="Scores">']

    for pct in (0, 25, 50, 75, 100):
        x = x_start + (pct / 100.0) * bar_max_width
        svg.append(f'<line x1="{x}" y1="{top_offset - 8}" x2="{x}" y2="{chart_height - bottom_offset}" class="grid"/>')
        svg.append(f'<text x="{x}" y="{top_offset - 16}" class="top-tick">{pct}</text>')

    y = top_offset + 10
    for it in items:
        val = it["val"]
        width = max(2.0, (val / 100.0) * bar_max_width)
        svg.append('<g>')
        svg.append(f'<title>{html.escape(it["name"])}: {html.escape(it["meta"])}</title>')
        svg.append(f'<text x="{x_start - 16}" y="{y + 4}" class="bar-label">{html.escape(it["name"])}</text>')
        svg.append(f'<text x="{x_start - 16}" y="{y + 18}" class="bar-meta">{html.escape(it["meta"])}</text>')
        svg.append(f'<rect x="{x_start}" y="{y - 6}" width="{width:.2f}" height="22" rx="4" class="bar"/>')
        svg.append(f'<text x="{x_start + width + 10:.1f}" y="{y + 4}" class="bar-score">{val:.1f}</text>')
        svg.append('</g>')
        y += row_height

    svg.append('</svg>')
    return "".join(svg)


def heat_cell(pct: float, display: str) -> str:
    """Heatmap cell shaded by `pct` (0-100) but labelled with `display`."""
    alpha = max(0.08, min(1.0, pct / 100.0))
    return f'<td class="heat" style="background:rgba(9,105,218,{alpha:.2f})"><b>{display}</b></td>'


def matrix_columns(tracks: Sequence[str]) -> List[tuple]:
    """(header, extractor) pairs for the performance matrix, one pair per metric."""
    def code_syn(track):
        return lambda td: (td.get(track, {}).get("syntax_validity", 0.0) * 100.0,
                           f"{td.get(track, {}).get('syntax_validity', 0.0) * 100.0:.0f}%")

    def code_p1(track):
        return lambda td: (td.get(track, {}).get("pass@1", 0.0) * 100.0,
                           f"{td.get(track, {}).get('pass@1', 0.0) * 100.0:.1f}%")

    def json_schema(td):
        v = td.get("json", {}).get("schema_validity", 0.0) * 100.0
        return v, f"{v:.0f}%"

    def prose_d1(td):
        v = td.get("prose", {}).get("distinct_1", 0.0)
        return v * 100.0, f"{v:.2f}"

    def prose_cls(td):
        v = td.get("prose", {}).get("sentence_closed_rate", 0.0) * 100.0
        return v, f"{v:.0f}%"

    cols: List[tuple] = []
    for track in tracks:
        label = TRACK_ABBREV.get(track, track[:4].upper())
        if track == "json":
            cols.append((f"{label} Syn", code_syn("json")))
            cols.append((f"{label} Sch", json_schema))
        elif track == "prose":
            cols.append((f"{label} D1", prose_d1))
            cols.append((f"{label} Cls", prose_cls))
        else:
            cols.append((f"{label} Syn", code_syn(track)))
            cols.append((f"{label} p@1", code_p1(track)))
    return cols


def generate_html(reports: List[Dict[str, Any]]) -> str:
    """Generate the full, responsive HTML document from the reports alone."""
    models_count = len(reports)
    tracks = collect_tracks(reports)
    task_counts = track_task_counts(reports, tracks)
    total_tasks = sum(task_counts.values())
    track_names = ", ".join(TRACK_LABELS.get(t, t) for t in tracks)

    stale = [r.get("checkpoint", "Unknown") for r in reports
             if int(r.get("harness_version", 1)) < HARNESS_VERSION]
    stale_banner = ""
    if stale:
        listed = ", ".join(html.escape(n) for n in stale)
        stale_banner = (
            '<div class="warn"><b>%d of %d rows were produced by an older harness '
            '(v&lt;%d) and are not comparable to the rest.</b> Re-run these checkpoints: '
            '<code>%s</code></div>' % (len(stale), len(reports), HARNESS_VERSION, listed)
        )

    best_model, best_score = "N/A", -1.0
    for r in reports:
        score = calculate_composite_score(r, "all")
        if score > best_score:
            best_score, best_model = score, r.get("checkpoint", "Unknown")

    chart_tracks = ["all"] + list(tracks)
    metrics = [
        ("composite", "composite capability score"),
        ("syntax", "syntax validity %"),
        ("pass1", "pass@1 / schema / closure %"),
        ("bpb", "BPB compression efficiency"),
    ]

    chart_views_html = [
        f'<div class="chart-view" data-view-key="{t}|{m_id}">{render_svg_chart(reports, t, m_id)}</div>'
        for t in chart_tracks
        for m_id, _ in metrics
    ]

    track_options = "".join(
        f'<option value="{html.escape(t)}">{html.escape(t)}</option>' for t in tracks
    )
    metric_options = "".join(
        f'<option value="{m_id}">{html.escape(label)}</option>' for m_id, label in metrics
    )
    version_badges = "".join(
        f'<span class="version-badge"><b>{html.escape(t)}</b>'
        f'<small>{task_counts[t]} task{"s" if task_counts[t] != 1 else ""}</small></span>'
        for t in tracks
    )

    # Matrix table
    columns = matrix_columns(tracks)
    matrix_head = "".join(f"<th>{html.escape(header)}</th>" for header, _ in columns)
    matrix_rows = []
    for r in reports:
        td = r.get("tracks", {})
        overall = calculate_composite_score(r, "all")
        style = r.get("prompt_style", "completion")
        version = int(r.get("harness_version", 1))
        style_badge = f'<span class="badge">{html.escape(style)}</span>'
        if version < HARNESS_VERSION:
            style_badge += ' <span class="badge stale">v%d</span>' % version
        cells = [f"<td>{style_badge}</td>", heat_cell(overall, f"{overall:.1f}%")]
        for _, extract in columns:
            pct, display = extract(td)
            cells.append(heat_cell(pct, display))
        matrix_rows.append(
            f'<tr><th class="left">{html.escape(r.get("checkpoint", "Unknown"))}</th>'
            + "".join(cells)
            + f'<td>{fmt_bpb(get_float(r, "bpb_code"))}</td>'
            + f'<td>{fmt_bpb(get_float(r, "bpb_prose"))}</td></tr>'
        )

    # Detailed track breakdown
    detail_rows = []
    for r in reports:
        name = r.get("checkpoint", "Unknown")
        params = f"{r.get('params', 0):,}" if r.get("params") else "—"
        ctx = str(r.get("block_size", "—"))
        td = r.get("tracks", {})
        for t_name in tracks:
            if t_name not in td:
                continue
            data = td[t_name]
            tasks_count = data.get("tasks", 0)
            p1 = f"{100 * data['pass@1']:.1f}%" if "pass@1" in data else "—"

            extras = []
            if "schema_validity" in data:
                extras.append(f"Schema: {100 * data['schema_validity']:.0f}%")
            if "distinct_1" in data:
                extras.append(f"D1: {data['distinct_1']:.2f}")
            if "sentence_closed_rate" in data:
                extras.append(f"Cls: {100 * data['sentence_closed_rate']:.0f}%")

            detail_rows.append(f"""<tr data-track="{html.escape(t_name)}">
                <td class="left"><b>{html.escape(name)}</b></td>
                <td>{params}</td>
                <td>{ctx}</td>
                <td><span class="badge">{html.escape(t_name)}</span></td>
                <td>{100 * data.get('syntax_validity', 0.0):.1f}%</td>
                <td>{p1}</td>
                <td>{data.get('solved_tasks', 0)}/{tasks_count}</td>
                <td>{100 * data.get('repetition_collapse', 0.0):.1f}%</td>
                <td>{100 * data.get('empty_rate', 0.0):.1f}%</td>
                <td>{html.escape(", ".join(extras)) or '—'}</td>
            </tr>""")

    # Task-level details
    task_rows = []
    for r in reports:
        name = r.get("checkpoint", "Unknown")
        for task in r.get("per_task", []):
            track = str(task.get("track", ""))
            task_rows.append(f"""<tr data-track="{html.escape(track)}">
                <td class="left">{html.escape(name)}</td>
                <td><span class="badge">{html.escape(track)}</span></td>
                <td class="left"><code>{html.escape(str(task.get('id', '')))}</code></td>
                <td>{task.get('correct', 0)}/{task.get('n', 0)}</td>
                <td><b>{100 * task.get('pass@1', 0.0):.1f}%</b></td>
            </tr>""")

    page_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>BRITTAIN Universal Benchmark Leaderboard</title>
  <style>
    :root {{
      --bg: #ffffff;
      --fg: #1f2328;
      --muted: #656d76;
      --card: #f6f8fa;
      --line: #d0d7de;
      --accent: #0969da;
      --accent2: #1a7f37;
      --badge: #8250df;
      --warn-bg: #fff8c5;
      --warn-line: #9a6700;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0d1117;
        --fg: #e6edf3;
        --muted: #8b949e;
        --card: #161b22;
        --line: #30363d;
        --accent: #58a6ff;
        --accent2: #3fb950;
        --badge: #a371f7;
        --warn-bg: #2d2a12;
        --warn-line: #d29922;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 30px;
      background: var(--bg);
      color: var(--fg);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    h1 {{ font-size: 24px; margin: 0; }}
    .muted, .sub {{ color: var(--muted); }}
    .sub {{ margin: 4px 0 18px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .summary .stat {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 14px 16px;
    }}
    .summary b {{
      display: block;
      font-size: 22px;
      overflow-wrap: anywhere;
    }}
    .version-strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 20px;
    }}
    .version-badge {{
      display: flex;
      align-items: baseline;
      gap: 7px;
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--card);
      font-size: 11px;
    }}
    .version-badge small {{ color: var(--muted); }}
    .filters {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px 18px;
      margin-bottom: 20px;
      align-items: center;
    }}
    .filters label {{
      display: flex;
      align-items: center;
      gap: 7px;
      font-weight: 600;
    }}
    select {{
      background: var(--card);
      color: var(--fg);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 10px;
      font-size: 13px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 20px;
      margin-bottom: 22px;
      overflow: auto;
    }}
    h2 {{ font-size: 16px; margin: 0 0 16px; }}
    .chart-view {{ display: none; }}
    .score-chart {{
      display: block;
      width: 100%;
      height: auto;
      min-width: 760px;
    }}
    .grid {{ stroke: var(--line); stroke-width: 1; }}
    .top-tick {{ fill: var(--muted); font-size: 11px; text-anchor: middle; }}
    .bar {{ fill: var(--accent); opacity: 0.92; }}
    .bar-label {{ fill: var(--fg); font-size: 13px; font-weight: 650; text-anchor: end; }}
    .bar-meta {{ fill: var(--muted); font-size: 10px; text-anchor: end; }}
    .bar-score {{ fill: var(--fg); font-size: 13px; font-weight: 750; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      white-space: nowrap;
      font-size: 12px;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: right;
    }}
    th {{
      color: var(--muted);
      position: sticky;
      top: 0;
      background: var(--card);
    }}
    td.left, th.left {{ text-align: left; }}
    .heatmap {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    .heat {{ min-width: 70px; text-align: center; }}
    .heat b {{ display: block; }}
    .badge {{
      display: inline-block;
      padding: 2px 7px;
      border-radius: 999px;
      background: color-mix(in srgb, var(--badge) 20%, transparent);
      color: var(--badge);
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .warn {{
      border: 1px solid var(--warn-line);
      background: var(--warn-bg);
      color: var(--fg);
      border-radius: 8px;
      padding: 12px 14px;
      margin: 14px 0 18px;
      font-size: 13px;
    }}
    .warn code {{ overflow-wrap: anywhere; }}
    .badge.stale {{
      background: color-mix(in srgb, var(--warn-line) 30%, transparent);
      color: var(--warn-line);
    }}
    details.archive > summary {{
      cursor: pointer;
      padding: 14px 18px;
      font-weight: 700;
    }}
    details.archive > .archive-body {{ padding: 0 18px 18px; }}
    @media (max-width: 760px) {{
      body {{ padding: 16px; }}
      .card {{ padding: 14px; }}
      .score-chart {{ min-width: 680px; }}
    }}
  </style>
</head>
<body>
  <h1>BRITTAIN Universal Benchmark Leaderboard</h1>
  {stale_banner}
  <p class="sub">{models_count} model configuration{"s" if models_count != 1 else ""} evaluated · {total_tasks} standardized task{"s" if total_tasks != 1 else ""} across {len(tracks)} generation track{"s" if len(tracks) != 1 else ""}</p>

  <div class="summary">
    <div class="stat"><b>{models_count}</b><span class="muted">Models Evaluated</span></div>
    <div class="stat"><b>{len(tracks)}</b><span class="muted">Tracks ({html.escape(track_names)})</span></div>
    <div class="stat"><b>{total_tasks}</b><span class="muted">Tasks in Universal Suite</span></div>
    <div class="stat"><b>{html.escape(best_model)}</b><span class="muted">Top Performer ({best_score:.1f})</span></div>
  </div>

  <div class="version-strip">
    {version_badges}
  </div>

  <div class="filters">
    <label>Track
      <select id="track">
        <option value="all">all tracks</option>
        {track_options}
      </select>
    </label>
    <label>Metric
      <select id="metric">
        {metric_options}
      </select>
    </label>
  </div>

  <div class="card">
    <h2 id="chart-heading">Composite capability score across models</h2>
    {"".join(chart_views_html)}
  </div>

  <div class="card">
    <h2>Task &amp; Track Performance Matrix</h2>
    <table class="heatmap">
      <thead>
        <tr>
          <th class="left">Model Checkpoint</th>
          <th>Prompt</th>
          <th>Overall</th>
          {matrix_head}
          <th>Code BPB</th>
          <th>Prose BPB</th>
        </tr>
      </thead>
      <tbody>
        {"".join(matrix_rows)}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Track Breakdown Details</h2>
    <table>
      <thead>
        <tr>
          <th class="left">Model</th>
          <th>Params</th>
          <th>Context</th>
          <th>Track</th>
          <th>Syntax %</th>
          <th>Pass@1</th>
          <th>Solved</th>
          <th>Collapse %</th>
          <th>Empty %</th>
          <th>Key Metrics</th>
        </tr>
      </thead>
      <tbody id="detail-tbody">
        {"".join(detail_rows)}
      </tbody>
    </table>
  </div>

  <details class="card archive">
    <summary>Individual Task Results ({len(task_rows)} records)</summary>
    <div class="archive-body">
      <p class="muted">Detailed completion records per individual task across tested checkpoints.</p>
      <table>
        <thead>
          <tr>
            <th class="left">Model</th>
            <th>Track</th>
            <th class="left">Task ID</th>
            <th>Correct / Samples</th>
            <th>Pass@1</th>
          </tr>
        </thead>
        <tbody id="task-tbody">
          {"".join(task_rows)}
        </tbody>
      </table>
    </div>
  </details>

  <script>
    const METRIC_LABELS = {json.dumps({m_id: label for m_id, label in metrics})};
    const trackSelect = document.getElementById('track');
    const metricSelect = document.getElementById('metric');
    const chartHeading = document.getElementById('chart-heading');

    function filter() {{
      const t = trackSelect.value;
      const m = metricSelect.value;
      const key = t + '|' + m;

      document.querySelectorAll('.chart-view').forEach(el => {{
        el.style.display = el.dataset.viewKey === key ? 'block' : 'none';
      }});

      const label = METRIC_LABELS[m] || m;
      chartHeading.textContent = label.charAt(0).toUpperCase() + label.slice(1) +
                                 (t === 'all' ? ' (all tracks)' : ' (' + t + ')');

      document.querySelectorAll('#detail-tbody tr, #task-tbody tr').forEach(tr => {{
        tr.style.display = (t === 'all' || tr.dataset.track === t) ? '' : 'none';
      }});
    }}

    trackSelect.onchange = filter;
    metricSelect.onchange = filter;
    filter();
  </script>
</body>
</html>
"""
    return page_html


def write_report(input_path: Path | str, output_path: Path | str = DEFAULT_OUTPUT) -> Path:
    """Render `input_path` (a results JSON) to `output_path` and return the output path."""
    reports = load_reports(Path(input_path))
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(generate_html(reports), encoding="utf-8")
    return out_path


def main() -> int:
    args = parse_args()
    input_file = Path(args.input) if args.input else find_latest_results_file()
    if not input_file or not input_file.exists():
        print(f"Error: No results JSON file found at {args.input or DEFAULT_RESULTS_DIR}/")
        return 1

    print(f"[*] Reading results from {input_file} ...")
    try:
        out_path = write_report(input_file, args.output)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}")
        return 1
    print(f"[✓] HTML report successfully generated at: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
