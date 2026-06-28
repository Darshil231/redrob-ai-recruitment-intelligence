"""HTML evaluation dashboard for ranking and submission runs."""

from __future__ import annotations

from collections import Counter
from html import escape
from pathlib import Path
from typing import Iterable, Mapping

from src.core.models import RankedCandidate


class EvaluationDashboard:
    """Writes a standalone run dashboard for operational review."""

    def export(
        self,
        results: Iterable[RankedCandidate],
        metrics: Mapping[str, object],
        path: str | Path,
        *,
        csv_generated: bool = False,
        validation_passed: bool = False,
    ) -> Path:
        ranked = list(results)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            self._html(
                ranked,
                metrics,
                csv_generated=csv_generated,
                validation_passed=validation_passed,
            ),
            encoding="utf-8",
        )
        return output_path

    def _html(
        self,
        ranked: list[RankedCandidate],
        metrics: Mapping[str, object],
        *,
        csv_generated: bool,
        validation_passed: bool,
    ) -> str:
        average_score = metrics.get("average_score", self._average_score(ranked))
        cards = [
            ("Candidates Parsed", metrics.get("candidates_parsed", 0)),
            ("Filtered", metrics.get("filtered", 0)),
            ("Rejected", metrics.get("rejected", 0)),
            ("Semantic Ranked", metrics.get("semantic_ranked", len(ranked))),
            ("Average Score", average_score),
            ("Execution Time", f"{metrics.get('execution_time_seconds', 0)}s"),
            ("Memory Usage", f"{metrics.get('memory_usage_mb', 0)} MB"),
            ("CSV Generated", self._status(csv_generated)),
            ("Validation Passed", self._status(validation_passed)),
        ]
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Evaluation Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026;
      --muted: #5d6b76;
      --line: #dbe3ea;
      --panel: #ffffff;
      --page: #f6f8fa;
      --accent: #146c94;
      --good: #13795b;
      --warn: #b45309;
    }}
    body {{
      margin: 0;
      background: var(--page);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1, h2 {{
      margin: 0;
      letter-spacing: 0;
    }}
    h1 {{
      font-size: 30px;
    }}
    h2 {{
      font-size: 18px;
      margin-bottom: 14px;
    }}
    .subtle {{
      color: var(--muted);
      margin: 6px 0 24px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}
    .card, section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .card {{
      padding: 16px;
      min-height: 82px;
    }}
    .label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .value {{
      font-size: 26px;
      font-weight: 700;
    }}
    .ok {{
      color: var(--good);
    }}
    .bad {{
      color: var(--warn);
    }}
    .sections {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 16px;
      margin-top: 18px;
    }}
    section {{
      padding: 18px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }}
    .bar {{
      background: #e9eef3;
      border-radius: 4px;
      height: 8px;
      overflow: hidden;
      min-width: 90px;
    }}
    .fill {{
      background: var(--accent);
      height: 100%;
    }}
    @media (max-width: 760px) {{
      .sections {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Evaluation Dashboard</h1>
    <p class="subtle">Run summary for candidate ranking and submission generation.</p>
    <div class="grid">
      {''.join(self._card(label, value) for label, value in cards)}
    </div>
    <div class="sections">
      <section>
        <h2>Top Skills</h2>
        {self._skills_table(ranked)}
      </section>
      <section>
        <h2>Behavior Distribution</h2>
        {self._behavior_table(ranked)}
      </section>
    </div>
  </main>
</body>
</html>
"""

    def _card(self, label: str, value: object) -> str:
        css_class = "value"
        if value == "Yes":
            css_class += " ok"
        elif value == "No":
            css_class += " bad"
        return (
            '<div class="card">'
            f'<div class="label">{escape(label)}</div>'
            f'<div class="{css_class}">{escape(str(value))}</div>'
            "</div>"
        )

    def _skills_table(self, ranked: list[RankedCandidate]) -> str:
        counts = Counter(skill for result in ranked for skill in result.matched_skills)
        if not counts:
            return "<p class=\"subtle\">No matched skills available.</p>"
        max_count = max(counts.values())
        rows = []
        for skill, count in counts.most_common(10):
            width = round((count / max_count) * 100)
            rows.append(
                "<tr>"
                f"<td>{escape(skill)}</td>"
                f"<td>{count}</td>"
                f'<td><div class="bar"><div class="fill" style="width: {width}%"></div></div></td>'
                "</tr>"
            )
        return "<table><thead><tr><th>Skill</th><th>Count</th><th>Share</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"

    def _behavior_table(self, ranked: list[RankedCandidate]) -> str:
        buckets = Counter(self._behavior_bucket(result.behavior_score) for result in ranked)
        labels = ["High", "Medium", "Low"]
        total = max(sum(buckets.values()), 1)
        rows = []
        for label in labels:
            count = buckets[label]
            width = round((count / total) * 100)
            rows.append(
                "<tr>"
                f"<td>{label}</td>"
                f"<td>{count}</td>"
                f'<td><div class="bar"><div class="fill" style="width: {width}%"></div></div></td>'
                "</tr>"
            )
        return "<table><thead><tr><th>Bucket</th><th>Candidates</th><th>Share</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"

    @staticmethod
    def _behavior_bucket(score: float) -> str:
        if score >= 0.75:
            return "High"
        if score >= 0.45:
            return "Medium"
        return "Low"

    @staticmethod
    def _average_score(ranked: list[RankedCandidate]) -> float:
        if not ranked:
            return 0.0
        return round(sum(result.final_score for result in ranked) / len(ranked), 3)

    @staticmethod
    def _status(value: bool) -> str:
        return "Yes" if value else "No"
