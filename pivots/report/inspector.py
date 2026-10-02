"""Pivot Inspector (Executed Pivots): a self-contained HTML page over an audit run (the demo URL).

    python -m pivots.report.inspector out/audit --out out/inspector.html

Reads rows.jsonl, pivots.json and summary.json written by pivots.audit.run and
embeds them, so the page is one static file that can be hosted anywhere.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("inspector_template.html")


def build(audit_dir: str, out: str, title_note: str = "") -> str:
    d = Path(audit_dir)
    rows = [json.loads(line) for line in open(d / "rows.jsonl", encoding="utf-8")]
    for r in rows:
        r.pop("self_agreement", None)
    prog = d / "progress.jsonl"
    if prog.exists():  # verified progress labels (pivots.audit.progress), joined by task/turn/action
        idx = {(p["task"], p["turn"], p["action"]): p for p in map(json.loads, open(prog, encoding="utf-8"))}
        for r in rows:
            p = idx.get((r["task"], r["turn"], r["action"]))
            if p:
                r["P"] = p["P"]
                r["checks"] = [p["checks_before"], p["checks_expert"], p["checks_after"]]
    data = {"rows": rows, "pivots": json.loads((d / "pivots.json").read_text(encoding="utf-8")),
            "summary": json.loads((d / "summary.json").read_text(encoding="utf-8")), "note": title_note}
    blob = json.dumps(data).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", blob)
    Path(out).write_text(document(html), encoding="utf-8", newline="\n")
    return out


def document(fragment: str) -> str:
    """Wrap the template (head content, then body) as a standalone page.

    Opened from disk, a page with no charset declaration is decoded by guesswork
    (Chromium read one build as GBK: every check mark and middle dot garbled), and
    without a doctype and viewport it renders in quirks mode at desktop width on phones.
    """
    head, sep, body = fragment.partition("</style>")
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"{head}{sep}\n</head>\n<body>\n{body}\n</body>\n</html>\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("audit_dir")
    ap.add_argument("--out", default="out/inspector.html")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    print(build(a.audit_dir, a.out, a.note))


if __name__ == "__main__":
    main()
