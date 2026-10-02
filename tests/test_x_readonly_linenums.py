"""H35: line-number prefixes (grep -n, cat -n, nl) are stripped before read-only informative-token recall.

In H30 the expert's `grep -n` output at csvsum t3 was mostly line numbers, so a working student who printed the same
code with sed -n (no numbers) recalled 5 of 32 tokens and lost credit (hand-close:0, E=1).
"""
from pivots.effect import EffectJudge
from pivots.effect.probe import Effects
from pivots.effect.xreward import informative_recall, informative_tokens, strip_line_numbers

# csvsum t3: the expert's `grep -n 'function summarize' -A 18 src/index.js` and the student's `sed -n '20,40p'`
REF_OUT = "21:function summarize(text, column = 'amount') {\n22-  const { header, rows } = parseCsv(text);\n23-  if (!header.includes(column)) throw new Error(`no column ${column}`);\n24-  let total = 0;\n25-  let count = 0;\n26-  let min = Infinity;\n27-  let max = -Infinity;\n28-  for (const row of rows) {\n29-    const v = row[column];\n30-    if (v === '') continue;\n31-    total += v;\n32-    count += 1;\n33-    min = Math.min(min, v);\n34-    max = Math.max(max, v);\n35-  }\n36-  if (count === 0) return { count: 0, total: 0, mean: 0, min: 0, max: 0 };\n37-  return { count, total, mean: total / count, min, max };\n38-}\n39-\n--\n45:function summarizeFile(path, column) {\n46-  return summarize(fs.readFileSync(path, 'utf8'), column);\n47-}\n48-\n49-module.exports = { parseCsv, summarize, format, summarizeFile };\n"
CAND_OUT = "// count, total, mean, min and max of one column; empty cells are skipped\nfunction summarize(text, column = 'amount') {\n  const { header, rows } = parseCsv(text);\n  if (!header.includes(column)) throw new Error(`no column ${column}`);\n  let total = 0;\n  let count = 0;\n  let min = Infinity;\n  let max = -Infinity;\n  for (const row of rows) {\n    const v = row[column];\n    if (v === '') continue;\n    total += v;\n    count += 1;\n    min = Math.min(min, v);\n    max = Math.max(max, v);\n  }\n  if (count === 0) return { count: 0, total: 0, mean: 0, min: 0, max: 0 };\n  return { count, total, mean: total / count, min, max };\n}\n\nfunction format(s) {\n"


def eff(output):
    return Effects(output, 0, "/app", {}, set(), roots=["/app"])


def test_csvsum_t3_h24_student_measured_recall_stays_under_threshold():
    # H35 measured this row at 5 of 9 informative tokens once the line numbers are gone: X stays 0 (H38 kept the
    # change on other rows; this one is locked at what it measures, not at the credit H35 hoped for)
    c = EffectJudge(world=None)._compare(eff(REF_OUT), eff(CAND_OUT), set(), {}, {})
    rec = c["detail"]["readonly_recall"]
    assert rec["recall"] == 0.5556 and rec["n"] == 9
    assert c["reward"] < 0.75
    assert not any(t.isdigit() for t in rec["recalled"] + rec["missed"])


def test_line_numbers_are_not_informative_tokens():
    toks = informative_tokens(REF_OUT)
    assert "parseCsv" in toks and "fs.readFileSync" in toks
    assert not toks & {"21", "22", "39", "45", "49"}


def test_prefixes_of_grep_cat_n_and_nl_are_stripped():
    assert strip_line_numbers("21:foo bar\n22-  baz\n--\n45:qux\n") == "foo bar\n  baz\n--\nqux\n"
    assert strip_line_numbers("     1\tall: build\n     2\t\tgo build\n") == "all: build\n\tgo build\n"


def test_dates_and_plain_numbers_are_kept():
    text = "2024-09-24 ok 17 rows\n2024-09-25 ok 18 rows\n"
    assert strip_line_numbers(text) == text
    assert {"2024-09-24", "17"} <= informative_tokens(text)


def test_numbered_read_of_the_wrong_lines_still_misses():
    wrong = "".join(f"{i}:// filler line {i}\n" for i in range(21, 50))
    assert informative_recall(REF_OUT, wrong)["recall"] < 0.6


def test_off_switch_restores_h30():
    j = EffectJudge(world=None, readonly_linenums=False)
    assert j._compare(eff(REF_OUT), eff(CAND_OUT), set(), {}, {})["reward"] < 0.75
