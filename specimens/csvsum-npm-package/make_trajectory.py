"""Writes trajectory.json for csvsum-npm-package (the expert's turns are authored here for JSON escaping)."""
import json
from pathlib import Path

T = "npm test 2>&1 | grep -E '^(not )?ok |^# (pass|fail) |is not defined'\n"
EDIT_PKG = ("node -e 'const fs = require(\"fs\"); const p = JSON.parse(fs.readFileSync(\"package.json\", \"utf8\"));"
            " p.main = \"src/index.js\"; p.bin = { csvsum: \"bin/csvsum.js\" }; p.files = [\"src\", \"bin\"];"
            " fs.writeFileSync(\"package.json\", JSON.stringify(p, null, 2) + \"\\n\");'\n")
FIX_SUM = r"""python3 - <<'EOF'
p = 'src/index.js'
s = open(p).read()
old = "    const v = row[column];\n    if (v === '') continue;\n"
new = "    if (row[column] === '') continue;\n    const v = Number(row[column]);\n"
assert s.count(old) == 1
open(p, 'w').write(s.replace(old, new))
EOF
"""


def turn(analysis, plan, cmds, done=False):
    return {"analysis": analysis, "plan": plan,
            "commands": [{"keystrokes": k, "duration": d} for k, d in cmds], "task_complete": done}


traj = [
    turn("I'm at /app. The csvsum npm package in /app/csvsum is broken in three ways (tests, require, packed CLI). "
         "I need to see its layout, package.json and README first.",
         "List the package files, then read package.json and the README.",
         [("ls -la /app/csvsum /app/csvsum/src /app/csvsum/bin /app/csvsum/test\n", 0.1),
          ("cat /app/csvsum/package.json\n", 0.1), ("cat /app/csvsum/README.md\n", 0.1)]),
    turn("package.json has several suspicious fields: \"type\": \"module\" while the sources use require and "
         "module.exports; \"main\" points to lib/index.js, but there is no lib/ directory (the code is in src/); \"bin\" "
         "points to bin/csvsum, but the file is bin/csvsum.js; and \"files\" only lists src, so bin/ would not be "
         "packed. Let me confirm with the test suite and a require.",
         "cd into the package, run npm test (filtered to the result lines), and try to require the package.",
         [("cd /app/csvsum\n", 0.1), (T, 3.0),
          ("node -e \"require('/app/csvsum')\" 2>&1 | grep -m1 Error\n", 1.0)]),
    turn("npm test fails at load time: `require is not defined in ES module scope`, because \"type\": \"module\" makes "
         "Node treat every .js file in the package as an ES module. require('/app/csvsum') fails with `Cannot find "
         "module '/app/csvsum/lib/index.js'`, which confirms the bad main entry.",
         "Remove the \"type\" field so the CommonJS sources load, and rerun the tests to reach the real test results.",
         [("sed -i '/\"type\": \"module\",/d' package.json\n", 0.1), ("grep -c '\"type\"' package.json\n", 0.1),
          (T, 3.0)]),
    turn("The suite now runs: 3 pass and 2 fail ('summarize adds numbers, not strings' and 'summarize skips empty "
         "cells and reads other columns'). The name of the first failing test points at summarize.",
         "Read summarize in src/index.js.",
         [("grep -n 'function summarize' -A 18 src/index.js\n", 0.1)]),
    turn("summarize adds the raw cell strings: `total += v` with v = row[column], a string, so 0 + '12.50' becomes "
         "'012.50' and the total is a concatenation. min and max happen to coerce. The empty-cell check must stay on "
         "the raw string (Number('') is 0).",
         "Skip empty cells on the raw string, then convert with Number() before adding. Rerun the tests.",
         [(FIX_SUM, 0.5), (T, 3.0)]),
    turn("All 5 tests pass now (# pass 5, # fail 0). Next is packaging: require and the packed CLI.",
         "Check what npm pack would ship and whether require works now.",
         [("npm pack --dry-run 2>&1 | grep -E 'notice [0-9.]+k?B '\n", 2.0),
          ("node -e \"require('/app/csvsum')\" 2>&1 | grep -m1 Error\n", 1.0)]),
    turn("The dry run ships only README.md, package.json and src/index.js: no bin/ directory, so the csvsum command "
         "would point at a missing file. require still fails on lib/index.js.",
         "Fix package.json with node (keeps valid JSON): main -> src/index.js, bin csvsum -> bin/csvsum.js, files -> "
         "src and bin. Then check the pack list and the exports.",
         [(EDIT_PKG, 0.5), ("npm pack --dry-run 2>&1 | grep -E 'notice [0-9.]+k?B '\n", 2.0),
          ("node -e \"console.log(Object.keys(require('/app/csvsum')).join(' '))\"\n", 1.0)]),
    turn("The pack list now has bin/csvsum.js and src/index.js, and require('/app/csvsum') exports parseCsv, summarize, "
         "format and summarizeFile.",
         "Install globally from /app/csvsum and run csvsum on the example from another directory.",
         [("npm install -g /app/csvsum 2>&1 | tail -n 2\n", 5.0), ("cd /\n", 0.1),
          ("csvsum /data/sales.csv\n", 1.0)]),
    turn("npm installed the package, but running csvsum fails with `use strict: command not found`: "
         "bin/csvsum.js has no shebang, so the kernel runs it as a shell script. npm copied the package into "
         "/usr/local/lib/node_modules/csvsum (install-links), so the fix has to go into /app/csvsum and be "
         "reinstalled.",
         "Add `#!/usr/bin/env node` as the first line of bin/csvsum.js, reinstall globally, and rerun on both columns.",
         [("sed -i '1i #!/usr/bin/env node' /app/csvsum/bin/csvsum.js && head -n 2 /app/csvsum/bin/csvsum.js\n", 0.1),
          ("npm install -g /app/csvsum 2>&1 | tail -n 2\n", 5.0),
          ("csvsum /data/sales.csv && csvsum /data/sales.csv discount\n", 1.0)]),
    turn("csvsum works from / now: count=5 total=567.74 mean=113.55 min=19.99 max=310.25 for amount (the empty "
         "amount of order 1004 is skipped) and count=4 total=22.00 for discount.",
         "Check the packed tarball on its own: pack /app/csvsum into /tmp, install the tarball into a scratch prefix, "
         "and run its csvsum.",
         [("cd /tmp && npm pack /app/csvsum 2>&1 | tail -n 1\n", 3.0),
          ("npm install -g --prefix /tmp/csvsum-check ./csvsum-0.3.0.tgz 2>&1 | tail -n 1\n", 5.0),
          ("/tmp/csvsum-check/bin/csvsum /data/sales.csv\n", 1.0)]),
    turn("The tarball installed into /tmp/csvsum-check gives the same output. Tests pass, require works, the packed "
         "CLI works, and the global csvsum works from any directory.",
         "The task is complete.", [], True),
]
Path(__file__).with_name("trajectory.json").write_text(json.dumps(traj, indent=1) + "\n", newline="\n")
