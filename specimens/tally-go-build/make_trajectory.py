"""Writes trajectory.json for tally-go-build (the expert's turns are authored here for JSON escaping)."""
import json
from pathlib import Path


def turn(analysis, plan, cmds, done=False):
    return {"analysis": analysis, "plan": plan,
            "commands": [{"keystrokes": k, "duration": d} for k, d in cmds], "task_complete": done}


traj = [
    turn("I'm at /app. The Go project in /app/tally does not build. I need to see its layout, the Makefile, go.mod and "
         "the VERSION file first.",
         "List the project files and read the Makefile, go.mod and VERSION.",
         [("find /app/tally -type f | sort\n", 0.1),
          ("cat /app/tally/Makefile /app/tally/go.mod /app/tally/VERSION\n", 0.1)]),
    turn("The module is github.com/acme/tally with the command in cmd/tally and one package in internal/agg. make build "
         "runs go build with -ldflags \"-X main.Version=$(VERSION)\" into bin/tally; VERSION is 1.4.0.",
         "cd into the project and run make build to see the failure.",
         [("cd /app/tally\n", 0.1), ("make build 2>&1 | tail -n 5\n", 10.0)]),
    turn("The build fails: cmd/tally/main.go:9 imports github.com/acme/tally/agg, which no module provides (lookup is "
         "disabled by GOPROXY=off). The package lives in internal/agg, so the import path is wrong.",
         "Confirm which files import the package and what the package directory holds.",
         [("grep -rn 'acme/tally' --include='*.go' .\n", 0.1), ("ls internal/agg\n", 0.1),
          ("head -n 3 internal/agg/agg.go\n", 0.1)]),
    turn("Only cmd/tally/main.go imports it, as github.com/acme/tally/agg; internal/agg holds agg.go (package agg) and "
         "agg_test.go.",
         "Fix the import path to github.com/acme/tally/internal/agg, rebuild, and check --version.",
         [("sed -i 's#\"github.com/acme/tally/agg\"#\"github.com/acme/tally/internal/agg\"#' cmd/tally/main.go\n", 0.1),
          ("make build 2>&1 | tail -n 3\n", 10.0), ("./bin/tally --version\n", 0.1)]),
    turn("It builds now, but ./bin/tally --version prints `tally dev` instead of 1.4.0, so the version is not being "
         "linked in. Before that, the unit tests.",
         "Run make test.",
         [("make test 2>&1 | tail -n 8\n", 10.0)]),
    turn("make test fails two tests: TestApostrophes gets [{quote 1} {it's 1} {don't 1}] and TestTopTies gets "
         "[{plum 3} {pear 2} {fig 2}] where apple should come before fig. Words with the same count come out in "
         "reverse alphabetical order.",
         "Read the sort in Top.",
         [("grep -n 'sort.Slice' -A 6 internal/agg/agg.go\n", 0.1)]),
    turn("The tie-breaker in Top is `return out[i].Word > out[j].Word`, which sorts equal counts in descending word "
         "order; the doc comment and the tests want alphabetical order.",
         "Flip the tie-breaker to `<` and rerun the tests.",
         [("sed -i 's/return out\\[i\\].Word > out\\[j\\].Word/return out[i].Word < out[j].Word/' internal/agg/agg.go\n",
           0.1), ("make test 2>&1 | tail -n 3\n", 10.0)]),
    turn("The tests pass now (ok github.com/acme/tally/internal/agg). Back to the version: the binary said `tally dev`.",
         "Compare the variable name in main.go with the -X flag in the Makefile.",
         [("grep -n 'version' cmd/tally/main.go\n", 0.1), ("grep -n 'LDFLAGS' Makefile\n", 0.1)]),
    turn("main.go declares `var version = \"dev\"` (lower case), but the Makefile links -X main.Version=..., and -X "
         "silently ignores a symbol that does not exist.",
         "Fix the Makefile to set main.version, rebuild, and check --version and a run on the example.",
         [("sed -i 's/-X main.Version=/-X main.version=/' Makefile\n", 0.1), ("make build 2>&1 | tail -n 2\n", 10.0),
          ("./bin/tally --version && ./bin/tally --top 3 /data/words.txt\n", 0.1)]),
    turn("`tally 1.4.0` now, and the example gives words=35 distinct=17 with the 5, then a 4 and river 4 (tied at 4, "
         "alphabetical).",
         "Install with make install (PREFIX=/usr/local) and run tally from the PATH in another directory.",
         [("make install PREFIX=/usr/local 2>&1 | tail -n 3\n", 10.0), ("cd /\n", 0.1),
          ("tally --version && tally --top 4 /data/words.txt\n", 0.1)]),
    turn("make install copied bin/tally to /usr/local/bin/tally, and from / `tally --version` prints tally 1.4.0 and "
         "`tally --top 4` lists the 5, a 4, river 4, road 4. The build, the tests, the version and the install are "
         "all fixed.",
         "The task is complete.", [], True),
]
Path(__file__).with_name("trajectory.json").write_text(json.dumps(traj, indent=1) + "\n", newline="\n")
