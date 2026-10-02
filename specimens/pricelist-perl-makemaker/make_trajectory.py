"""Writes trajectory.json for pricelist-perl-makemaker (the expert's turns are authored here for JSON escaping)."""
import json
from pathlib import Path


def turn(analysis, plan, cmds, done=False):
    return {"analysis": analysis, "plan": plan,
            "commands": [{"keystrokes": k, "duration": d} for k, d in cmds], "task_complete": done}


traj = [
    turn("I'm at /app. The Perl distribution in /app/Acme-Pricelist has to build, pass its tests and be installed so "
         "pricelist-report works from the PATH. First its layout, Makefile.PL and README.",
         "cd into the distribution, list its files and read Makefile.PL and the README.",
         [("cd /app/Acme-Pricelist\n", 0.1), ("find . -type f | sort\n", 0.1), ("cat Makefile.PL README\n", 0.1)]),
    turn("It is an ExtUtils::MakeMaker dist: lib/Acme/Pricelist.pm, bin/pricelist-report, t/01-parse.t and "
         "t/02-summary.t. Makefile.PL lists EXE_FILES => ['bin/pricelist_report'], with an underscore. The README wants "
         "items by line total (largest first, ties by sku), then TOTAL and AVG with two decimals.",
         "Generate the Makefile and build, to see where it breaks.",
         [("perl Makefile.PL && make 2>&1 | tail -n 5\n", 10.0)]),
    turn("perl Makefile.PL writes the Makefile, but make stops: No rule to make target 'bin/pricelist_report', needed "
         "by 'blib/script/pricelist_report'. The script is bin/pricelist-report.",
         "Fix EXE_FILES to the real file name, regenerate the Makefile, build, and run the tests.",
         [("sed -i \"s#'bin/pricelist_report'#'bin/pricelist-report'#\" Makefile.PL && grep -n EXE_FILES Makefile.PL\n",
           0.1), ("perl Makefile.PL > /dev/null && make 2>&1 | tail -n 2\n", 10.0),
          ("make test 2>&1 | tail -n 12\n", 10.0)]),
    turn("The build copies bin/pricelist-report into blib/script now. make test fails: t/02-summary.t "
         "fails subtest 1 of 3 (Result: FAIL, 1/7 subtests failed); t/01-parse.t is not in the failure summary.",
         "Read the first test of t/02-summary.t and the sort in summarize().",
         [("sed -n '6,7p' t/02-summary.t\n", 0.1), ("grep -n -A 4 '^sub summarize' lib/Acme/Pricelist.pm\n", 0.1)]),
    turn("Test 1 expects the skus in the order Y W Z X for totals 100, 10, 10 and 9.5. summarize() sorts with `$b->{total} cmp $a->{total}`: cmp compares strings, so 9.5 sorts above 100 and 10. "
         "Line totals need the numeric <=>.",
         "Switch the total comparison to <=>, keeping the sku tie-break, and rerun the tests.",
         [("sed -i 's/\\$b->{total} cmp \\$a->{total}/$b->{total} <=> $a->{total}/' lib/Acme/Pricelist.pm && "
           "grep -n 'sort {' lib/Acme/Pricelist.pm\n", 0.1), ("make test 2>&1 | tail -n 4\n", 10.0)]),
    turn("All tests pass (Files=2, Tests=7, Result: PASS). Before installing, the report itself against the README.",
         "Run the built script from blib on /data/prices.csv.",
         [("perl -Mblib blib/script/pricelist-report /data/prices.csv\n", 1.0)]),
    turn("The items come out in the right order (C300 99.99, A100 13.50, B200 12.50, E500 10.00, D400 9.60) and "
         "TOTAL 145.59, but the last line is `AVG 29`: format_report prints the mean with %d, and the README wants two "
         "decimals (29.12). The tests do not cover it.",
         "Print AVG with %.2f, rebuild and rerun the tests and the report.",
         [("sed -i 's/sprintf(\"AVG %d\", \\$s->{avg})/sprintf(\"AVG %.2f\", $s->{avg})/' lib/Acme/Pricelist.pm && "
           "grep -n 'AVG' lib/Acme/Pricelist.pm\n", 0.1), ("make test 2>&1 | tail -n 1\n", 10.0),
          ("perl -Mblib blib/script/pricelist-report /data/prices.csv | tail -n 2\n", 1.0)]),
    turn("Tests still pass, and the report ends with TOTAL 145.59 and AVG 29.12.",
         "Install it with make install, then run pricelist-report from the PATH in / and check which module it loads.",
         [("make install 2>&1 | tail -n 4\n", 10.0), ("cd /\n", 0.1),
          ("pricelist-report /data/prices.csv && perl -MAcme::Pricelist -e 'print \"$INC{q(Acme/Pricelist.pm)} "
           "$Acme::Pricelist::VERSION\\n\"'\n", 1.0)]),
    turn("make install put the module in /usr/local/share/perl/5.36.0/Acme/Pricelist.pm and the script in "
         "/usr/local/bin/pricelist-report. From /, pricelist-report prints the sorted items, TOTAL 145.59 and AVG "
         "29.12, and perl loads Acme::Pricelist 0.03 from the site library. The build, the sort, the AVG format and "
         "the install are fixed.",
         "The task is complete.", [], True),
]
Path(__file__).with_name("trajectory.json").write_text(json.dumps(traj, indent=1) + "\n", newline="\n")
