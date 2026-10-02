"""Writes trajectory.json for tb-processing-pipeline (the expert's turns are authored here for JSON escaping)."""
import json
from pathlib import Path


def turn(analysis, plan, cmds, done=False):
    return {"analysis": analysis, "plan": plan,
            "commands": [{"keystrokes": k, "duration": d} for k, d in cmds], "task_complete": done}


traj = [
    turn("I'm at /app. The pipeline /app/run_pipeline.sh fails, and I have to fix the scripts and whatever they depend "
         "on so it runs end to end. First I'll look at the files, their permissions and the main script.",
         "List /app with permissions and print run_pipeline.sh.",
         [("ls -la /app\n", 0.1), ("cat /app/run_pipeline.sh; echo\n", 0.1)]),
    turn("run_pipeline.sh runs ./collect_data.sh, ./process_data.sh and ./generate_report.sh in order from the "
         "current directory and stops at the first failure. The permissions are wrong: run_pipeline.sh and "
         "process_data.sh are -rw-r--r-- (not executable), and collect_data.sh is --wx--x--x (not readable).",
         "cd to /app and run the pipeline directly, then through bash, to see how far each gets.",
         [("cd /app\n", 0.1), ("./run_pipeline.sh\n", 0.5), ("bash run_pipeline.sh\n", 0.5)]),
    turn("./run_pipeline.sh fails with Permission denied. Through bash, collect_data.sh succeeds (I am root, so the "
         "missing read bit does not stop me, but it would stop a normal user), and then ./process_data.sh fails "
         "with Permission denied. There may be more problems behind that one.",
         "Check the first line of every script with cat -A for bad shebangs and CRLF line endings, and check the "
         "output directory the scripts write to.",
         [("for f in *.sh; do printf '%s: ' \"$f\"; head -n 1 \"$f\" | cat -A; done\n", 0.1),
          ("grep -c $'\\r' process_data.sh\n", 0.1), ("ls -ld /data /data/output\n", 0.1)]),
    turn("process_data.sh has DOS line endings (its shebang reads #!/bin/bash^M$, and 12 of its lines end in a CR), "
         "so the kernel would look for an interpreter called `bash\\r`. generate_report.sh has the shebang "
         "#!/bin/nonexistent. /data/output has mode d--------- (000), which only root can write to.",
         "Fix everything: make all four scripts readable and executable (755), strip the CRs from process_data.sh, "
         "restore the #!/bin/bash shebang in generate_report.sh, and give /data/output mode 755.",
         [("chmod 755 run_pipeline.sh collect_data.sh process_data.sh generate_report.sh\n", 0.1),
          ("sed -i 's/\\r$//' process_data.sh\n", 0.1),
          ("sed -i '1s|^#!/bin/nonexistent$|#!/bin/bash|' generate_report.sh\n", 0.1),
          ("chmod 755 /data/output\n", 0.1),
          ("ls -l *.sh; ls -ld /data/output; head -n 1 generate_report.sh process_data.sh | cat -A\n", 0.1)]),
    turn("All four scripts are now -rwxr-xr-x, /data/output is drwxr-xr-x, and both shebangs read #!/bin/bash$ "
         "with no ^M.",
         "Run the pipeline the normal way and look at what it produced.",
         [("./run_pipeline.sh; echo \"exit=$?\"\n", 1.0),
          ("ls -l /data/output; cat /data/output/processed_data.txt\n", 0.1)]),
    turn("./run_pipeline.sh now runs every step (Data collected, Data processed, Report generated) and ends with "
         "`Pipeline completed successfully!` and exit=0. /data/output has raw_data.txt, processed_data.txt "
         "(`TEST DATA`) and final_report.txt.",
         "The task is complete.", [], True),
]
Path(__file__).with_name("trajectory.json").write_text(json.dumps(traj, indent=1) + "\n", newline="\n")
