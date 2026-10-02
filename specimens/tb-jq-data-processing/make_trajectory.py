"""Writes trajectory.json for tb-jq-data-processing (the expert's turns are authored here for JSON escaping)."""
import json
from pathlib import Path

FILTER = ('[.[] | select(.status == "active") | {user_id: .id, username: .username, email: .email, '
          'last_login: (.last_login | split("T")[0]), role_count: (.roles | length), '
          'primary_role: (if (.roles | length) > 0 then .roles[0] else null end)}] | sort_by(.username)')
CMD = f"cat > /app/cmd.sh <<'EOF'\njq --indent 2 '{FILTER}' /app/users.json > /app/active_users_report.json\nEOF\n"
YAML = f"cat > /app/jq_args.yaml <<'EOF'\nfilter: '{FILTER}'\nargs:\n  - \"--indent 2\"\nEOF\n"
REBUILD = ("jq $(yq -r '.args[]' /app/jq_args.yaml) \"$(yq -r '.filter' /app/jq_args.yaml)\" /app/users.json "
           "> /tmp/from_yaml.json && cmp /tmp/from_yaml.json /app/active_users_report.json && echo same\n")


def turn(analysis, plan, cmds, done=False):
    return {"analysis": analysis, "plan": plan,
            "commands": [{"keystrokes": k, "duration": d} for k, d in cmds], "task_complete": done}


traj = [
    turn("I'm at /app. I have to turn /app/users.json into /app/active_users_report.json with jq only (active users, "
         "renamed fields, date-only last_login, role_count, primary_role or null, sorted by username, 2-space "
         "indent), save the exact jq command to /app/cmd.sh, and save the filter and arguments to /app/jq_args.yaml. "
         "First I'll look at the input and the tools.",
         "List /app, check jq and yq, and look at the size and first record of users.json.",
         [("ls -la /app\n", 0.1), ("jq --version; yq --version\n", 0.5),
          ("jq 'length' /app/users.json; jq '.[0]' /app/users.json\n", 0.2)]),
    turn("users.json holds 500 users. jq is 1.6, and yq is the Debian yq wrapper around jq (it reports version 0.0.0), which is what the YAML check will call. The first record (id 1, johnsonjoshua) is active, has last_login 2025-06-12T20:59:10.109812Z and 5 roles, and a .auto_gen_key.dat checksum file sits next to users.json, so the input must not be modified.",
         "Profile the input with jq: the status values, how many users have no roles, and the last_login formats.",
         [("jq -c 'group_by(.status) | map({status: .[0].status, n: length})' /app/users.json\n", 0.2),
          ("jq '[.[] | select(.roles | length == 0)] | length' /app/users.json\n", 0.2),
          ("jq -r '.[].last_login' /app/users.json | cut -c11- | sort | uniq -c | head -n 3\n", 0.2)]),
    turn("There are 354 active, 96 inactive and 50 pending users. 70 users have an empty roles array, so primary_role has to be an explicit null for the active ones. Every last_login has the shape YYYY-MM-DDT20:59:10.109812Z, so the date is the part before the T.",
         "Write the jq command to /app/cmd.sh first (so the saved command is exactly the one that runs), then run it "
         "and look at the start of the output.",
         [(CMD, 0.1), ("bash /app/cmd.sh\n", 0.5),
          ("jq 'length' /app/active_users_report.json; head -n 17 /app/active_users_report.json\n", 0.2)]),
    turn("cmd.sh ran, and the report has 354 entries, the same as the active count. It starts with abrown (user_id 319) and acosta, each with user_id, username, email, a date-only last_login, role_count and primary_role, indented 2 spaces per level.",
         "Write /app/jq_args.yaml with the same filter (single-quoted YAML, so the double quotes inside stay literal) "
         "and --indent 2 as the only argument, then rebuild the output from the YAML the way the task describes and "
         "compare it byte for byte.",
         [(YAML, 0.1), (REBUILD, 1.0)]),
    turn("The output rebuilt from jq_args.yaml with the task's yq command line is byte-identical to the report (cmp printed `same`).",
         "Check the remaining requirements on the output: sorted usernames, no inactive users, and nulls for empty "
         "roles.",
         [("jq '[.[].username] == ([.[].username] | sort)' /app/active_users_report.json\n", 0.2),
          ("jq --slurpfile out /app/active_users_report.json '[.[] | select(.status != \"active\") | .id] as $bad | "
           "[$out[0][] | select(.user_id as $i | $bad | index($i))] | length' /app/users.json\n", 0.2),
          ("jq '[.[] | select(.primary_role == null)] | length' /app/active_users_report.json\n", 0.2)]),
    turn("The usernames are in sorted order (true), no inactive or pending user's id is in the report (0), and 51 active users have no roles and carry primary_role null. The report, /app/cmd.sh and /app/jq_args.yaml are all in place, and users.json is untouched.", "The task is complete.", [], True),
]
Path(__file__).with_name("trajectory.json").write_text(json.dumps(traj, indent=1) + "\n", newline="\n")
