"""Writes trajectory.json for tb-fix-git (the expert's turns are authored here for JSON escaping)."""
import json
from pathlib import Path

LOST = "28ee3d9"


def turn(analysis, plan, cmds, done=False):
    return {"analysis": analysis, "plan": plan,
            "commands": [{"keystrokes": k, "duration": d} for k, d in cmds], "task_complete": done}


traj = [
    turn("I'm in /app/personal-site. The user committed changes, then checked out master, and now the changes are "
         "gone. That usually means the commit was made on a detached HEAD and is no longer on any branch. First "
         "I'll look at the repository state.",
         "Check the status, the branches and the history of all refs.",
         [("git status\n", 0.1), ("git branch -a\n", 0.1), ("git log --oneline --graph --all\n", 0.1)]),
    turn("master is checked out with a clean tree, and it is the only branch. Its history has 4 commits (Initial "
         "site, Add publications, Update about, Update title), and no ref points at anything else, so the lost "
         "commit is not reachable from a branch. The reflog records every HEAD move, including commits made while "
         "detached.",
         "Read the reflog.", [("git reflog\n", 0.1)]),
    turn("The reflog shows the story: after `Update title`, HEAD moved to HEAD~1 (the Update about commit), a commit "
         f"`Move to Stanford` was made there as {LOST}, and then master was checked out again, which left it "
         "behind.",
         "Inspect that commit: what it changed and against which parent.",
         [(f"git show --stat {LOST}\n", 0.1), (f"git diff master {LOST} --stat\n", 0.1)]),
    turn(f"{LOST} (`Move to Stanford`) changes _includes/about.md and _layouts/default.html. Compared with master it "
         "differs only in those two files. Its parent is the Update about commit, so master's later `Update title` "
         "commit is not in it.",
         "Merge the lost commit into master.",
         [(f"git merge -m 'Merge lost Move to Stanford changes' {LOST}\n", 0.5)]),
    turn("The merge stopped with a conflict in _layouts/default.html: master's `Update title` and the lost commit "
         "both edited the same part of it. about.md is not reported, so it came over without a conflict.",
         "Look at the conflict hunk.", [("git diff\n", 0.1)]),
    turn("The only conflict is the <h3> headline: master has `PhD Candidate @ UW` and the lost commit has `Postdoc @ "
         "Stanford`. The lost commit is the user's latest change (the move to Stanford), so its version wins; "
         "master changed nothing else in this file.",
         "Take the lost commit's side of default.html, stage it, and conclude the merge.",
         [("git checkout --theirs _layouts/default.html\n", 0.1), ("git add _layouts/default.html\n", 0.1),
          ("git commit --no-edit\n", 0.5)]),
    turn("The merge commit is created.",
         "Check the history, the tree state, and that both files carry the Stanford changes with no conflict "
         "markers left.",
         [("git log --oneline --graph -6\n", 0.1), ("git status --short | wc -l\n", 0.1),
          ("grep -n -E 'Stanford|<<<<<<<|>>>>>>>' _includes/about.md _layouts/default.html\n", 0.1)]),
    turn("master now has a merge commit whose second parent is the recovered `Move to Stanford` commit. The working "
         "tree is clean (0 changed files), and about.md (`Postdoctoral Researcher at Stanford CS`) and "
         "default.html (`Postdoc @ Stanford`) have the user's changes with no conflict markers.",
         "The task is complete.", [], True),
]
Path(__file__).with_name("trajectory.json").write_text(json.dumps(traj, indent=1) + "\n", newline="\n")
