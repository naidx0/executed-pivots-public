"""Writes trajectory.json for tb-grid-pattern-transform (the expert's turns are authored here for JSON escaping)."""
import json
from pathlib import Path

EX = ("import numpy as np; x = np.array([[8, 6], [6, 4]]); "
      "y = np.array([[8,6,8,6,8,6],[6,4,6,4,6,4],[6,8,6,8,6,8],[4,6,4,6,4,6],[8,6,8,6,8,6],[6,4,6,4,6,4]]); ")
COMPARE = "python3 -c '" + EX + "t = np.tile(x, (3, 3)); print(t); print(\"rows equal:\", (t == y).all(axis=1))'\n"
BLOCKS = ("python3 -c '" + EX + "[print(i, j, y[2*i:2*i+2, 2*j:2*j+2].tolist()) for i in range(3) for j in range(3)]; "
          "print(\"fliplr:\", np.fliplr(x).tolist())'\n")
SOLVE = r'''cat > /app/grid_transform.py <<'EOF'
import numpy as np


def solve(input_grid):
    """
    Transform the input grid according to the pattern.

    Args:
        input_grid (List[List[int]]): 2D list of integers representing the input grid
            0: Black     1: Blue       2: Red        3: Green     4: Yellow
            5: Grey      6: Hot Pink   7: Orange     8: Light Blue 9: Maroon

    Returns:
        List[List[int]]: 2D list of integers representing the transformed grid
    """
    grid = np.array(input_grid)
    flipped = np.fliplr(grid)
    # a 3x3 arrangement of copies: block rows 0 and 2 repeat the grid, block row 1 repeats it mirrored left-right
    rows = [np.hstack([grid] * 3), np.hstack([flipped] * 3), np.hstack([grid] * 3)]
    return np.vstack(rows).astype(int).tolist()
EOF
'''
TRY = ("python3 -c 'import sys; sys.path.append(\"/app\"); from grid_transform import solve; import numpy as np; "
       "y = [[8,6,8,6,8,6],[6,4,6,4,6,4],[6,8,6,8,6,8],[4,6,4,6,4,6],[8,6,8,6,8,6],[6,4,6,4,6,4]]; "
       "r = solve([[8, 6], [6, 4]]); print(r == y, type(r).__name__, type(r[0][0]).__name__); "
       "print(np.array(solve([[1, 2], [3, 5]])))'\n")


def turn(analysis, plan, cmds, done=False):
    return {"analysis": analysis, "plan": plan,
            "commands": [{"keystrokes": k, "duration": d} for k, d in cmds], "task_complete": done}


traj = [
    turn("I'm at /app. I need to create /app/grid_transform.py with a solve(input_grid) that maps a 2x2 grid to a 6x6 "
         "grid, following the example: [[8,6],[6,4]] -> [[8,6,8,6,8,6],[6,4,6,4,6,4],[6,8,6,8,6,8],[4,6,4,6,4,6],"
         "[8,6,8,6,8,6],[6,4,6,4,6,4]]. First I'll check the directory and that numpy (imported by the template) is "
         "available.",
         "List /app and print the numpy version.",
         [("ls -la /app\n", 0.1), ("python3 -c 'import numpy; print(numpy.__version__)'\n", 1.0)]),
    turn("/app is empty, and numpy 2.3.1 is installed. The output is 3 times the input in each direction, so the "
         "first guess is plain tiling of the 2x2 grid.",
         "Compare np.tile(input, (3, 3)) with the example output row by row.",
         [(COMPARE, 1.0)]),
    turn("Plain tiling matches rows 0, 1, 4 and 5 but not rows 2 and 3 (rows equal: True True False False True "
         "True), so the middle band of blocks is transformed differently.",
         "Print each 2x2 block of the example output, and the input mirrored left-right, to identify that transform.",
         [(BLOCKS, 1.0)]),
    turn("All three blocks in block rows 0 and 2 are the input [[8, 6], [6, 4]], and all three blocks in block row 1 "
         "are [[6, 8], [4, 6]], which is exactly np.fliplr(input). The rule: block rows 0 and 2 repeat the grid three "
         "times, and block row 1 repeats its left-right mirror three times.",
         "Write /app/grid_transform.py with that rule, returning a plain list of Python ints, and check it on the "
         "example and on another input.",
         [(SOLVE, 0.1), ("cd /tmp\n", 0.1), (TRY, 1.0)]),
    turn("solve reproduces the example exactly (True), returns a list of ints, and on [[1, 2], [3, 5]] gives the "
         "same structure: rows 1 2 1 2 1 2 / 3 5 3 5 3 5, then the mirrored 2 1 2 1 2 1 / 5 3 5 3 5 3, then the "
         "original rows again.",
         "Confirm the module imports cleanly the way a test script would, from another directory with /app on "
         "sys.path.",
         [("cd / && python3 -c 'import sys; sys.path.append(\"/app\"); import grid_transform; "
           "print(grid_transform.solve([[7, 9], [4, 3]])[2])'\n", 1.0)]),
    turn("The module imports from / and the third row for [[7, 9], [4, 3]] is the mirrored first row, [9, 7, 9, 7, 9, 7]. "
         "/app/grid_transform.py implements the transformation.",
         "The task is complete.", [], True),
]
Path(__file__).with_name("trajectory.json").write_text(json.dumps(traj, indent=1) + "\n", newline="\n")
