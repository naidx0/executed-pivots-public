"""Writes trajectory.json for tb-cpp-compatibility (the expert's turns are authored here for JSON escaping)."""
import json
from pathlib import Path

TEST = r"""mkdir -p /tmp/sumcheck && cat > /tmp/sumcheck/check.cpp <<'EOF'
#include <array>
#include <cstdio>
#include "/app/sum_array.h"

int main()
{
    constexpr std::array<int, 5> a{{1, 2, 3, 4, 5}};
    constexpr std::array<int, 0> empty{};
    static_assert(sum_array(a) == 15, "sum of 1..5");
    static_assert(sum_array(empty) == 0, "empty sum");
    const std::array<double, 3> d{{0.5, 1.25, -2.0}};
    std::printf("%d %g\n", sum_array(a), sum_array(d));
}
EOF
"""
HEADER = r"""cat > /app/sum_array.h <<'EOF'
#include <array>
#include <cstddef>

// C++11 constexpr functions may only contain a single return statement, so the loop becomes recursion.
template <typename T, std::size_t N>
constexpr T sum_array_from(const std::array<T, N> &arr, std::size_t i)
{
    return i < N ? arr[i] + sum_array_from(arr, i + 1) : T(0);
}

template <typename T, std::size_t N>
constexpr T sum_array(const std::array<T, N> &arr)
{
    return sum_array_from(arr, 0);
}
EOF
"""


def turn(analysis, plan, cmds, done=False):
    return {"analysis": analysis, "plan": plan,
            "commands": [{"keystrokes": k, "duration": d} for k, d in cmds], "task_complete": done}


traj = [
    turn("I'm at /app. The template in /app/sum_array.h does not compile in C++11 projects, and I have to rewrite it "
         "to be C++11-compatible with the same interface. First I'll read the header and check the compiler.",
         "Print the header and the g++ version.",
         [("cat /app/sum_array.h\n", 0.1), ("g++ --version | head -n 1\n", 0.1)]),
    turn("sum_array is `template <typename T, std::size_t N> constexpr T sum_array(const std::array<T, N> &arr)` "
         "whose body declares a local and runs a for loop. That is a relaxed constexpr function, which needs C++14; "
         "C++11 allows only a single return statement in a constexpr function. The compiler is g++ 12.2 (Debian).",
         "Write a small check program that uses sum_array in constant expressions (static_assert, including an empty "
         "array) and at run time.",
         [(TEST, 0.1)]),
    turn("The check program is written to /tmp/sumcheck/check.cpp.",
         "Compile it with -std=c++11 to reproduce the failure, and with -std=c++14 for comparison.",
         [("cd /tmp/sumcheck\n", 0.1),
          ("g++ -std=c++11 check.cpp -o check11 2>&1 | grep -E 'error' | cut -c1-150\n", 2.0),
          ("g++ -std=c++14 check.cpp -o check14 && ./check14\n", 2.0)]),
    turn("With -std=c++11, both static_asserts fail (`non-constant condition for static assertion`): sum_array, for "
         "N = 5 and for N = 0, cannot be evaluated in a constant expression, because its body (a local variable and "
         "a loop) is not a valid C++11 constexpr function. With -std=c++14 it compiles, and the program prints 15 and "
         "-0.25. So the only problem is the C++14 function body.",
         "Rewrite sum_array.h for C++11: keep the sum_array signature and replace the loop with a recursive "
         "single-return helper using the ternary operator, which also handles N == 0. Then rebuild the check with "
         "-std=c++11.",
         [(HEADER, 0.1), ("g++ -std=c++11 -Wall -Wextra check.cpp -o check11 && ./check11\n", 2.0)]),
    turn("The check now compiles with -std=c++11 -Wall -Wextra without warnings, both static_asserts (1..5 and the "
         "empty array) hold at compile time, and it prints 15 -0.25 at run time. The interface is unchanged.",
         "Make sure the C++14 and C++17 builds still work too.",
         [("g++ -std=c++14 check.cpp -o check14 && g++ -std=c++17 check.cpp -o check17 && ./check17\n", 3.0)]),
    turn("The header compiles and gives the same results as C++11, C++14 and C++17. /app/sum_array.h is now "
         "C++11-compatible, with the same sum_array interface.",
         "The task is complete.", [], True),
]
Path(__file__).with_name("trajectory.json").write_text(json.dumps(traj, indent=1) + "\n", newline="\n")
