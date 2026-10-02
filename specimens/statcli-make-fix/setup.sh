mkdir -p /app/statcli/src /app/statcli/include /app/statcli/tests/cases
cd /app/statcli
printf 'CC = gcc\nCFLAGS = -Wall -O2 -Iinclude\nOBJS = build/main.o build/stats.o\n\nbin/statcli: $(OBJS)\n\t@mkdir -p bin\n\t$(CC) $(CFLAGS) -o $@ $(OBJS)\n\nbuild/%%.o: src/%%.c include/stats.h\n\t@mkdir -p build\n\t$(CC) $(CFLAGS) -c $< -o $@\n\ntest: bin/statcli\n\t./tests/run_tests.sh\n\nclean:\n\trm -rf build bin\n\n.PHONY: test clean\n' > Makefile
cat > include/stats.h <<'EOF'
#ifndef STATS_H
#define STATS_H
#include <stddef.h>

double stats_mean(const double *xs, size_t n);
double stats_median(const double *xs, size_t n);
double stats_stddev(const double *xs, size_t n);

#endif
EOF
cat > src/stats.c <<'EOF'
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "stats.h"

double stats_mean(const double *xs, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += xs[i];
    return n ? s / n : 0.0;
}

static int cmp_double(const void *a, const void *b) {
    return (int)(*(const double *)a - *(const double *)b);
}

double stats_median(const double *xs, size_t n) {
    if (n == 0) return 0.0;
    double *tmp = malloc(n * sizeof *tmp);
    memcpy(tmp, xs, n * sizeof *tmp);
    qsort(tmp, n, sizeof *tmp, cmp_double);
    double m = (n % 2) ? tmp[n / 2] : (tmp[n / 2 - 1] + tmp[n / 2]) / 2.0;
    free(tmp);
    return m;
}

double stats_stddev(const double *xs, size_t n) {
    if (n == 0) return 0.0;
    double mu = stats_mean(xs, n), acc = 0.0;
    for (size_t i = 0; i < n; i++) acc += (xs[i] - mu) * (xs[i] - mu);
    return sqrt(acc / n);
}
EOF
cat > src/main.c <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include "stats.h"

int main(void) {
    size_t cap = 64, n = 0;
    double *xs = malloc(cap * sizeof *xs), x;
    while (scanf("%lf", &x) == 1) {
        if (n == cap) xs = realloc(xs, (cap *= 2) * sizeof *xs);
        xs[n++] = x;
    }
    printf("count=%zu\n", n);
    printf("mean=%.3f\n", stats_mean(xs, n));
    printf("median=%.3f\n", stats_median(xs, n));
    printf("stddev=%.3f\n", stats_stddev(xs, n));
    free(xs);
    return 0;
}
EOF
cat > tests/run_tests.sh <<'EOF'
#!/bin/bash
# Runs every tests/cases/*.in through bin/statcli and diffs against the matching .out
cd "$(dirname "$0")/.."
fail=0
for in in tests/cases/*.in; do
    exp="${in%.in}.out"
    if diff "$exp" <(./bin/statcli < "$in") > /dev/null; then
        echo "PASS $(basename "$in" .in)"
    else
        echo "FAIL $(basename "$in" .in)"
        diff "$exp" <(./bin/statcli < "$in")
        fail=1
    fi
done
exit $fail
EOF
chmod +x tests/run_tests.sh
printf '1 2 3 4 5\n' > tests/cases/integers.in
printf 'count=5\nmean=3.000\nmedian=3.000\nstddev=1.414\n' > tests/cases/integers.out
printf '10 2 38 23 38 23 21\n' > tests/cases/unsorted.in
printf 'count=7\nmean=22.143\nmedian=23.000\nstddev=12.299\n' > tests/cases/unsorted.out
printf '3.7 3.2 3.9 3.1\n' > tests/cases/fractions.in
printf 'count=4\nmean=3.475\nmedian=3.450\nstddev=0.334\n' > tests/cases/fractions.out
cat > README.md <<'EOF'
statcli: summary statistics for numbers on stdin.

    make            # builds bin/statcli
    make test       # runs tests/cases/*.in and compares with *.out
    echo 1 2 3 | ./bin/statcli
EOF
