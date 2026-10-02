mkdir -p /app/tally/cmd/tally /app/tally/internal/agg /data
# the machine is offline: no module proxy, no checksum database
go env -w GOPROXY=off GOSUMDB=off GOFLAGS=-mod=mod
cat > /app/tally/go.mod <<'EOF'
module github.com/acme/tally

go 1.19
EOF
echo 1.4.0 > /app/tally/VERSION
cat > /app/tally/Makefile <<'EOF'
# tally: word-frequency CLI
VERSION := $(shell cat VERSION)
LDFLAGS := -X main.Version=$(VERSION)
PREFIX ?= /usr/local

.PHONY: build test install

build:
	go build -ldflags "$(LDFLAGS)" -o bin/tally ./cmd/tally

test:
	go test ./...

install: build
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 0755 bin/tally $(DESTDIR)$(PREFIX)/bin/tally
EOF
cat > /app/tally/cmd/tally/main.go <<'EOF'
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"

	"github.com/acme/tally/agg"
)

// version is set at link time from the VERSION file (see the Makefile).
var version = "dev"

func main() {
	top := flag.Int("top", 5, "number of words to print")
	showVersion := flag.Bool("version", false, "print the version and exit")
	flag.Parse()
	if *showVersion {
		fmt.Println("tally", version)
		return
	}
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: tally [--top N] FILE")
		os.Exit(2)
	}
	f, err := os.Open(flag.Arg(0))
	if err != nil {
		fmt.Fprintln(os.Stderr, "tally:", err)
		os.Exit(1)
	}
	defer f.Close()
	c := agg.New()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		c.AddLine(sc.Text())
	}
	if err := sc.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "tally:", err)
		os.Exit(1)
	}
	fmt.Printf("words=%d distinct=%d\n", c.Total(), c.Distinct())
	for _, e := range c.Top(*top) {
		fmt.Printf("%s %d\n", e.Word, e.Count)
	}
}
EOF
cat > /app/tally/internal/agg/agg.go <<'EOF'
// Package agg counts words.
package agg

import (
	"sort"
	"strings"
	"unicode"
)

// Entry is one word and how often it occurs.
type Entry struct {
	Word  string
	Count int
}

// Counter counts lower-cased words; a word is a run of letters and apostrophes.
type Counter struct {
	counts map[string]int
	total  int
}

// New returns an empty Counter.
func New() *Counter { return &Counter{counts: map[string]int{}} }

// AddLine counts the words of one line.
func (c *Counter) AddLine(line string) {
	split := func(r rune) bool { return !unicode.IsLetter(r) && r != '\'' }
	for _, w := range strings.FieldsFunc(strings.ToLower(line), split) {
		w = strings.Trim(w, "'")
		if w == "" {
			continue
		}
		c.counts[w]++
		c.total++
	}
}

// Total is the number of words counted.
func (c *Counter) Total() int { return c.total }

// Distinct is the number of different words.
func (c *Counter) Distinct() int { return len(c.counts) }

// Top returns the n most frequent words, most frequent first; words with the same count are in
// alphabetical order.
func (c *Counter) Top(n int) []Entry {
	out := make([]Entry, 0, len(c.counts))
	for w, k := range c.counts {
		out = append(out, Entry{w, k})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		return out[i].Word > out[j].Word
	})
	if n < len(out) {
		out = out[:n]
	}
	return out
}
EOF
cat > /app/tally/internal/agg/agg_test.go <<'EOF'
package agg

import (
	"reflect"
	"testing"
)

func TestCounts(t *testing.T) {
	c := New()
	c.AddLine("The cat, the HAT; the 'end'")
	c.AddLine("")
	if c.Total() != 6 || c.Distinct() != 4 {
		t.Fatalf("Total=%d Distinct=%d, want 6 and 4", c.Total(), c.Distinct())
	}
}

func TestApostrophes(t *testing.T) {
	c := New()
	c.AddLine("don't 'quote' it's")
	want := []Entry{{"don't", 1}, {"it's", 1}, {"quote", 1}}
	if got := c.Top(5); !reflect.DeepEqual(got, want) {
		t.Fatalf("Top(5) = %v, want %v", got, want)
	}
}

func TestTopTies(t *testing.T) {
	c := New()
	c.AddLine("pear apple fig apple pear fig kiwi plum plum plum")
	want := []Entry{{"plum", 3}, {"apple", 2}, {"fig", 2}}
	if got := c.Top(3); !reflect.DeepEqual(got, want) {
		t.Fatalf("Top(3) = %v, want %v", got, want)
	}
}
EOF
cat > /app/tally/README.md <<'EOF'
# tally

Word frequencies of a text file.

    tally [--top N] FILE     # prints words=W distinct=D, then the N most frequent words
    tally --version          # prints "tally <version>", the version comes from the VERSION file

Build with `make build` (writes bin/tally), run the unit tests with `make test`, and install with
`make install` (PREFIX defaults to /usr/local). Go 1.19, no third-party modules; the machine is offline.
EOF
cat > /data/words.txt <<'EOF'
The river runs to the sea, and the sea runs to the river's end.
A river is a road; a road is a river that stays.
Sea salt, river silt, road dust: the road remembers.
EOF
# warm the Go build cache for the standard library packages the project uses (offline, deterministic)
mkdir -p /tmp/gowarm && cd /tmp/gowarm
printf 'module warm\n\ngo 1.19\n' > go.mod
printf 'package main\n\nimport (\n\t"bufio"\n\t"flag"\n\t"fmt"\n\t"os"\n\t"reflect"\n\t"sort"\n\t"strings"\n\t"testing"\n\t"unicode"\n)\n\nvar _ = testing.Short\nvar _ = reflect.DeepEqual\n\nfunc main() { _ = bufio.NewScanner(os.Stdin); _ = flag.Int; sort.Ints(nil); _ = strings.ToLower; _ = unicode.IsLetter; fmt.Println() }\n' > main.go
printf 'package main\n\nimport "testing"\n\nfunc TestWarm(t *testing.T) {}\n' > main_test.go
go build -o /dev/null . && go test . > /dev/null
cd / && rm -rf /tmp/gowarm
