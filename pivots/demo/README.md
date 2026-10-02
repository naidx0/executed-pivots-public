# Executed Pivots live demo: try a step

A small web app where anyone can test the idea by hand. Pick a pivot (one of the 17 decisive turns across the
six specimen tasks). The page shows the terminal so far and the expert's next command batch. Write your own
batch, edit the expert's, or start from a stored student sample or a control, then press **Run in a fork**.

The server forks the world at that pivot with the overlay backend and runs your batch in the fork. It then shows
three scores side by side:

- **J**: the shipped string reward (`pivots.effect.jreward`, the port of NeMo Gym's string-only `terminus_judge`),
  with its similarity score.
- **X**: the executed effect (`pivots.effect.EffectJudge`), with the effect score, the state and output match, the
  penalties, the files the expert changed (and whether you changed them too), and any other files you changed.
- **P**: the verifier checks passing right after your step, next to the count after the expert's step and before
  either (the rule from `pivots.audit.progress`). You can switch P off to save a few seconds.

The page also shows your terminal next to the expert's. Three one-click examples at the top make the "keystroke is
not the effect" point. Each one runs live:

| Example | Pivot | Candidate | J | X | P |
|---|---|---|---|---|---|
| Different keystrokes, same effect | `ledger-git-revert`, turn 4 of 7 | student `GIT_EDITOR=true git revert cdf209b` | 0 (sim 0.70) | 1 (0.97) | 1 (12 of 12 checks) |
| One flag dropped | `billing-invoice-bugfix`, turn 4 of 7 | control: the teacher's `sed -i` without `-i` | 1 (sim 0.99) | 0 (0.04) | 0 (6, expert 9) |
| "Done" said too early | `billing-invoice-bugfix`, turn 4 of 7 | control: the teacher's keystrokes plus `task_complete: true` | 1 (sim 1.00) | 0 (premature_complete) | 0 (claimed done, task fails) |

Scores come from the same code paths the audit and the NeMo Gym server use. Nothing is mocked or looked up.
`stored.json` holds the student samples and the audit's labels, which the page shows as "In the audit" next to each
stored candidate. They are never used to score. On all 218 stored actions at the 17 decisive pivots (controls and
students), the live J, X and P match the v4 audit exactly.

## Run it locally (Linux or WSL2)

```bash
scripts/demo.sh                 # http://127.0.0.1:7860
PORT=8080 scripts/demo.sh
```

The script creates `.venv-demo` on first run and installs `.[demo]` (FastAPI, uvicorn, httpx). It then starts
`python -m pivots.demo`. Worlds live in `/tmp/xp-demo-$USER`, which is deleted on exit. Startup builds the six worlds
and prepares the expert runs in the background, which takes about a minute. A pivot opened before then is built on
demand in a few seconds.

To run it without the script: `pip install -e '.[demo]' && python -m pivots.demo --port 7860`. Settings are
`--timeout` (seconds per batch, default 10), `--concurrency` (sandbox runs at once, default 2), `--store`, `--no-warm`,
`--pivots all` (open every turn, not only the 17 decisive ones; keep the default in public), or the `XP_DEMO_*`
environment variables.

**What it needs:**
- Linux or WSL2, with unprivileged user namespaces and overlayfs;
- Python 3.11 or newer;
- `unshare`, `setpriv` and `prlimit` (util-linux);
- the tools the specimens use: `git`, `gcc`, `make`, `awk`, `tar` and `gzip`.

It runs as an **ordinary user**, never as root: it refuses to start as root (the security model below explains why).
It needs no Docker and no sudo. Memory use is about 55 MB RSS for the server, plus short-lived sandbox processes.

## Security model (user input is untrusted)

Every sandbox run goes through `pivots/demo/sandbox.py`, the teacher's runs included, so X still compares like with like:

- **Capabilities.** The batch runs as uid 0 of a user namespace. `setpriv` empties the bounding, inheritable and
  ambient sets, sets `no_new_privs`, and locks the securebits so uid 0 cannot regain capabilities on exec. As a
  result `chroot`, `mount` and raw device access fail inside the fork, which closes the classic chroot escape.
  The API tests check `CapEff: 0000000000000000`, and that `chroot` and `mount` are refused.
- **Network.** Each fork gets its own empty network namespace, so there is no network. The tests check this.
- **Host files.** Host files are visible read-only, as the base image. `/home`, `/root`, `/tmp` and `/mnt` are
  hidden or empty. Writes land in a per-run overlay layer that is thrown away.
- **Time.** Each batch has a timeout (10 s by default), enforced by the probe's `timeout` and by the step runner.
  Any single sandbox run is hard-capped at 60 s.
- **Resources.** `prlimit` sets 256 processes, 2 GiB of address space, 32 MiB per file written, 60 CPU seconds and
  no core files.
- **Output.** The server keeps only the first 256 KiB and the last 4 MiB of a run's output
  (`OverlayWorld.run(max_output=...)`), so a batch that prints gigabytes cannot exhaust its memory.
- **Input.** Request bodies are limited to 64 KiB, keystrokes to 8,000 characters and 200 lines, and a raw
  Terminus-2 action to 32,000 characters. NUL bytes are refused.
- **Load.** Two sandbox runs can go at once, with a queue of 6. Past that, or after a 45 s wait, the server
  answers 429 with `Retry-After`.
- **Scratch state.** The layers a visitor's run creates are garbage-collected whenever the server goes idle. Nothing
  a visitor types is stored or logged beyond uvicorn's access log line.

Why not root: under real root, uid 0 inside the fork maps to real root. Even without capabilities, it would then own
the host's root-owned files and device nodes. As an ordinary user, uid 0 inside the fork maps only to that user.

This is defence in depth on a backend that was built for correctness. It is not a hardened multi-tenant sandbox. For
a public URL, run it on a throwaway VM with nothing else on it (next section). Longer term, the Nebius Sandboxes
backend (microVMs, `cleave/world/contree.py`) is the right boundary.

## Host it

### Nebius VM (recommended)

1. A small CPU VM is enough: 4 vCPU and 8 to 16 GB, Ubuntu 24.04. Keep nothing else on it.
2. Ubuntu 24.04 restricts unprivileged user namespaces through AppArmor. Allow them:
   `echo kernel.apparmor_restrict_unprivileged_userns=0 | sudo tee /etc/sysctl.d/60-userns.conf && sudo sysctl --system`
3. Install the tools and create a user:
   `sudo apt-get install -y python3-venv git gcc make && sudo useradd -m demo`
4. As `demo`, clone the repo and run `scripts/demo.sh`. It listens on 127.0.0.1:7860.
5. Put TLS in front, for example Caddy: `caddy reverse-proxy --from <host name> --to 127.0.0.1:7860`. For
   reboots, a systemd unit can run `scripts/demo.sh` as `demo`.

### Docker

```bash
docker build -t executed-pivots-demo .
docker run --rm -p 7860:7860 \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined --security-opt systempaths=unconfined \
  -v xp-demo-store:/mnt/xp-store executed-pivots-demo
```

It needs no `--privileged` and no added capabilities, and the process is uid 10001. The three `--security-opt`
flags are needed because:
- Docker's seccomp profile blocks `unshare(CLONE_NEWUSER)` and `mount`;
- the docker-default AppArmor profile blocks overlay mounts;
- Docker's masked `/proc` paths make the kernel refuse a fresh procfs in the fork's pid namespace.

The volume is needed because overlayfs cannot use the container's own overlay root as an upper directory. The host
kernel must still allow unprivileged user namespaces (on Ubuntu, the sysctl in step 2 above). The image was
built and run end to end on Docker Desktop (WSL2 kernel) on 2026-09-24 with exactly these flags.

### Hugging Face Spaces

Not expected to work. Docker Spaces run with the default seccomp and AppArmor restrictions and do not accept
`--security-opt`, so the overlay backend cannot make its namespaces. The server checks this at startup and exits with
a clear message instead of serving a broken page. Use a VM, or host only the static Pivot Inspector on Spaces.

## API

| Method | Path | Body or result |
|---|---|---|
| GET | `/` | the page |
| GET | `/api/pivots` | decisive pivots, the one-click examples, input limits |
| GET | `/api/pivot/{task:turn}` | instruction, terminal so far, expert batch and terminal after it, candidates (controls rebuilt from the expert's batch, stored students, audit labels) |
| POST | `/api/run` | `{"pivot", "source"}` for a stored candidate as stored, or `{"pivot", "keystrokes", "task_complete"}` for an edit, or `{"pivot", "action"}` for raw Terminus-2 JSON. Add `"verify": false` to skip P. It returns J, X, P, both terminals and the scored text. |
| GET | `/api/health` | worlds built, active and waiting runs, limits |

An edited batch is scored as a Terminus-2 response with one command holding the keystrokes. A last line with no
newline gets one, as if Enter was pressed. J joins keystrokes, so splitting a batch into several commands would
not change J.

## Tests

`tests/test_demo_api.py` has 11 tests. They skip without FastAPI or overlay support. They cover:
- the page and the pivot list;
- the three examples, which give the expected J, X and P and agree with the stored audit labels;
- an edited batch;
- the timeout;
- the input limits (keystroke and action length, line count, body size, unknown pivot or source, NUL);
- no network and no capabilities;
- bounded output;
- the 429 when busy;
- that scratch layers are reclaimed.

They pass as root and as an ordinary user. The Playwright run that produced the screenshots is described in the
project notes (`nebius/demo/README.md`).

## Known limits

- The worlds are the six specimen tasks we wrote, not rebuilt Terminal-Pivot tasks. The stored students are Claude
  samples standing in for Nemotron Nano.
- The overlay backend uses the host's root filesystem as the base image. Tool versions (git, gcc, awk) come from the
  machine or container the demo runs on, so a batch's output can differ slightly from the audit machine's. On this
  container, all 218 stored actions reproduce the audit's labels exactly.
- The files your batch changed are derived from X's missing and extra lists, which cap at 20 paths each.
- The fork's `/dev` is the host's `/dev`, bind-mounted, and read-mostly for an ordinary user (for example,
  `/dev/kmsg` may be readable). The Docker image narrows this to Docker's minimal `/dev`.
- One process, one machine. The concurrency limit is global, not per visitor, and there is no rate limit per IP.
  Put a proxy with rate limiting in front for a busy public URL.
- Fonts load from Google Fonts. Offline, the page falls back to system fonts.
