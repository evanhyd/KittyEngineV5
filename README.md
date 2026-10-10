# KittyEngineV5
A C++ bitboard chess engine with template-meta programming performance optimizations.

## Optional Syzygy tablebases

Syzygy support is **off by default**. Build from a Visual Studio developer shell:

```powershell
msbuild KittyEngineV5.sln /p:Configuration=Release /p:Platform=x64 /p:EnableSyzygy=true
```

Use `/p:EnableSyzygy=false` to compile it out completely. The shared
`KittyEngine.Build.props` controls both projects, including their object dependencies.
It selects a C++20 `constexpr` configuration header and the enabled-only source
files; there are no feature macros in the integration. The configuration uses
`<build_config.h>` so the enabled include directory can override the default
header. Fathom retains its upstream preprocessor code.
Default builds retain the original executable location, `x64/Release/` for
Release. Enabled builds write to `x64/Release/syzygy/`. Each project keeps its
intermediate files in separate `plain/` and `syzygy/` directories, so switching
does not require deleting intermediate files.
Debug and Release Profiler use the same switch. The supported build target for
this integration is x64. Install the project's v145 C++ tools and Windows SDK;
`/p:WindowsTargetPlatformVersion=...` can select another installed SDK. Tests
require GoogleTest, as before.

### Install the five-piece tablebases

From the repository root, run this once in Windows PowerShell 5.1 or PowerShell 7:

```powershell
.\tools\install-syzygy.ps1
```

This installs the complete standard **3–5-piece WDL (`.rtbw`) and DTZ (`.rtbz`)
collection**: 290 files totaling **938.4 MiB**, including the smaller tables
needed after captures and promotions. Piece counts include both kings. The
default folder is `Tablebases\syzygy` inside the checkout; it is ignored by Git.
To use another location, preferably on an SSD:

```powershell
.\tools\install-syzygy.ps1 -Destination 'D:\Chess Tables\syzygy'
```

The installer downloads from the [Lichess mirror](https://tablebase.lichess.ovh/tables/standard/)
and checks every file against the sizes and SHA-256 hashes pinned in
[`tools/syzygy-3-4-5.json`](tools/syzygy-3-4-5.json). The manifest records the
published checksum sources and their hashes. Downloads are sequential, with
up to three attempts per file. A file receives its final name only after
verification succeeds.

If installation is interrupted, rerun the same command. Verified files are
reused, damaged files are replaced, and unfinished files restart from the
beginning. A failed download leaves the previous destination file untouched.
An already complete installation can be verified again without network access
by rerunning the installer.

### Configure the engine

After installation, the script prints the UCI commands with the absolute path
to the installed tables. Set **SyzygyPath** to that folder in your chess GUI's
engine options, or send the printed commands directly to an enabled engine.
For example, after installing to `D:\Chess Tables\syzygy`:

```text
uci
setoption name SyzygyPath value D:\Chess Tables\syzygy
setoption name SyzygyProbeLimit value 5
setoption name SyzygyProbeDepth value 1
isready
position fen 4k3/8/8/8/8/8/8/R3K3 w - - 17 1
go depth 6
```

Windows paths can contain spaces; separate multiple directories with `;`.
`SyzygyProbeLimit` accepts 0–7 (default 5); 0 disables probing at runtime.
`SyzygyProbeDepth` accepts 0–64 (default 1), measured in remaining search plies.
The engine does not save UCI options itself; configure the path in your GUI or
send it each time the engine starts. Installation is an explicit setup step:
cloning, building, and first use do not download tables. No tablebase files or
trained `weights.bin` are bundled, and the engine never downloads data during play.
An empty `SyzygyPath` or `<empty>` also disables probing. Path discovery reports
the largest discovered material, not a guarantee of complete coverage. Missing
files cause ordinary search fallback. Options apply between searches; the
existing UCI loop is synchronous. Reconfiguration clears search caches while
preserving the current position and game history. Enabled builds report `tbhits`.

### Implementation and performance contract

The pinned [Fathom source](third_party/fathom/README.kitty.md) is compiled directly
into enabled builds. It reads Syzygy files through memory mappings. The compact
`tablebase.*` module owns its lifecycle, translates bitboards and moves, and
exposes optional results. There are no subprocesses, network probes, virtual
calls, or per-probe heap allocations in the adapter. Initial file mapping and
Fathom's lazy table preparation can allocate and perform I/O.

`if constexpr` selects the search path, and empty template specializations
remove disabled service and search state. When compiled out, there are no
tablebase members, options, counters, probe checks, or linked Fathom objects. An enabled build with no usable path or with
limit 0 selects the ordinary search specialization once, before iterative
deepening. Active search uses cheap eligibility checks before converting a
position. Each iteration selects ordinary search when even capturing one piece
every ply cannot reach the configured piece limit before the WDL depth cutoff;
this also avoids checks in early iterations of timed searches.
Root DTZ probing remains eligible regardless of that depth cutoff. The neural
evaluator is unchanged. The original `SearchResult` fields and `UciProtocol`
interface are preserved. Enabled UCI option handling lives in `tablebase_uci.*`;
`Board::tablebaseHits()` exposes the last completed search iteration's hit count
without adding a field to `SearchResult`.

DTZ ranks legal root moves once per `go`, then search considers only the best
rank throughout iterative deepening. Search WDL probes require a zero halfmove
clock, sufficient remaining depth, supported piece count, and no castling rights.
Root DTZ ranking falls back to ordinary search when the reversible game history
contains a repeated position: Syzygy does not encode that history, and its ranks
must not exclude a repetition draw or overwrite its score. Interior WDL probing
can resume after a capture or pawn move resets the clock and repetition window.
Quiescence does not probe directly. The fifty-move rule is always respected:
cursed wins and blessed losses count as draws. DTZ ranks at the rounding
boundary do not claim an exact outcome. Proven tablebase wins use score 30000
(adjusted by ply inside search); only an actual searched mate is reported as
`score mate`.

Draw checks precede probes and cached scores, with checkmate taking precedence
at the fifty-move boundary. Repetition keys omit an en-passant square only when
no legal en-passant capture exists; the evaluator's raw position hash stays
unchanged. Cache keys include the halfmove clock when the fifty-move boundary
is reachable within the hard 64-ply search limit, and results affected by a
tablebase or repetition are not stored as history-independent cache scores.
These draw/cache corrections also apply to disabled builds and are separate
from the zero tablebase-overhead guarantee. This is not a general solution to
graph-history interaction in heuristic transposition caches.

Probe latency depends on installed material, storage and the OS file cache.
The existing engine checks its time budget between completed iterations, so a
cold root probe is included in elapsed time but cannot be interrupted. This
change does not add asynchronous search or claim hard time deadlines.

### Verification

The installer has a standalone offline regression test, with no extra packages:

```powershell
.\tools\test-install-syzygy.ps1
```

It covers complete installation, interrupted and corrupt downloads, retries,
offline reruns, paths containing spaces and brackets, and manifest validation.

The small optional fixture set is downloaded only by an explicit test setup step:

```powershell
.\tools\fetch-syzygy-fixtures.ps1
$env:KITTY_SYZYGY_TEST_PATH = "$PWD\DiagnosticsScratch\syzygy"
.\x64\Release\syzygy\KittyEngineTest.exe
```

The script pins the python-chess fixture revision and checks all 70 files with
SHA-256. This is a 4.15 MiB test subset, not the recommended production set.
Real-file tests explicitly skip when the environment variable is absent.
Run both build variants and the enabled Debug/AddressSanitizer configuration.
GoogleTest must use compatible runtime and sanitizer settings in Debug.

For an independent decoder comparison and reproducible performance samples:

```powershell
.\tools\build-benchmark.ps1 -Name plain
.\tools\build-benchmark.ps1 -EnableSyzygy -Name syzygy
python -m pip install python-chess==1.999
python tools\check-syzygy.py DiagnosticsScratch\bench\syzygy\bench.exe DiagnosticsScratch\syzygy
python tools\compare-benchmarks.py --plain DiagnosticsScratch\bench\plain\bench.exe --enabled DiagnosticsScratch\bench\syzygy\bench.exe --tables DiagnosticsScratch\syzygy
```

The comparison checks both side orientations, legal moves, WDL, DTZ ranks,
en passant, promotions and nonzero clocks. It also exercises the full depth-3
search at probe depths 0 and 1, checking chosen moves, scores, legal PVs and
restoration of the board, with the transposition cache reused between positions.
Benchmarks use the handcrafted evaluator so they do not require private model
weights; they do not establish
neural-evaluator throughput or playing-strength gains. `build-benchmark.ps1`
also accepts `-SourceDirectory` for another compatible source checkout and
`-Assembly` for compiler listings. Raw timing samples remain in `DiagnosticsScratch/bench/`.

### Measured validation (2026-10-10)

On Windows x64, Ryzen 9 3900XT, MSVC 19.50.35728, AVX2 and link-time
optimization: Release passed 103 tests with the feature compiled out and 117
with it enabled. All 117 enabled tests also passed in Debug/AddressSanitizer.
The independent python-chess comparison passed 3,503 WDL/clock checks, 3,503
complete root rankings, and 7,006 depth-3 searches (probe depths 0 and 1).
Regression tests cover the inclusive capture/depth boundary and a game-history
repetition draw that must override a static tablebase loss. A second comparison
sampled all 145 installed 3-5-piece materials: 583 WDL/clock and root-ranking
checks, plus 1,166 depth-3 searches.

The disabled benchmark has no Fathom symbols in its link map and retains the
50,176-byte `.text` section size of the draw/cache prerequisite (`0e3f095`).
Disassembly shows no added instructions in the recursive search functions;
there are data-address and instruction-order differences. The C++20 refactor
is **not byte-identical** to the earlier implementation, so the earlier exact
binary-identity result no longer applies. `tools/check-disabled-code.py` is a
strict byte comparison, not a general performance test. The disabled search,
board, search-result and original protocol object sizes also match that
prerequisite. The separate draw-rule corrections remain part of both builds.

The following are median milliseconds from 15 interleaved rounds, five repeats,
excluding the first round and first repeat. Processes were pinned to logical
CPU 23 (`--cpu 23`). These are warm OS-cache samples using the small fixture set.

| Fixed-depth position | Compiled out | Enabled, inactive | Enabled, probing | Probing hits |
| --- | ---: | ---: | ---: | ---: |
| Start, depth 5 | 6.523 | 6.650 | 6.654 | 0 |
| Kiwipete, depth 4 | 79.615 | 80.103 | 80.187 | 0 |
| Middlegame, depth 5 | 30.342 | 30.918 | 30.966 | 0 |
| KRvK, depth 6 | 1.284 | 1.293 | 1.578 | 20 |
| KPvK, depth 7 | 0.591 | 0.595 | 1.083 | 517 |

The compiled-out build was within 1% of the draw/cache prerequisite in these
samples. Outside coverage, the enabled build took about 0.7-2.1% more time than
the compiled-out build. Against the previous macro-based enabled binary, the
active rook and pawn samples took 8.8% and 4.0% more time, respectively. The
C++20 refactor changes compiler layout; these small handcrafted-evaluator
benchmarks do not establish a universal enabled-overhead bound.

The prerequisite and previous enabled binaries were included with `--corrected`
and `--previous-enabled` in the same interleaved run. All corresponding
before/after node counts, scores, best moves and hit counts matched. Raw data:
`DiagnosticsScratch/bench/cpp20-comparison.csv`. Endgame searches can take
longer than searches without tablebases because their scores, root move sets
and search trees change (KPvK searched 8,464 nodes instead of 4,558). These
samples establish neither an Elo gain nor production NNUE performance, and
do not measure a cold disk cache.

### Tablebase lookup throughput

To time successful queries through the actual adapter, independently of search
and evaluation:

```powershell
.\tools\build-benchmark.ps1 -EnableSyzygy -Benchmark tablebase -Name lookup-rate
python tools\benchmark-tablebases.py --binary DiagnosticsScratch\bench\lookup-rate\bench.exe --tables DiagnosticsScratch\syzygy --cpu 23
```

The runner uses python-chess to generate deterministic, shuffled legal positions
with zero halfmove clocks. FEN parsing and console output are excluded from the
timing. Each query goes through Kitty's eligibility checks and bitboard
conversion. Root queries additionally include legal move generation, Fathom DTZ
ranking of all moves, and matching the results to engine moves. A root query
can perform many internal probes; its count is not a count of individual table
reads. Failed probes make the benchmark fail rather than inflate throughput.

On the same Ryzen 9 3900XT/MSVC machine, pinned to logical CPU 23, the medians
of seven warm rounds (at least one second each) were:

| Corpus | Positions | WDL queries/second | Full root DTZ queries/second |
| --- | ---: | ---: | ---: |
| 3 pieces | 2,502 | 5,465,160 | 23,364 |
| 4 pieces | 15,001 | 2,161,540 | 12,021 |
| Mixed | 17,503 | 2,323,070 | 12,690 |

For the mixed corpus this is about 0.430 microseconds per WDL query and 78.8
microseconds per full root query, averaged over a timed batch. Mixed-corpus
rounds ranged from 2.081–2.400 million WDL queries/second and 11,333–13,494
root queries/second. Every query succeeded and per-pass checksums were stable.

The first mixed-corpus pass in a fresh process measured 1.452 million WDL
queries/second and 13,339 root queries/second. Table discovery took about
74–78 ms and was measured separately. These first passes include lazy mapping
and preparation, but the OS file cache was **not flushed**; they are not cold
disk measurements. The 70 fixture files total 4.15 MiB. These results do not
predict throughput for larger 5–7-piece collections or parallel probing.

Exact position corpora, initialization logs and all timing samples are saved
locally under `DiagnosticsScratch/bench/probe-rate/`. The runner supports
`--samples-per-material`, `--rounds`, `--seconds`, and `--output` for reruns.

## Authorship
The core engine, including bitboards, search, and evaluation, was coded manually.  
The UCI protocol implementation and terminal UI were done with the assistance from AI.

## Perft benchmarks
AMD Ryzen 7 4800H, 2.90 GHz  
DDR4, 3200MT/s, L1 cache: 512 KB, L2 cache: 4.0 MB, L3 cache: 8.0 MB  

### Normalized Move Generation
#### Initial Position
depth 1, nodes 20, time 0 ms, speed 20 knps  
depth 2, nodes 400, time 0 ms, speed 400 knps  
depth 3, nodes 8902, time 0 ms, speed 8902 knps  
depth 4, nodes 197281, time 0 ms, speed 197281 knps  
depth 5, nodes 4865609, time 15 ms, speed 324373 knps  
depth 6, nodes 119060324, time 394 ms, speed 302183 knps  
depth 7, nodes 3195901860, time 10117 ms, speed 315894 knps  

#### Kiwipete
depth 1, nodes 48, time 0 ms, speed 48 knps  
depth 2, nodes 2039, time 0 ms, speed 2039 knps  
depth 3, nodes 97862, time 0 ms, speed 97862 knps  
depth 4, nodes 4085603, time 10 ms, speed 408560 knps  
depth 5, nodes 193690690, time 475 ms, speed 407769 knps  
depth 6, nodes 8031647685, time 21556 ms, speed 372594 knps  

#### Rook Endgame
depth 1, nodes 14, time 0 ms, speed 14 knps  
depth 2, nodes 191, time 0 ms, speed 191 knps  
depth 3, nodes 2812, time 0 ms, speed 2812 knps  
depth 4, nodes 43238, time 0 ms, speed 43238 knps  
depth 5, nodes 674624, time 2 ms, speed 337312 knps  
depth 6, nodes 11030083, time 40 ms, speed 275752 knps  
depth 7, nodes 178633661, time 652 ms, speed 273978 knps  

### Raw Move Generation
#### Initial Position  
depth 1, nodes 20, time 0ms, speed 20 knps  
depth 2, nodes 400, time 0ms, speed 400 knps  
depth 3, nodes 8902, time 0ms, speed 8902 knps  
depth 4, nodes 197281, time 0ms, speed 197281 knps  
depth 5, nodes 4865609, time 12ms, speed 405467 knps  
depth 6, nodes 119060740, time 299ms, speed 398196 knps  
depth 7, nodes 3195919204, time 7881ms, speed 405522 knps  

#### Kiwipete  
depth 1, nodes 48, time 0ms, speed 48 knps  
depth 2, nodes 2039, time 0ms, speed 2039 knps  
depth 3, nodes 97863, time 0ms, speed 97863 knps  
depth 4, nodes 4085690, time 7ms, speed 583670 knps  
depth 5, nodes 193696718, time 314ms, speed 616868 knps  
depth 6, nodes 8031974901, time 14559ms, speed 551684 knps  

#### Rook Endgame  
depth 1, nodes 14, time 0ms, speed 14 knps  
depth 2, nodes 191, time 0ms, speed 191 knps  
depth 3, nodes 2812, time 0ms, speed 2812 knps  
depth 4, nodes 43238, time 0ms, speed 43238 knps  
depth 5, nodes 674624, time 2ms, speed 337312 knps  
depth 6, nodes 11030083, time 31ms, speed 355809 knps  
depth 7, nodes 178633661, time 488ms, speed 366052 knps  

## Credits
- Bitboard-based chess engine guidance from Code Monkey King: [!channel 1](https://www.youtube.com/channel/UClA-jNuyJKqN-xCm7KPG_XA) and [!channel 2](https://www.youtube.com/channel/UCB9-prLkPwgvlKKqDgXhsMQ).
- Neural network architecture reference from [!David Miller](http://www.millermattson.com/dave/).
