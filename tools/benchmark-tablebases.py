"""Measure successful adapter WDL and full root-DTZ queries on local tables.

Requires python-chess==1.999. Build with build-benchmark.ps1 -Benchmark tablebase
-EnableSyzygy. This does not download tables or flush the OS disk cache.
"""
import argparse
import csv
import importlib.util
import io
import pathlib
import random
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--tables", required=True)
    parser.add_argument("--cpu", type=int, help="Windows logical CPU for this process and its children")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seconds", type=float, default=1)
    parser.add_argument("--samples-per-material", type=int, default=500)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("DiagnosticsScratch/bench/probe-rate"))
    args = parser.parse_args()
    if args.rounds < 1 or not 0.1 <= args.seconds <= 60 or args.samples_per_material < 1:
        parser.error("Use positive rounds/samples and a duration between 0.1 and 60 seconds")
    if args.cpu is not None:
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), 1 << args.cpu):
            raise ctypes.WinError(ctypes.get_last_error())

    tables = pathlib.Path(args.tables).resolve()
    spec = importlib.util.spec_from_file_location("syzygy_oracle", pathlib.Path(__file__).with_name("check-syzygy.py"))
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    boards = [b for b in oracle.cases(tables, args.samples_per_material) if b.halfmove_clock == 0]
    random.Random(1977).shuffle(boards)
    if not boards:
        parser.error("No positions could be generated from the installed WDL materials")
    args.output.mkdir(parents=True, exist_ok=True)
    groups = {str(n): [b for b in boards if len(b.piece_map()) == n]
              for n in sorted({len(b.piece_map()) for b in boards})}
    groups["mixed"] = boards
    rows = []
    print("pieces,query,positions,median_per_second,min_per_second,max_per_second,mean_microseconds,first_pass_per_second", flush=True)
    for label, group in groups.items():
        corpus = args.output / f"positions-{label}.fen"
        corpus.write_text("".join(b.fen(en_passant="fen") + "\n" for b in group), encoding="utf-8")
        for mode in ("wdl", "root"):
            run = subprocess.run([str(pathlib.Path(args.binary).resolve()), str(tables), str(corpus.resolve()),
                                  mode, str(args.rounds), str(args.seconds)],
                                 text=True, capture_output=True, check=True)
            (args.output / f"{label}-{mode}.log").write_text(run.stderr, encoding="utf-8")
            samples = list(csv.DictReader(io.StringIO(run.stdout)))
            assert len(samples) == args.rounds + 1, run.stdout
            assert all(int(s["failures"]) == 0 for s in samples), samples
            # A checksum depends only on positions, not how many complete passes
            # fit in a timed round. Check it against the initial single pass.
            checksum = int(samples[0]["checksum"])
            assert all(int(s["checksum"]) == checksum * (int(s["probes"]) // len(group)) for s in samples)
            rows.extend(dict(pieces=label, query=mode, positions=len(group), **s) for s in samples)
            rates = [float(s["probes_per_second"]) for s in samples if s["phase"] == "warm"]
            median = statistics.median(rates)
            print(f"{label},{mode},{len(group)},{median:.0f},{min(rates):.0f},{max(rates):.0f},"
                  f"{1e6 / median:.3f},{float(samples[0]['probes_per_second']):.0f}", flush=True)
    with (args.output / "samples.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Raw samples and exact position corpora: {args.output}")


if __name__ == "__main__":
    main()
