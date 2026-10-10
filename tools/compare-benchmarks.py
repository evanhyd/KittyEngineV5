"""Interleave fixed-depth runs; preserve raw samples and report medians.

All binaries must be built from tools/search_benchmark.cpp with identical flags.
The handcrafted evaluator makes this independent of the untracked weights.bin.
"""
import argparse
import csv
import io
import pathlib
import random
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain", required=True)
    parser.add_argument("--enabled", required=True)
    parser.add_argument("--tables", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--corrected")
    parser.add_argument("--previous-enabled", help="Earlier enabled build for an interleaved before/after comparison")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cpu", type=int, help="Windows logical CPU to pin this runner and its child processes to")
    parser.add_argument("--output", default="DiagnosticsScratch/bench/comparison.csv")
    args = parser.parse_args()
    if args.rounds < 2 or args.repeats < 2:
        parser.error("At least two rounds and repeats are needed to exclude warmup samples")
    if args.cpu is not None:
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), 1 << args.cpu):
            raise ctypes.WinError(ctypes.get_last_error())
    variants = {"plain": (args.plain, "-"), "inactive": (args.enabled, "-"),
                "active": (args.enabled, str(pathlib.Path(args.tables).resolve()))}
    if args.previous_enabled:
        variants["previous-inactive"] = (args.previous_enabled, "-")
        variants["previous-active"] = (args.previous_enabled, str(pathlib.Path(args.tables).resolve()))
    for label in ("baseline", "corrected"):
        if binary := getattr(args, label):
            variants[label] = (binary, "-")
    rng = random.Random(1977)
    rows = []
    for round_number in range(args.rounds):
        order = list(variants)
        rng.shuffle(order)
        for label in order:
            binary, tables = variants[label]
            run = subprocess.run([str(pathlib.Path(binary).resolve()), tables, str(args.repeats)],
                                 check=True, text=True, capture_output=True)
            rows.extend(dict(variant=label, round=round_number, **row)
                        for row in csv.DictReader(io.StringIO(run.stdout)))
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("variant,position,median_ms,vs_plain_percent,nodes,tbhits")
    for label in variants:
        for position in dict.fromkeys(r["position"] for r in rows):
            sample = [r for r in rows if r["variant"] == label and r["position"] == position
                      and int(r["round"]) > 0 and int(r["repeat"]) > 0]
            plain = [float(r["milliseconds"]) for r in rows if r["variant"] == "plain"
                     and r["position"] == position and int(r["round"]) > 0 and int(r["repeat"]) > 0]
            median = statistics.median(float(r["milliseconds"]) for r in sample)
            percent = (median / statistics.median(plain) - 1) * 100
            print(f'{label},{position},{median:.4f},{percent:+.2f},'
                  f'{sample[0]["nodes"]},{sample[0]["tbhits"]}')
    print(f"Raw samples: {output}")


if __name__ == "__main__":
    main()
