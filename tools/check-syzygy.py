"""Compare Kitty's adapter with python-chess 1.999 (chess 1.11.2).

Usage: python tools/check-syzygy.py BENCH_EXE TABLE_DIRECTORY
Install python-chess separately. No network or table downloads occur here.
"""
import pathlib
import random
import subprocess
import sys

import chess
import chess.syzygy


def cases(directory):
    rng = random.Random(1977)
    for material in sorted(directory.glob("*.rtbw")):
        white, black = material.stem.split("v")
        for _ in range(50):
            while True:
                board = chess.Board(None)
                squares = rng.sample(range(64), len(white) + len(black))
                for symbol, square in zip(white + black.lower(), squares):
                    board.set_piece_at(square, chess.Piece.from_symbol(symbol))
                board.turn = rng.choice([chess.WHITE, chess.BLACK])
                if board.is_valid() and any(board.legal_moves):
                    break
            yield board
            board = board.copy()
            board.halfmove_clock = rng.choice([17, 80, 98, 99])
            yield board
    # Legal en passant and both promotion directions, including underpromotions.
    for fen in ["8/8/8/3pP3/8/4K3/8/k7 w - d6 0 1",
                "8/4P3/4K3/8/8/8/k7/8 w - - 0 1",
                "8/K7/8/8/8/4k3/4p3/8 b - - 0 1"]:
        yield chess.Board(fen)


def root_rank(tb, board, move):
    child = board.copy()
    child.push(move)
    if child.halfmove_clock == 0:
        v = [-1, -101, 0, 101, 1][-tb.probe_wdl(child) + 2]
    else:
        v = -tb.probe_dtz(child)
        v += (v > 0) - (v < 0)
        if v == 2 and child.is_checkmate():
            v = 1
    clock = board.halfmove_clock
    # Fathom's documented DTZ root ranking, using independently decoded values.
    if v > 0:
        return 1000 if v + clock <= 99 else 1000 - v - clock
    if v < 0:
        return -1000 if -v * 2 + clock < 100 else -1000 - v + clock
    return 0


def main():
    binary, directory = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]).resolve()
    boards = list(cases(directory))
    run = subprocess.run([str(binary), str(directory), "probe"],
                         input="".join(b.fen(en_passant="fen") + "\n" for b in boards),
                         text=True, capture_output=True, check=True)
    lines = run.stdout.splitlines()
    assert len(lines) == len(boards), (len(lines), len(boards), run.stderr)
    roots = 0
    with chess.syzygy.open_tablebase(directory) as tb:
        for board, line in zip(boards, lines):
            wdl, outcome, *moves = line.split()
            expected_wdl = tb.probe_wdl(board) if board.halfmove_clock == 0 else 9
            assert int(wdl) == expected_wdl, (board.fen(), line, expected_wdl)
            ranks = {m.uci(): root_rank(tb, board, m) for m in board.legal_moves}
            best = max(ranks.values())
            expected_moves = {m for m, r in ranks.items() if r == best}
            expected_outcome = 1 if best > 900 else -1 if best < -900 else 9 if abs(best) == 900 else 0
            assert set(moves) == expected_moves, (board.fen(), moves, expected_moves, ranks)
            assert int(outcome) == expected_outcome, (board.fen(), outcome, expected_outcome)
            roots += 1
    print(f"Passed {len(boards)} WDL/clock checks and {roots} complete root move rankings.")


if __name__ == "__main__":
    main()
