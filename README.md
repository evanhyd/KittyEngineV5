# KittyEngineV5
A C++ bitboard chess engine with template-meta programming performance optimizations.

## Authorship
The core engine, including bitboards, search, and evaluation, was coded manually.  
The UCI protocol implementation and terminal UI were done with the assistance from AI.

## Searching Policy

### NegamaxSearchingPolicy

- Iterative deepening
- Alpha-beta pruning
- Aspiration windows
- Move ordering
- Transposition table
- Quiescence search
- Repetition and 50-move draw detection

## TimeControl Policy

### EqualPercentageTimeControlPolicy

- Budgets time from a percentage of the remaining clock or the move increment, whichever is greater.
- Checks whether to continue after each completed search depth.

## Evaluation Policy

### Handcrafted evaluation

The handcrafted policy combines:

- Material values and piece-square tables.
- Mobility for knights, bishops, and rooks.
- Isolated, doubled, and passed pawn scores.
- Bishop-pair and open or semi-open rook-file bonuses.
- King placement, pawn shelter, open files near the king, and nearby attacks.
- Penalties for overloaded defenders.

Pawn and king placement scores are blended between opening and endgame values according to the remaining material.

### MLP evaluation

- Architecture: 836 inputs, 256 and 32 hidden neurons, and one output; all values are floats.
- Inputs: 768 piece-square features, four castling rights, and 64 en passant squares.
- Perspective: inputs are encoded relative to each side, with one accumulator per perspective.
- Activations: ReLU in the hidden layers and `sigmoid(x / 400)` at the output.
- Score: the output is converted to centipawns.
- Inference: move callbacks update first-layer sums; later layers run when a score is requested.
- Weights: loaded from `weights.bin` in the current working directory, using the existing file format.

## UCI Command

| Command | Action |
| --- | --- |
| `uci` | Print engine identification and `uciok`. |
| `isready` | Reply with `readyok`. |
| `ucinewgame` | Reset search and evaluation state for a new game. The current board position is left for a subsequent `position` command to set. |
| `position startpos [moves <move> ...]` | Set the starting position and optionally play UCI moves. |
| `position fen <six FEN fields> [moves <move> ...]` | Set a FEN position and optionally play UCI moves. |
| `go depth <N>` | Search to depth `N`, up to the engine's maximum depth. |
| `go wtime <ms> btime <ms> [winc <ms>] [binc <ms>] [movestogo <N>]` | Search using clock times and optional increments. A `depth` limit may also be supplied. |
| `quit` | Exit the engine. |
| `play <move>` | Play one UCI move on the current board; terminal extension. |
| `perft depth <N> [detail]` | Count legal move sequences; `detail` also reports captures, en passant, castling, and promotions. Terminal extension. |

Search progress is reported with `info` lines followed by `bestmove`. The `go` command runs synchronously; commands such as `stop` and `setoption` are not implemented.

## Perft benchmarks
AMD Ryzen 7 4800H, 2.90 GHz  
DDR4, 3200MT/s, L1 cache: 512 KB, L2 cache: 4.0 MB, L3 cache: 8.0 MB  

### Normalized Move Generation
#### Initial Position
FEN: `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1`

depth 1, nodes 20, time 0 ms, speed 20 knps  
depth 2, nodes 400, time 0 ms, speed 400 knps  
depth 3, nodes 8902, time 0 ms, speed 8902 knps  
depth 4, nodes 197281, time 0 ms, speed 197281 knps  
depth 5, nodes 4865609, time 15 ms, speed 324373 knps  
depth 6, nodes 119060324, time 394 ms, speed 302183 knps  
depth 7, nodes 3195901860, time 10117 ms, speed 315894 knps  

#### Kiwipete
FEN: `r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1`

depth 1, nodes 48, time 0 ms, speed 48 knps  
depth 2, nodes 2039, time 0 ms, speed 2039 knps  
depth 3, nodes 97862, time 0 ms, speed 97862 knps  
depth 4, nodes 4085603, time 10 ms, speed 408560 knps  
depth 5, nodes 193690690, time 475 ms, speed 407769 knps  
depth 6, nodes 8031647685, time 21556 ms, speed 372594 knps  

#### Rook Endgame
FEN: `8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1`

depth 1, nodes 14, time 0 ms, speed 14 knps  
depth 2, nodes 191, time 0 ms, speed 191 knps  
depth 3, nodes 2812, time 0 ms, speed 2812 knps  
depth 4, nodes 43238, time 0 ms, speed 43238 knps  
depth 5, nodes 674624, time 2 ms, speed 337312 knps  
depth 6, nodes 11030083, time 40 ms, speed 275752 knps  
depth 7, nodes 178633661, time 652 ms, speed 273978 knps  

## Credits
- Bitboard-based chess engine guidance from Code Monkey King: [Code Monkey King](https://www.youtube.com/channel/UClA-jNuyJKqN-xCm7KPG_XA) and [Chess Programming](https://www.youtube.com/channel/UCB9-prLkPwgvlKKqDgXhsMQ).
- Neural network architecture reference from [David Miller](http://www.millermattson.com/dave/).
- Chess engine development reference: [Chess Programming Wiki](https://chessprogramming.org/).
