# KittyEngineV5
A C++ bitboard chess engine with template-meta programming performance optimizations.

## Authorship
The core engine, including bitboards, search, and evaluation, was coded manually.  
The UCI protocol implementation and terminal UI were done with the assistance from AI.

## Perft benchmarks
AMD Ryzen 7 4800H, 2.90 GHz  
DDR4, 3200MT/s, L1 cache: 512 KB, L2 cache: 4.0 MB, L3 cache: 8.0 MB  

### Initial Position  
depth 1, nodes 20, time 0ms, speed 20 knps  
depth 2, nodes 400, time 0ms, speed 400 knps  
depth 3, nodes 8902, time 0ms, speed 8902 knps  
depth 4, nodes 197281, time 0ms, speed 197281 knps  
depth 5, nodes 4865609, time 12ms, speed 405467 knps  
depth 6, nodes 119060740, time 299ms, speed 398196 knps  
depth 7, nodes 3195919204, time 7881ms, speed 405522 knps  

### Kiwipete  
depth 1, nodes 48, time 0ms, speed 48 knps  
depth 2, nodes 2039, time 0ms, speed 2039 knps  
depth 3, nodes 97863, time 0ms, speed 97863 knps  
depth 4, nodes 4085690, time 7ms, speed 583670 knps  
depth 5, nodes 193696718, time 314ms, speed 616868 knps  
depth 6, nodes 8031974901, time 14559ms, speed 551684 knps  

### Rook Endgame  
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
