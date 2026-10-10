# Fathom in KittyEngine

Source: [jdart1/Fathom](https://github.com/jdart1/Fathom), commit
`c9c6fef0dddc05d2e242c183acf5833149ab676d`.
`LICENSE` is copied from the repository root; the other files are from `src/`.
Keep the MIT license and the copyright notices when distributing this code.

Only `tbprobe.c` is compiled. It includes `tbchess.c`; do not compile that file
separately. Kitty builds it as C++20 with `/Zc:__cplusplus`, which selects the
upstream C++20 allocation and atomic initialization path. `TB_NO_HELPER_API`
omits unused helper exports. Thread support remains enabled. Release uses the
engine's normal optimization and link-time optimization settings.

Local changes to `tbprobe.c`:

1. Close the Windows mapping handle if `MapViewOfFile` fails.
2. Return a failed probe after a mapping failure instead of terminating the engine.
3. Keep the incomplete-file diagnostic on stderr and remove its stdout duplicate,
   so it cannot corrupt the UCI stream.

No decompression, indexing, move generation, or table-format changes are made.
Kitty-specific conversion and configuration live in `KittyEngineV5/tablebase.*`.

Upstream SHA-256 values, before the three local changes:

| File | SHA-256 |
| --- | --- |
| LICENSE | c8038055839bd02f995cd7d1bba19657720097f5604411a7977ebcb5b4a37759 |
| stdendian.h | e5b8187eec89ef83e731ea92be8ca0df4b533d8d21075fc4e2c86b4b00d7b7c3 |
| tbchess.c | 119f19fa8714686798ca7be011e5b234f99c93aec751e9492f2664757d6c029d |
| tbconfig.h | 69fe1821b471164bd759cf46c486d55b7fa727e348a5ee306b18d19537ac3a03 |
| tbprobe.c | 7ed0cc80626271342bfe413d6c421756b6851f977a064461f5f555e1b598df35 |
| tbprobe.h | dc7808dc58c2a1af921a612f94041b917dfac94d3215a76a2b2b0c53c94ea561 |
