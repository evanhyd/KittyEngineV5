#pragma once
#include <iostream>
#include "move.h"

void printBitboard(uint64_t bitboard) {
  using std::cout;

  auto fmtFlag = cout.flags();
  for (size_t i = 0; i < kRankSize; ++i) {
    cout << (kRankSize - i) << '|';
    for (size_t j = 0; j < kFileSize; ++j) {
      cout << ' ' << getSquare(bitboard, i * kRankSize + j);
    }
    cout << '\n';
  }
  cout << "   A B C D E F G H\n";
  cout << "Bitboard Hex: "  << std::hex << bitboard << "\n\n";
  cout.flags(fmtFlag);
}
