#include <iostream>
#include <ranges>
#include "utility.h"
#include "board.h"
#include "move.h"

using namespace std;

int main() {
  for (size_t i : ranges::iota_view(0, 64)) {
    printBitboard(kKingAttackTable[i]);
  }

  _pext_u64(1, 1);
}
