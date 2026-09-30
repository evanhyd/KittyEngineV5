#include "terminal_interface_policy.h"
#include "notation.h"
#include <algorithm>

namespace bb::user_interface {
  TerminalInterfacePolicy::TerminalInterfacePolicy(std::istream& uciInput,
                                                   std::ostream& uciOutput,
                                                   std::ostream& humanOutput)
    : uciInput_(uciInput), uciOutput_(uciOutput), humanOutput_(humanOutput) {}

  void TerminalInterfacePolicy::reportProtocolError(std::string_view message) {
    std::string safeMessage{message};
    std::replace_if(safeMessage.begin(), safeMessage.end(),
                    [](char ch) { return ch == '\r' || ch == '\n'; }, ' ');
    uciOutput_ << "info string error: " << safeMessage << '\n';
  }

  void TerminalInterfacePolicy::renderState(const BoardState& state) {
    printBuffer_.clear();
    for (Square rank = 0; rank < kSideSize; ++rank) {
      printBuffer_ += "  -------------------------------\n ";
      for (Square file = 0; file < kSideSize; ++file) {
        const auto occupant = state.getPieceAt(rankFileToSquare(rank, file));
        const char symbol = occupant
          ? notation::pieceToAsciiVisual(std::get<0>(*occupant), std::get<1>(*occupant))
          : ' ';
        printBuffer_ += "| ";
        printBuffer_ += symbol;
        printBuffer_ += ' ';
      }
      printBuffer_ += "| ";
      printBuffer_ += static_cast<char>('8' - rank);
      printBuffer_ += '\n';
    }
    printBuffer_ += "  -------------------------------\n   a   b   c   d   e   f   g   h\n";
    printBuffer_ += std::format("Side to move: {}\nCastling: {}\nEn passant: {}\nHalfmove clock: {}\nFullmove number: {}\n",
                         notation::colorToString(state.getColorToMove()),
                         notation::castleToString(state.getCastlingRights()),
                         notation::squareToString(state.getEnpassantSquare()),
                         state.getHalfmoveClock(), state.getFullmoveNumber());
    printBuffer_ += "FEN: ";
    printBuffer_ += notation::boardToFen(state);
    printBuffer_ += '\n';

    if (printBuffer_ == lastFrame_) {
      return;
    }

    humanOutput_.write(printBuffer_.data(), static_cast<std::streamsize>(printBuffer_.size()));
    humanOutput_.flush();
    if (humanOutput_) {
      lastFrame_ = printBuffer_;
    }
  }
}
