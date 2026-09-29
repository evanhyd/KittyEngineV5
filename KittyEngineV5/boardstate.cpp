#include "boardstate.h"
#include "notation.h"
#include <sstream>
#include <stdexcept>

namespace bb {
  std::pair<BoardState, Color> BoardState::fromFEN(std::string_view fen) {
    std::istringstream input{std::string(fen)};
    std::string placement;
    std::string side;
    std::string castling;
    std::string enpassant;
    if (!(input >> placement >> side >> castling >> enpassant)) {
      throw std::invalid_argument("FEN needs placement, side, castling, and en passant fields");
    }

    BoardState state{};
    Square square = 0;
    Square completedRanks = 0;
    for (const char symbol : placement) {
      if (symbol == '/') {
        if (completedRanks == kSideSize - 1 || square != (completedRanks + 1) * kSideSize) {
          throw std::invalid_argument("Invalid FEN rank");
        }
        ++completedRanks;
        continue;
      }
      if (symbol >= '1' && symbol <= '8') {
        square += symbol - '0';
      } else {
        if (square >= (completedRanks + 1) * kSideSize) {
          throw std::invalid_argument("Invalid FEN rank width");
        }
        const auto [color, piece] = notation::asciiToPiece(symbol);
        state.bitboards_[color][piece] = setSquare(state.bitboards_[color][piece], square);
        ++square;
      }
      if (square > (completedRanks + 1) * kSideSize) {
        throw std::invalid_argument("Invalid FEN rank width");
      }
    }
    if (completedRanks != kSideSize - 1 || square != kSquareSize) {
      throw std::invalid_argument("Invalid FEN placement");
    }

    const Color sideToMove = notation::parseSideToMove(side);
    state.castlePermission_ = notation::parseCastlingRights(castling);

    state.enpassant_ = notation::stringToSquare(enpassant);
    if (state.enpassant_ != NO_SQUARE && enpassant[1] != '3' && enpassant[1] != '6') {
      throw std::invalid_argument("Invalid FEN en passant square");
    }

    state.fullmove_ = 1;
    if (input >> state.halfmove_) {
      if (!(input >> state.fullmove_) || state.halfmove_ < 0 || state.fullmove_ <= 0) {
        throw std::invalid_argument("Invalid FEN move counters");
      }
    } else if (!input.eof()) {
      throw std::invalid_argument("Invalid FEN halfmove counter");
    }

    std::string extraField;
    if (input >> extraField) {
      throw std::invalid_argument("Too many FEN fields");
    }
    return {state, sideToMove};
  }
}
