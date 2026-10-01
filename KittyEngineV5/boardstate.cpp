#include "boardstate.h"
#include "notation.h"
#include <sstream>
#include <stdexcept>

namespace bb {
  BoardState::BoardState() noexcept
    : bitboards_{}, color_{}, castlePermission_{}, enpassant_{}, halfmove_{}, fullmove_{} {}

  BoardState::BoardState(std::string_view fen) : BoardState() {
    setPosition(fen);
  }

  void BoardState::setPosition(std::string_view fen) {
    BoardState parsed{};

    std::istringstream input{std::string(fen)};
    std::string placement;
    std::string side;
    std::string castling;
    std::string enpassant;
    if (!(input >> placement >> side >> castling >> enpassant)) {
      throw std::invalid_argument("FEN needs placement, side, castling, and en passant fields");
    }

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
        parsed.bitboards_[color][piece] = setSquare(parsed.bitboards_[color][piece], square);
        ++square;
      }
      if (square > (completedRanks + 1) * kSideSize) {
        throw std::invalid_argument("Invalid FEN rank width");
      }
    }
    if (completedRanks != kSideSize - 1 || square != kSquareSize) {
      throw std::invalid_argument("Invalid FEN placement");
    }

    parsed.color_ = notation::stringToSide(side);
    parsed.castlePermission_ = notation::stringToCastling(castling);

    parsed.enpassant_ = notation::stringToSquare(enpassant);
    if (parsed.enpassant_ != kNoSquare && enpassant[1] != '3' && enpassant[1] != '6') {
      throw std::invalid_argument("Invalid FEN en passant square");
    }

    parsed.fullmove_ = 1;
    if (input >> parsed.halfmove_) {
      if (!(input >> parsed.fullmove_) || parsed.halfmove_ < 0 || parsed.fullmove_ <= 0) {
        throw std::invalid_argument("Invalid FEN move counters");
      }
    } else if (!input.eof()) {
      throw std::invalid_argument("Invalid FEN halfmove counter");
    }

    std::string extraField;
    if (input >> extraField) {
      throw std::invalid_argument("Too many FEN fields");
    }
    *this = parsed;
  }
}
