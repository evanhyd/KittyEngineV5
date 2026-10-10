#include "tablebase.h"
#if KITTY_ENABLE_SYZYGY
#include "../third_party/fathom/tbprobe.h"
#include <atomic>
#include <charconv>
#include <limits>
#include <stdexcept>
#include <intrin.h>

namespace bb::tablebase {
  namespace {
    std::atomic_flag owned = ATOMIC_FLAG_INIT;

    // Fathom numbers a1=0, Kitty numbers a8=0. Reverse rank bytes only.
    uint64_t flip(uint64_t bits) noexcept { return _byteswap_uint64(bits); }

    struct Position {
      uint64_t white, black, kings, queens, rooks, bishops, knights, pawns;
      unsigned clock, ep;
      bool whiteToMove;
      explicit Position(const BoardState& s) noexcept
        : white(flip(s.getOccupancy(White))), black(flip(s.getOccupancy(Black))),
          kings(pieces(s, King)), queens(pieces(s, Queen)), rooks(pieces(s, Rook)),
          bishops(pieces(s, Bishop)), knights(pieces(s, Knight)), pawns(pieces(s, Pawn)),
          clock(static_cast<unsigned>(s.getHalfmoveClock())),
          ep(s.getEnpassantSquare() == NoSquare ? 0 : s.getEnpassantSquare() ^ 56),
          whiteToMove(s.getSideToMove() == White) {}
      static uint64_t pieces(const BoardState& s, Piece p) noexcept {
        return flip(s.getPieces(White, p) | s.getPieces(Black, p));
      }
    };

    int parseNumber(std::string_view value, int maximum) {
      int n = 0;
      auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), n);
      if (error != std::errc{} || end != value.data() + value.size() || n < 0 || n > maximum) {
        throw std::invalid_argument("Syzygy option is outside its supported range");
      }
      return n;
    }
  }

  Service::Service() {
    if (owned.test_and_set()) {
      throw std::logic_error("Only one Syzygy service may exist per process");
    }
  }

  Service::~Service() {
    if (initialized_) {
      tb_free();
    }
    owned.clear();
  }

  std::string Service::setOption(std::string_view name, std::string_view value) {
    if (name == "SyzygyPath") {
      if (value.find('\0') != std::string_view::npos) {
        throw std::invalid_argument("Invalid Syzygy path");
      }
      path_ = value == "<empty>" ? "" : std::string(value);
      largest_ = 0;
      initialized_ = true;
      if (!tb_init(path_.c_str())) {
        throw std::runtime_error("Could not initialize Syzygy tables");
      }
      largest_ = TB_LARGEST;
      return "Syzygy maximum discovered piece count " + std::to_string(largest_) +
        "; coverage depends on installed WDL and DTZ files";
    }
    if (name == "SyzygyProbeLimit") {
      limit_ = static_cast<unsigned>(parseNumber(value, 7));
    } else if (name == "SyzygyProbeDepth") {
      probeDepth_ = parseNumber(value, 64);
    } else {
      throw std::invalid_argument("Unknown Syzygy option");
    }
    return {};
  }

  std::optional<Wdl> Service::probeWdl(const BoardState& state) const {
    if (!enabled() || state.getHalfmoveClock() != 0 || !covers(state)) {
      return std::nullopt;
    }
    const Position p(state);
    const unsigned result = tb_probe_wdl(p.white, p.black, p.kings, p.queens, p.rooks,
      p.bishops, p.knights, p.pawns, 0, 0, p.ep, p.whiteToMove);
    if (result == TB_RESULT_FAILED) {
      return std::nullopt;
    }
    return static_cast<Wdl>(static_cast<int>(result) - 2);
  }

  std::optional<RootResult> Service::rankRoot(const BoardState& state, const MoveList& legalMoves, bool repeated) const {
    // DTZ cannot account for draws available from the actual game history.
    // Fathom's hasRepeated flag encourages progress; it does not resolve these
    // claims. Leave both move selection and scoring to search in this case.
    if (repeated || !enabled() || !covers(state) || state.getHalfmoveClock() >= 100 || legalMoves.empty()) {
      return std::nullopt;
    }
    const Position p(state);
    TbRootMoves ranks;
    if (!tb_probe_root_dtz(p.white, p.black, p.kings, p.queens, p.rooks,
        p.bishops, p.knights, p.pawns, p.clock, 0, p.ep, p.whiteToMove, repeated, true, &ranks) ||
        ranks.size != legalMoves.size()) {
      return std::nullopt;
    }

    constexpr Piece promotions[] = {NoPiece, Queen, Rook, Bishop, Knight};
    int bestRank = std::numeric_limits<int>::min();
    RootResult result;
    std::array<bool, 218> matched{};
    for (unsigned i = 0; i < ranks.size; ++i) {
      const auto& entry = ranks.moves[i];
      const unsigned promotion = TB_MOVE_PROMOTES(entry.move);
      if (promotion >= std::size(promotions)) {
        return std::nullopt;
      }
      size_t found = legalMoves.size();
      for (size_t j = 0; j < legalMoves.size(); ++j) {
        const auto& move = legalMoves[j];
        if (move.getSource() == (static_cast<unsigned>(TB_MOVE_FROM(entry.move)) ^ 56u) &&
            move.getDest() == (static_cast<unsigned>(TB_MOVE_TO(entry.move)) ^ 56u) &&
            move.getPromotedPieceType() == promotions[promotion]) {
          found = j;
          break;
        }
      }
      if (found == legalMoves.size() || matched[found]) {
        return std::nullopt;
      }
      matched[found] = true;
      if (entry.tbRank > bestRank) {
        bestRank = entry.tbRank;
        result.bestMoves.clear();
      }
      if (entry.tbRank == bestRank) {
        result.bestMoves.push(legalMoves[found]);
      }
    }
    if (bestRank > 900) {
      result.outcome = 1;
    } else if (bestRank < -900) {
      result.outcome = -1;
    } else if (bestRank > -900 && bestRank < 900) {
      result.outcome = 0;
    }
    return result;
  }
}
#endif
