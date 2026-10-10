#include "tablebase.h"
#include "../third_party/fathom/tbprobe.h"
#include <atomic>
#include <charconv>
#include <limits>
#include <stdexcept>
#include <intrin.h>

namespace bb::tablebase {
  namespace {
    constexpr unsigned kRankFlipMask = 56;
    constexpr int kDtzOutcomeBoundary = 900;
    std::atomic_flag owned{};

    // Fathom numbers a1=0, Kitty numbers a8=0. Reverse rank bytes only.
    uint64_t flip(uint64_t bits) noexcept { return _byteswap_uint64(bits); }

    struct Position {
      uint64_t white;
      uint64_t black;
      uint64_t kings;
      uint64_t queens;
      uint64_t rooks;
      uint64_t bishops;
      uint64_t knights;
      uint64_t pawns;
      unsigned clock;
      unsigned ep;
      bool whiteToMove;

      explicit Position(const BoardState& state) noexcept
        : white(flip(state.getOccupancy(White))), black(flip(state.getOccupancy(Black))),
          kings(pieces(state, King)), queens(pieces(state, Queen)), rooks(pieces(state, Rook)),
          bishops(pieces(state, Bishop)), knights(pieces(state, Knight)), pawns(pieces(state, Pawn)),
          clock(static_cast<unsigned>(state.getHalfmoveClock())),
          ep(state.getEnpassantSquare() == NoSquare ? 0 : state.getEnpassantSquare() ^ kRankFlipMask),
          whiteToMove(state.getSideToMove() == White) {}

      static uint64_t pieces(const BoardState& state, Piece piece) noexcept {
        return flip(state.getPieces(White, piece) | state.getPieces(Black, piece));
      }
    };

    int parseNumber(std::string_view value, int maximum) {
      int number = 0;
      auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
      if (error != std::errc{} || end != value.data() + value.size() || number < 0 || number > maximum) {
        throw std::invalid_argument("Syzygy option is outside its supported range");
      }
      return number;
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
    const Position position(state);
    const unsigned result = tb_probe_wdl(position.white, position.black, position.kings, position.queens, position.rooks,
      position.bishops, position.knights, position.pawns, 0, 0, position.ep, position.whiteToMove);
    if (result == std::numeric_limits<unsigned>::max()) {
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
    const Position position(state);
    TbRootMoves ranks;
    if (!tb_probe_root_dtz(position.white, position.black, position.kings, position.queens, position.rooks,
        position.bishops, position.knights, position.pawns, position.clock, 0, position.ep, position.whiteToMove, repeated, true, &ranks) ||
        ranks.size != legalMoves.size()) {
      return std::nullopt;
    }

    constexpr Piece kPromotions[] = {NoPiece, Queen, Rook, Bishop, Knight};
    int bestRank = std::numeric_limits<int>::min();
    RootResult result;
    std::array<bool, 218> matched{};
    for (unsigned i = 0; i < ranks.size; ++i) {
      const auto& entry = ranks.moves[i];
      // Fathom packs destination, source and promotion into bits 0:5, 6:11 and 12:14.
      const unsigned promotion = (entry.move >> 12) & 7u;
      if (promotion >= std::size(kPromotions)) {
        return std::nullopt;
      }
      size_t found = legalMoves.size();
      for (size_t j = 0; j < legalMoves.size(); ++j) {
        const auto& move = legalMoves[j];
        if (move.getSource() == (((entry.move >> 6) & 63u) ^ kRankFlipMask) &&
            move.getDest() == ((entry.move & 63u) ^ kRankFlipMask) &&
            move.getPromotedPieceType() == kPromotions[promotion]) {
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
    if (bestRank > kDtzOutcomeBoundary) {
      result.outcome = 1;
    } else if (bestRank < -kDtzOutcomeBoundary) {
      result.outcome = -1;
    } else if (bestRank > -kDtzOutcomeBoundary && bestRank < kDtzOutcomeBoundary) {
      result.outcome = 0;
    }
    return result;
  }
}
