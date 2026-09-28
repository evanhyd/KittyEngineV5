#include "../KittyEngineV5/board.cpp"
#include "../KittyEngineV5/perft_driver.h"
#include <algorithm>
#include <array>
#include <vector>
#include <gtest/gtest.h>

namespace {

struct LegalMove {
  Square srce;
  Square dest;
  Piece piece;
  Piece promotion;
  bool enpassant;
  bool doublePush;
  bool kingSideCastle;
  bool queenSideCastle;
};

struct CollectMoves {
  static inline std::vector<LegalMove> moves;

  template <MoveType type>
  static void acceptMove(const BoardState&, Move<type> move) {
    moves.push_back({move.srce, move.dest, type.movedPiece, type.promotionPiece,
                     type.isEnpassant, type.isDoublePush, type.isKingSideCastle,
                     type.isQueenSideCastle});
  }
};

template <Color color>
const std::vector<LegalMove>& legalMoves(const BoardState& state) {
  CollectMoves::moves.clear();
  state.enumerateMoves<color, CollectMoves>();
  return CollectMoves::moves;
}

size_t countMoves(const std::vector<LegalMove>& moves, Square srce, Square dest) {
  return std::count_if(moves.begin(), moves.end(), [=](const LegalMove& move) {
    return move.srce == srce && move.dest == dest;
  });
}

Bitboard slowSliderAttack(Piece piece, Square srce, Bitboard occupancy) {
  const std::array<std::pair<int, int>, 4> rookDirections{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
  const std::array<std::pair<int, int>, 4> bishopDirections{{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}}};
  const auto& directions = piece == kRook ? rookDirections : bishopDirections;
  Bitboard attacks = 0;
  for (const auto& [dr, df] : directions) {
    int rank = static_cast<int>(getSquareRank(srce)) + dr;
    int file = static_cast<int>(getSquareFile(srce)) + df;
    while (rank >= 0 && rank < 8 && file >= 0 && file < 8) {
      const Square dest = rankFileToSquare(rank, file);
      attacks = setSquare(attacks, dest);
      if (isSquareSet(occupancy, dest)) {
        break;
      }
      rank += dr;
      file += df;
    }
  }
  return attacks;
}

} // namespace

TEST(AttackTables, EveryRelevantSliderOccupancy) {
  for (Square square = 0; square < kSquareSize; ++square) {
    SCOPED_TRACE("square " + std::to_string(square));
    for (Piece piece : {kBishop, kRook}) {
      const Bitboard mask = internal::sliderAttackTables[square][piece - kBishop].maxAttackNoEdge;
      for (Bitboard occupancy = mask;; occupancy = (occupancy - 1) & mask) {
        const Bitboard actual = piece == kBishop ? getAttack<kBishop>(square, occupancy)
                                                  : getAttack<kRook>(square, occupancy);
        EXPECT_EQ(actual, slowSliderAttack(piece, square, occupancy));
        if (occupancy == 0) {
          break;
        }
      }
    }
  }
}

TEST(AttackTables, CornerLeapersDoNotWrapFiles) {
  EXPECT_EQ(getAttack<kKnight>(A8), toBitboard(B6, C7));
  EXPECT_EQ(getAttack<kKnight>(H1), toBitboard(F2, G3));
  EXPECT_EQ(getAttack<kKing>(A8), toBitboard(A7, B7, B8));
  EXPECT_EQ((getAttack<kPawn, kWhite>(A7)), toBitboard(B8));
  EXPECT_EQ((getAttack<kPawn, kBlack>(H2)), toBitboard(G1));
}

TEST(BoardState, ParsesFen) {
  const auto state = BoardState::fromFEN("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 4 12");

  EXPECT_EQ(state.getColor(), kWhite);
  EXPECT_TRUE(isSquareSet(state.bitboards_[kWhite][kPawn], E5));
  EXPECT_TRUE(isSquareSet(state.bitboards_[kBlack][kPawn], D5));
  EXPECT_EQ(state.enpassant_, D6);
  EXPECT_EQ(state.castlePermission_,
            kKingCastlePermission[kWhite] | kQueenCastlePermission[kWhite] |
            kKingCastlePermission[kBlack] | kQueenCastlePermission[kBlack]);
  EXPECT_EQ(state.halfmove_, 4u);
  EXPECT_EQ(state.fullmove_, 12u);
}

TEST(LegalMoves, InitialPosition) {
  const auto state = BoardState::fromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(moves.size(), 20u);
  EXPECT_EQ(countMoves(moves, E2, E4), 1u);
  EXPECT_EQ(countMoves(moves, E1, G1), 0u);
  const auto doublePush = std::find_if(moves.begin(), moves.end(), [](const LegalMove& move) {
    return move.srce == E2 && move.dest == E4;
  });
  ASSERT_NE(doublePush, moves.end());
  EXPECT_TRUE(doublePush->doublePush);
}

TEST(LegalMoves, KingMustEscapeCheck) {
  const auto state = BoardState::fromFEN("4r2k/8/8/8/8/8/8/4K3 w - - 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(moves.size(), 4u);
  EXPECT_TRUE(std::all_of(moves.begin(), moves.end(), [](const LegalMove& move) {
    return move.piece == kKing;
  }));
}

TEST(LegalMoves, PinnedKnightCannotMove) {
  const auto state = BoardState::fromFEN("4r2k/8/8/8/8/8/4N3/4K3 w - - 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_TRUE(std::none_of(moves.begin(), moves.end(), [](const LegalMove& move) {
    return move.srce == E2;
  }));
}

TEST(LegalMoves, CastlingMovesRookAndClearsRights) {
  const auto state = BoardState::fromFEN("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E1, G1), 1u);
  EXPECT_EQ(countMoves(moves, E1, C1), 1u);
  const auto afterCastle = state.makeMove<MoveType{kWhite, kKing, 0, false, false, true, false}>({E1, G1});
  EXPECT_TRUE(isSquareSet(afterCastle.bitboards_[kWhite][kKing], G1));
  EXPECT_TRUE(isSquareSet(afterCastle.bitboards_[kWhite][kRook], F1));
  EXPECT_FALSE(isSquareSet(afterCastle.bitboards_[kWhite][kRook], H1));
  EXPECT_NE(afterCastle.castlePermission_ & kKingCastlePermission[kWhite], kKingCastlePermission[kWhite]);
  EXPECT_NE(afterCastle.castlePermission_ & kQueenCastlePermission[kWhite], kQueenCastlePermission[kWhite]);
}

TEST(LegalMoves, CannotCastleThroughAttack) {
  const auto state = BoardState::fromFEN("4k3/8/8/8/8/5r2/8/R3K2R w KQ - 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E1, G1), 0u);
  EXPECT_EQ(countMoves(moves, E1, C1), 1u);
}

TEST(LegalMoves, CannotCastleOutOfCheck) {
  const auto state = BoardState::fromFEN("4r2k/8/8/8/8/8/8/R3K2R w KQ - 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E1, G1), 0u);
  EXPECT_EQ(countMoves(moves, E1, C1), 0u);
}

TEST(LegalMoves, BlackCanCastleOnBothSides) {
  const auto state = BoardState::fromFEN("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1");
  const auto& moves = legalMoves<kBlack>(state);

  EXPECT_EQ(countMoves(moves, E8, G8), 1u);
  EXPECT_EQ(countMoves(moves, E8, C8), 1u);
}

TEST(LegalMoves, QueenSideCastleMayPassAttackedRookSquare) {
  const auto state = BoardState::fromFEN("1r5k/8/8/8/8/8/8/R3K2R w KQ - 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E1, C1), 1u);
}

TEST(LegalMoves, PinnedEnemyKnightStillAttacksCastlingPath) {
  const auto state = BoardState::fromFEN("6k1/8/8/8/8/6n1/6Q1/4K2R w K - 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E1, G1), 0u);
}

TEST(BoardState, MovingRookPermanentlyRemovesItsCastlingRight) {
  const auto state = BoardState::fromFEN("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
  const auto afterRookMove = state.makeMove<MoveType{kWhite, kRook, 0, false, false, false, false}>({H1, H2});
  const auto afterBlackMove = afterRookMove.makeMove<MoveType{kBlack, kKing, 0, false, false, false, false}>({E8, E7});
  const auto afterRookReturns = afterBlackMove.makeMove<MoveType{kWhite, kRook, 0, false, false, false, false}>({H2, H1});
  const auto& moves = legalMoves<kWhite>(afterRookReturns);

  EXPECT_EQ(countMoves(moves, E1, G1), 0u);
  EXPECT_EQ(countMoves(moves, E1, C1), 1u);
}

TEST(BoardState, CapturingCornerRookRemovesCastlingRight) {
  const auto state = BoardState::fromFEN("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
  const auto afterCapture = state.makeMove<MoveType{kBlack, kRook, 0, false, false, false, false}>({A8, A1});

  EXPECT_NE(afterCapture.castlePermission_ & kQueenCastlePermission[kWhite], kQueenCastlePermission[kWhite]);
  EXPECT_NE(afterCapture.castlePermission_ & kQueenCastlePermission[kBlack], kQueenCastlePermission[kBlack]);
  EXPECT_EQ(afterCapture.castlePermission_ & kKingCastlePermission[kWhite], kKingCastlePermission[kWhite]);
}

TEST(LegalMoves, PawnHasFourPromotionChoices) {
  const auto state = BoardState::fromFEN("7k/P7/8/8/8/8/8/7K w - - 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, A7, A8), 4u);
  const auto promoted = state.makeMove<MoveType{kWhite, kPawn, kQueen, false, false, false, false}>({A7, A8});
  EXPECT_FALSE(isSquareSet(promoted.bitboards_[kWhite][kPawn], A7));
  EXPECT_TRUE(isSquareSet(promoted.bitboards_[kWhite][kQueen], A8));
}

TEST(BoardState, CaptureRemovesOpponentPiece) {
  const auto state = BoardState::fromFEN("7k/8/8/8/8/8/4p3/4R2K w - - 17 1");
  const auto afterCapture = state.makeMove<MoveType{kWhite, kRook, 0, false, false, false, false}>({E1, E2});

  EXPECT_TRUE(isSquareSet(afterCapture.bitboards_[kWhite][kRook], E2));
  EXPECT_FALSE(isSquareSet(afterCapture.bitboards_[kWhite][kRook], E1));
  EXPECT_FALSE(isSquareSet(afterCapture.bitboards_[kBlack][kPawn], E2));
  EXPECT_EQ(afterCapture.halfmove_, 0u);
}

TEST(BoardState, EnPassantCaptureExpiresImmediately) {
  const auto state = BoardState::fromFEN("7k/2pp4/8/4P3/8/8/8/7K b - - 0 1");
  const auto afterPush = state.makeMove<MoveType{kBlack, kPawn, 0, false, true, false, false}>({D7, D5});
  EXPECT_EQ(afterPush.enpassant_, D6);

  const auto& whiteMoves = legalMoves<kWhite>(afterPush);
  const auto capture = std::find_if(whiteMoves.begin(), whiteMoves.end(), [](const LegalMove& move) {
    return move.srce == E5 && move.dest == D6 && move.enpassant;
  });
  ASSERT_NE(capture, whiteMoves.end());

  const auto afterCapture = afterPush.makeMove<MoveType{kWhite, kPawn, 0, true, false, false, false}>({E5, D6});
  EXPECT_EQ(afterCapture.enpassant_, NO_SQUARE);
  EXPECT_TRUE(isSquareSet(afterCapture.bitboards_[kWhite][kPawn], D6));
  EXPECT_FALSE(isSquareSet(afterCapture.bitboards_[kBlack][kPawn], D5));

  const auto& blackMoves = legalMoves<kBlack>(afterCapture);
  EXPECT_EQ(countMoves(blackMoves, C7, D6), 1u);
  EXPECT_TRUE(std::none_of(blackMoves.begin(), blackMoves.end(), [](const LegalMove& move) {
    return move.enpassant;
  }));
}

TEST(BoardState, EnPassantExpiresAfterQuietMove) {
  const auto state = BoardState::fromFEN("7k/3p4/8/4P3/8/8/8/7K b - - 0 1");
  const auto afterPush = state.makeMove<MoveType{kBlack, kPawn, 0, false, true, false, false}>({D7, D5});
  const auto afterQuietMove = afterPush.makeMove<MoveType{kWhite, kKing, 0, false, false, false, false}>({H1, G1});

  EXPECT_EQ(afterQuietMove.enpassant_, NO_SQUARE);
  EXPECT_EQ(afterQuietMove.getColor(), kBlack);
  EXPECT_EQ(afterQuietMove.fullmove_, 2u);
}

TEST(LegalMoves, DoubleCheckAllowsOnlyKingMoves) {
  const auto state = BoardState::fromFEN("7k/8/8/3p1p2/4K3/8/8/R7 w - - 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_FALSE(moves.empty());
  EXPECT_TRUE(std::all_of(moves.begin(), moves.end(), [](const LegalMove& move) {
    return move.piece == kKing;
  }));
}

TEST(LegalMoves, KingCannotRetreatIntoOpenedRookRay) {
  const auto state = BoardState::fromFEN("4r2k/8/8/8/8/8/4K3/8 w - - 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E2, E1), 0u);
}

TEST(LegalMoves, KingCannotMoveNextToEnemyKing) {
  const auto state = BoardState::fromFEN("8/8/8/8/8/8/6k1/4K3 w - - 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E1, F1), 0u);
  EXPECT_EQ(countMoves(moves, E1, D1), 1u);
}

TEST(LegalMoves, PinnedRookCanMoveOnlyOnPinLine) {
  const auto state = BoardState::fromFEN("4r2k/8/8/8/8/8/4R3/4K3 w - - 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E2, E8), 1u);
  EXPECT_EQ(countMoves(moves, E2, D2), 0u);
  EXPECT_EQ(countMoves(moves, E2, F2), 0u);
}

TEST(LegalMoves, DiagonallyPinnedPawnCanCapturePinner) {
  const auto state = BoardState::fromFEN("7k/8/8/8/8/4b3/3P4/2K5 w - - 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, D2, E3), 1u);
  EXPECT_EQ(countMoves(moves, D2, D3), 0u);
  EXPECT_EQ(countMoves(moves, D2, D4), 0u);
}

TEST(LegalMoves, DoublePawnPushCanBlockDiagonalCheck) {
  const auto state = BoardState::fromFEN("7k/8/1b6/8/8/8/3P4/6K1 w - - 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, D2, D4), 1u);
  EXPECT_EQ(countMoves(moves, D2, D3), 0u);
}

TEST(LegalMoves, PawnCannotJumpOverOccupiedSquare) {
  const auto state = BoardState::fromFEN("7k/8/8/8/8/4n3/4P3/7K w - - 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E2, E3), 0u);
  EXPECT_EQ(countMoves(moves, E2, E4), 0u);
}

TEST(LegalMoves, PawnDoublePushNeedsEmptyDestination) {
  const auto state = BoardState::fromFEN("7k/8/8/8/4n3/8/4P3/7K w - - 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E2, E3), 1u);
  EXPECT_EQ(countMoves(moves, E2, E4), 0u);
}

TEST(LegalMoves, EdgePawnsDoNotWrapFiles) {
  const auto state = BoardState::fromFEN("7k/8/8/P6P/8/8/8/K7 w - - 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, A5, A6), 1u);
  EXPECT_EQ(countMoves(moves, H5, H6), 1u);
  EXPECT_TRUE(std::all_of(moves.begin(), moves.end(), [](const LegalMove& move) {
    return move.piece != kPawn || (move.srce == A5 && move.dest == A6) ||
           (move.srce == H5 && move.dest == H6);
  }));
}

TEST(LegalMoves, CapturePromotionHasFourChoices) {
  const auto state = BoardState::fromFEN("1r5k/P7/8/8/8/8/8/7K w - - 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, A7, A8), 4u);
  EXPECT_EQ(countMoves(moves, A7, B8), 4u);
}

TEST(LegalMoves, BlackCapturePromotionHasFourChoices) {
  const auto state = BoardState::fromFEN("7k/8/8/8/8/8/p7/1R5K b - - 0 1");
  const auto& moves = legalMoves<kBlack>(state);

  EXPECT_EQ(countMoves(moves, A2, A1), 4u);
  EXPECT_EQ(countMoves(moves, A2, B1), 4u);
}

TEST(LegalMoves, WhiteEnPassantCanCaptureCheckingPawn) {
  const auto state = BoardState::fromFEN("7k/8/8/3pP3/4K3/8/8/8 w - d6 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E5, D6), 1u);
  EXPECT_TRUE(std::any_of(moves.begin(), moves.end(), [](const LegalMove& move) {
    return move.srce == E5 && move.dest == D6 && move.enpassant;
  }));
}

TEST(LegalMoves, WhiteEnPassantCanBlockBishopCheck) {
  const auto state = BoardState::fromFEN("5b1k/8/8/3pP3/1K6/8/8/8 w - d6 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E5, D6), 1u);
}

TEST(LegalMoves, EnPassantCannotExposeHorizontalRook) {
  const auto state = BoardState::fromFEN("7k/8/8/r1PpK3/8/8/8/8 w - d6 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, C5, D6), 0u);
}

TEST(LegalMoves, EnPassantCannotExposeDiagonalBishop) {
  const auto state = BoardState::fromFEN("7b/8/8/3pP3/8/8/8/K6k w - d6 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E5, D6), 0u);
}

TEST(LegalMoves, BlackEnPassantIsGenerated) {
  const auto state = BoardState::fromFEN("7k/8/8/8/3Pp3/8/8/7K b - d3 0 1");
  const auto& moves = legalMoves<kBlack>(state);

  EXPECT_EQ(countMoves(moves, E4, D3), 1u);
  EXPECT_TRUE(std::any_of(moves.begin(), moves.end(), [](const LegalMove& move) {
    return move.srce == E4 && move.dest == D3 && move.enpassant;
  }));
}

TEST(LegalMoves, BlackEnPassantCanCaptureCheckingPawn) {
  const auto state = BoardState::fromFEN("8/8/8/4k3/3Pp3/8/8/7K b - d3 0 1");
  const auto& moves = legalMoves<kBlack>(state);

  EXPECT_EQ(countMoves(moves, E4, D3), 1u);
  EXPECT_TRUE(std::any_of(moves.begin(), moves.end(), [](const LegalMove& move) {
    return move.srce == E4 && move.dest == D3 && move.enpassant;
  }));
}

TEST(LegalMoves, KingCannotCaptureDefendedCheckingPiece) {
  const auto state = BoardState::fromFEN("7k/8/8/8/8/6n1/4r3/4K3 w - - 0 1");
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E1, E2), 0u);
}

TEST(LegalMoves, CheckmateAndStalemateHaveNoMoves) {
  const auto mate = BoardState::fromFEN("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1");
  const auto stalemate = BoardState::fromFEN("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1");

  EXPECT_TRUE(legalMoves<kBlack>(mate).empty());
  EXPECT_TRUE(legalMoves<kBlack>(stalemate).empty());
}

namespace {

template <size_t size>
void expectDetailedCounts(const char* fen, const std::array<perft::Result, size>& expected) {
  constexpr perft::Config config{false, false, true};
  const BoardState state = BoardState::fromFEN(fen);
  for (uint32_t depth = 1; depth < size; ++depth) {
    SCOPED_TRACE("depth " + std::to_string(depth));
    const auto actual = perft::countPerft<config>(state, depth);
    EXPECT_EQ(actual.nodes, expected[depth].nodes);
    EXPECT_EQ(actual.captures, expected[depth].captures);
    EXPECT_EQ(actual.enpassants, expected[depth].enpassants);
    EXPECT_EQ(actual.castles, expected[depth].castles);
    EXPECT_EQ(actual.promotions, expected[depth].promotions);
  }
}

template <size_t size>
void expectNodeCounts(const char* fen, const std::array<uint64_t, size>& expected) {
  constexpr perft::Config config{false, true, false};
  const BoardState state = BoardState::fromFEN(fen);
  for (uint32_t depth = 1; depth < size; ++depth) {
    SCOPED_TRACE("depth " + std::to_string(depth));
    EXPECT_EQ(perft::countPerft<config>(state, depth).nodes, expected[depth]);
  }
}

} // namespace

TEST(PerftCounts, InitialPositionEveryDepth) {
  // source: https://chessprogramming.org/Perft_Results#initial-position
  expectDetailedCounts("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                       std::array<perft::Result, 7>{{
                         {}, {20, 0, 0, 0, 0}, {400, 0, 0, 0, 0},
                         {8902, 34, 0, 0, 0}, {197281, 1576, 0, 0, 0},
                         {4865609, 82719, 258, 0, 0}, {119060324, 2812008, 5248, 0, 0}
                       }});
}

TEST(PerftCounts, KiwipeteEveryDepth) {
  // source: https://chessprogramming.org/Perft_Results#position-2
  expectDetailedCounts("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
                       std::array<perft::Result, 5>{{
                         {}, {48, 8, 0, 2, 0}, {2039, 351, 1, 91, 0},
                         {97862, 17102, 45, 3162, 0}, {4085603, 757163, 1929, 128013, 15172}
                       }});
}

TEST(PerftCounts, RookEndgameEveryDepth) {
  // source: https://chessprogramming.org/Perft_Results#position-3
  expectDetailedCounts("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -",
                       std::array<perft::Result, 7>{{
                         {}, {14, 1, 0, 0, 0}, {191, 14, 0, 0, 0},
                         {2812, 209, 2, 0, 0}, {43238, 3348, 123, 0, 0},
                         {674624, 52051, 1165, 0, 0}, {11030083, 940350, 33325, 0, 7552}
                       }});
}

TEST(PerftCounts, MirrorViewEveryDepth) {
  // source: https://chessprogramming.org/Perft_Results#position-4
  expectDetailedCounts("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
                       std::array<perft::Result, 6>{{
                         {}, {6, 0, 0, 0, 0}, {264, 87, 0, 6, 48},
                         {9467, 1021, 4, 0, 120}, {422333, 131393, 0, 7795, 60032},
                         {15833292, 2046173, 6512, 0, 329464}
                       }});
}

TEST(PerftCounts, TalkChessBugEveryDepth) {
  // source: https://chessprogramming.org/Perft_Results#position-5
  expectNodeCounts("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
                   std::array<uint64_t, 6>{{0, 44, 1486, 62379, 2103487, 89941194}});
}

TEST(PerftCounts, StevenAltEveryDepth) {
  // source: https://chessprogramming.org/Perft_Results#position-6
  expectNodeCounts("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
                   std::array<uint64_t, 6>{{0, 46, 2079, 89890, 3894594, 164075551}});
}

TEST(PerftCounts, HorizontalEnPassantPinEveryDepth) {
  expectNodeCounts("7k/3p1p2/8/r1P1K1Pr/8/8/8/8 b - - 0 1",
                   std::array<uint64_t, 7>{{0, 23, 160, 3995, 26757, 712872, 5070440}});
}

TEST(PerftCounts, DiagonalEnPassantPinEveryDepth) {
  expectNodeCounts("7k/4p2q/2q5/3P1P2/4K3/8/8/8 b - - 0 1",
                   std::array<uint64_t, 7>{{0, 36, 201, 6985, 42904, 1511423, 9034785}});
}

TEST(PerftCounts, HorizontalCannotEnPassantEveryDepth) {
  expectNodeCounts("7k/r2pK3/8/2P5/8/8/8/8 b - - 0 1",
                   std::array<uint64_t, 7>{{0, 14, 93, 1489, 8497, 143911, 900561}});
}
