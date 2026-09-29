#include "boardstate.h"
#include <algorithm>
#include <array>
#include <string>
#include <vector>
#include <gtest/gtest.h>

using namespace bb;

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

template <Color color>
std::vector<LegalMove> legalMoves(const BoardState& state) {
  MoveList moves;
  state.generateMoves<color>(moves);
  std::vector<LegalMove> result;
  for (const Move& move : moves) {
    result.push_back({move.getSource(), move.getDest(), move.getMovedPiece(),
                      move.getPromotedPieceType(), move.isEnpassant(), move.isDoublePush(),
                      move.isCastling() && move.getDest() > move.getSource(),
                      move.isCastling() && move.getDest() < move.getSource()});
  }
  return result;
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

TEST(LegalMoves, InitialPosition) {
  const auto state = BoardState::fromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").first;
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
  const auto state = BoardState::fromFEN("4r2k/8/8/8/8/8/8/4K3 w - - 0 1").first;
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(moves.size(), 4u);
  EXPECT_TRUE(std::all_of(moves.begin(), moves.end(), [](const LegalMove& move) {
    return move.piece == kKing;
  }));
}

TEST(LegalMoves, PinnedKnightCannotMove) {
  const auto state = BoardState::fromFEN("4r2k/8/8/8/8/8/4N3/4K3 w - - 0 1").first;
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_TRUE(std::none_of(moves.begin(), moves.end(), [](const LegalMove& move) {
    return move.srce == E2;
  }));
}

TEST(LegalMoves, CastlingMovesRookAndClearsRights) {
  const auto state = BoardState::fromFEN("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").first;
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E1, G1), 1u);
  EXPECT_EQ(countMoves(moves, E1, C1), 1u);
  auto afterCastle = state;
  afterCastle.makeMove<kWhite>(Move(E1, G1, kKing, kNoPiece, Move::kCastlingFlag));
  EXPECT_TRUE(isSquareSet(afterCastle.bitboards_[kWhite][kKing], G1));
  EXPECT_TRUE(isSquareSet(afterCastle.bitboards_[kWhite][kRook], F1));
  EXPECT_FALSE(isSquareSet(afterCastle.bitboards_[kWhite][kRook], H1));
  EXPECT_NE(afterCastle.castlePermission_ & kKingCastlePermission[kWhite], kKingCastlePermission[kWhite]);
  EXPECT_NE(afterCastle.castlePermission_ & kQueenCastlePermission[kWhite], kQueenCastlePermission[kWhite]);
}

TEST(LegalMoves, CannotCastleThroughAttack) {
  const auto state = BoardState::fromFEN("4k3/8/8/8/8/5r2/8/R3K2R w KQ - 0 1").first;
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E1, G1), 0u);
  EXPECT_EQ(countMoves(moves, E1, C1), 1u);
}

TEST(LegalMoves, CannotCastleOutOfCheck) {
  const auto state = BoardState::fromFEN("4r2k/8/8/8/8/8/8/R3K2R w KQ - 0 1").first;
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E1, G1), 0u);
  EXPECT_EQ(countMoves(moves, E1, C1), 0u);
}

TEST(LegalMoves, BlackCanCastleOnBothSides) {
  const auto state = BoardState::fromFEN("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1").first;
  const auto& moves = legalMoves<kBlack>(state);

  EXPECT_EQ(countMoves(moves, E8, G8), 1u);
  EXPECT_EQ(countMoves(moves, E8, C8), 1u);
}

TEST(LegalMoves, QueenSideCastleMayPassAttackedRookSquare) {
  const auto state = BoardState::fromFEN("1r5k/8/8/8/8/8/8/R3K2R w KQ - 0 1").first;
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E1, C1), 1u);
}

TEST(LegalMoves, PinnedEnemyKnightStillAttacksCastlingPath) {
  const auto state = BoardState::fromFEN("6k1/8/8/8/8/6n1/6Q1/4K2R w K - 0 1").first;
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E1, G1), 0u);
}

TEST(BoardState, MovingRookPermanentlyRemovesItsCastlingRight) {
  const auto state = BoardState::fromFEN("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").first;
  auto afterRookReturns = state;
  afterRookReturns.makeMove<kWhite>(Move(H1, H2, kRook));
  afterRookReturns.makeMove<kBlack>(Move(E8, E7, kKing));
  afterRookReturns.makeMove<kWhite>(Move(H2, H1, kRook));
  const auto& moves = legalMoves<kWhite>(afterRookReturns);

  EXPECT_EQ(countMoves(moves, E1, G1), 0u);
  EXPECT_EQ(countMoves(moves, E1, C1), 1u);
}

TEST(BoardState, CapturingCornerRookRemovesCastlingRight) {
  const auto state = BoardState::fromFEN("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1").first;
  auto afterCapture = state;
  afterCapture.makeMove<kBlack>(Move(A8, A1, kRook, kNoPiece, Move::kCaptureFlag));

  EXPECT_NE(afterCapture.castlePermission_ & kQueenCastlePermission[kWhite], kQueenCastlePermission[kWhite]);
  EXPECT_NE(afterCapture.castlePermission_ & kQueenCastlePermission[kBlack], kQueenCastlePermission[kBlack]);
  EXPECT_EQ(afterCapture.castlePermission_ & kKingCastlePermission[kWhite], kKingCastlePermission[kWhite]);
}

TEST(LegalMoves, PawnHasFourPromotionChoices) {
  const auto state = BoardState::fromFEN("7k/P7/8/8/8/8/8/7K w - - 0 1").first;
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, A7, A8), 4u);
  auto promoted = state;
  promoted.makeMove<kWhite>(Move(A7, A8, kPawn, kQueen));
  EXPECT_FALSE(isSquareSet(promoted.bitboards_[kWhite][kPawn], A7));
  EXPECT_TRUE(isSquareSet(promoted.bitboards_[kWhite][kQueen], A8));
}

TEST(BoardState, CaptureRemovesOpponentPiece) {
  const auto state = BoardState::fromFEN("7k/8/8/8/8/8/4p3/4R2K w - - 17 1").first;
  auto afterCapture = state;
  afterCapture.makeMove<kWhite>(Move(E1, E2, kRook, kNoPiece, Move::kCaptureFlag));

  EXPECT_TRUE(isSquareSet(afterCapture.bitboards_[kWhite][kRook], E2));
  EXPECT_FALSE(isSquareSet(afterCapture.bitboards_[kWhite][kRook], E1));
  EXPECT_FALSE(isSquareSet(afterCapture.bitboards_[kBlack][kPawn], E2));
  EXPECT_EQ(afterCapture.halfmove_, 0);
}

TEST(BoardState, EnPassantCaptureExpiresImmediately) {
  const auto state = BoardState::fromFEN("7k/2pp4/8/4P3/8/8/8/7K b - - 0 1").first;
  auto afterPush = state;
  afterPush.makeMove<kBlack>(Move(D7, D5, kPawn, kNoPiece, Move::kDoublePushFlag));
  EXPECT_EQ(afterPush.enpassant_, D6);

  const auto& whiteMoves = legalMoves<kWhite>(afterPush);
  const auto capture = std::find_if(whiteMoves.begin(), whiteMoves.end(), [](const LegalMove& move) {
    return move.srce == E5 && move.dest == D6 && move.enpassant;
  });
  ASSERT_NE(capture, whiteMoves.end());

  auto afterCapture = afterPush;
  afterCapture.makeMove<kWhite>(Move(E5, D6, kPawn, kNoPiece, Move::kEnpassantFlag));
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
  const auto state = BoardState::fromFEN("7k/3p4/8/4P3/8/8/8/7K b - - 0 1").first;
  auto afterQuietMove = state;
  afterQuietMove.makeMove<kBlack>(Move(D7, D5, kPawn, kNoPiece, Move::kDoublePushFlag));
  afterQuietMove.makeMove<kWhite>(Move(H1, G1, kKing));

  EXPECT_EQ(afterQuietMove.enpassant_, NO_SQUARE);
  EXPECT_EQ(afterQuietMove.fullmove_, 2);
}

TEST(LegalMoves, DoubleCheckAllowsOnlyKingMoves) {
  const auto state = BoardState::fromFEN("7k/8/8/3p1p2/4K3/8/8/R7 w - - 0 1").first;
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_FALSE(moves.empty());
  EXPECT_TRUE(std::all_of(moves.begin(), moves.end(), [](const LegalMove& move) {
    return move.piece == kKing;
  }));
}

TEST(LegalMoves, KingCannotRetreatIntoOpenedRookRay) {
  const auto state = BoardState::fromFEN("4r2k/8/8/8/8/8/4K3/8 w - - 0 1").first;
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E2, E1), 0u);
}

TEST(LegalMoves, KingCannotMoveNextToEnemyKing) {
  const auto state = BoardState::fromFEN("8/8/8/8/8/8/6k1/4K3 w - - 0 1").first;
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E1, F1), 0u);
  EXPECT_EQ(countMoves(moves, E1, D1), 1u);
}

TEST(LegalMoves, PinnedRookCanMoveOnlyOnPinLine) {
  const auto state = BoardState::fromFEN("4r2k/8/8/8/8/8/4R3/4K3 w - - 0 1").first;
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E2, E8), 1u);
  EXPECT_EQ(countMoves(moves, E2, D2), 0u);
  EXPECT_EQ(countMoves(moves, E2, F2), 0u);
}

TEST(LegalMoves, DiagonallyPinnedPawnCanCapturePinner) {
  const auto state = BoardState::fromFEN("7k/8/8/8/8/4b3/3P4/2K5 w - - 0 1").first;
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, D2, E3), 1u);
  EXPECT_EQ(countMoves(moves, D2, D3), 0u);
  EXPECT_EQ(countMoves(moves, D2, D4), 0u);
}

TEST(LegalMoves, DoublePawnPushCanBlockDiagonalCheck) {
  const auto state = BoardState::fromFEN("7k/8/1b6/8/8/8/3P4/6K1 w - - 0 1").first;
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, D2, D4), 1u);
  EXPECT_EQ(countMoves(moves, D2, D3), 0u);
}

TEST(LegalMoves, PawnCannotJumpOverOccupiedSquare) {
  const auto state = BoardState::fromFEN("7k/8/8/8/8/4n3/4P3/7K w - - 0 1").first;
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E2, E3), 0u);
  EXPECT_EQ(countMoves(moves, E2, E4), 0u);
}

TEST(LegalMoves, PawnDoublePushNeedsEmptyDestination) {
  const auto state = BoardState::fromFEN("7k/8/8/8/4n3/8/4P3/7K w - - 0 1").first;
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E2, E3), 1u);
  EXPECT_EQ(countMoves(moves, E2, E4), 0u);
}

TEST(LegalMoves, EdgePawnsDoNotWrapFiles) {
  const auto state = BoardState::fromFEN("7k/8/8/P6P/8/8/8/K7 w - - 0 1").first;
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, A5, A6), 1u);
  EXPECT_EQ(countMoves(moves, H5, H6), 1u);
  EXPECT_TRUE(std::all_of(moves.begin(), moves.end(), [](const LegalMove& move) {
    return move.piece != kPawn || (move.srce == A5 && move.dest == A6) ||
           (move.srce == H5 && move.dest == H6);
  }));
}

TEST(LegalMoves, CapturePromotionHasFourChoices) {
  const auto state = BoardState::fromFEN("1r5k/P7/8/8/8/8/8/7K w - - 0 1").first;
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, A7, A8), 4u);
  EXPECT_EQ(countMoves(moves, A7, B8), 4u);
}

TEST(LegalMoves, BlackCapturePromotionHasFourChoices) {
  const auto state = BoardState::fromFEN("7k/8/8/8/8/8/p7/1R5K b - - 0 1").first;
  const auto& moves = legalMoves<kBlack>(state);

  EXPECT_EQ(countMoves(moves, A2, A1), 4u);
  EXPECT_EQ(countMoves(moves, A2, B1), 4u);
}

TEST(LegalMoves, WhiteEnPassantCanCaptureCheckingPawn) {
  const auto state = BoardState::fromFEN("7k/8/8/3pP3/4K3/8/8/8 w - d6 0 1").first;
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E5, D6), 1u);
  EXPECT_TRUE(std::any_of(moves.begin(), moves.end(), [](const LegalMove& move) {
    return move.srce == E5 && move.dest == D6 && move.enpassant;
  }));
}

TEST(LegalMoves, WhiteEnPassantCanBlockBishopCheck) {
  const auto state = BoardState::fromFEN("5b1k/8/8/3pP3/1K6/8/8/8 w - d6 0 1").first;
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E5, D6), 1u);
}

TEST(LegalMoves, EnPassantCannotExposeHorizontalRook) {
  const auto state = BoardState::fromFEN("7k/8/8/r1PpK3/8/8/8/8 w - d6 0 1").first;
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, C5, D6), 0u);
}

TEST(LegalMoves, EnPassantCannotExposeDiagonalBishop) {
  const auto state = BoardState::fromFEN("7b/8/8/3pP3/8/8/8/K6k w - d6 0 1").first;
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E5, D6), 0u);
}

TEST(LegalMoves, BlackEnPassantIsGenerated) {
  const auto state = BoardState::fromFEN("7k/8/8/8/3Pp3/8/8/7K b - d3 0 1").first;
  const auto& moves = legalMoves<kBlack>(state);

  EXPECT_EQ(countMoves(moves, E4, D3), 1u);
  EXPECT_TRUE(std::any_of(moves.begin(), moves.end(), [](const LegalMove& move) {
    return move.srce == E4 && move.dest == D3 && move.enpassant;
  }));
}

TEST(LegalMoves, BlackEnPassantCanCaptureCheckingPawn) {
  const auto state = BoardState::fromFEN("8/8/8/4k3/3Pp3/8/8/7K b - d3 0 1").first;
  const auto& moves = legalMoves<kBlack>(state);

  EXPECT_EQ(countMoves(moves, E4, D3), 1u);
  EXPECT_TRUE(std::any_of(moves.begin(), moves.end(), [](const LegalMove& move) {
    return move.srce == E4 && move.dest == D3 && move.enpassant;
  }));
}

TEST(LegalMoves, KingCannotCaptureDefendedCheckingPiece) {
  const auto state = BoardState::fromFEN("7k/8/8/8/8/6n1/4r3/4K3 w - - 0 1").first;
  const auto& moves = legalMoves<kWhite>(state);

  EXPECT_EQ(countMoves(moves, E1, E2), 0u);
}

TEST(LegalMoves, CheckmateAndStalemateHaveNoMoves) {
  const auto mate = BoardState::fromFEN("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1").first;
  const auto stalemate = BoardState::fromFEN("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1").first;

  EXPECT_TRUE(legalMoves<kBlack>(mate).empty());
  EXPECT_TRUE(legalMoves<kBlack>(stalemate).empty());
}

TEST(BoardState, EveryLegalMoveMakesAndUnmakesExactly) {
  // source: https://www.chessprogramming.org/Perft_Results
  const std::array<const char*, 6> positions{{
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"
  }};
  for (const char* fen : positions) {
    SCOPED_TRACE(fen);
    auto [state, sideToMove] = BoardState::fromFEN(fen);
    const BoardState original = state;
    MoveList moves;
    if (sideToMove == kWhite) {
      state.generateMoves<kWhite>(moves);
    } else {
      state.generateMoves<kBlack>(moves);
    }
    for (const Move& move : moves) {
      SCOPED_TRACE(std::to_string(move.getSource()) + " to " + std::to_string(move.getDest()));
      if (sideToMove == kWhite) {
        const auto undo = state.makeMove<kWhite>(move);
        state.unmakeMove<kWhite>(move, undo);
      } else {
        const auto undo = state.makeMove<kBlack>(move);
        state.unmakeMove<kBlack>(move, undo);
      }
      EXPECT_EQ(state.bitboards_, original.bitboards_);
      EXPECT_EQ(state.castlePermission_, original.castlePermission_);
      EXPECT_EQ(state.enpassant_, original.enpassant_);
      EXPECT_EQ(state.halfmove_, original.halfmove_);
      EXPECT_EQ(state.fullmove_, original.fullmove_);
    }
  }
}
