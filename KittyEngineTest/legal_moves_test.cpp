#include "boardstate.h"
#include "position_fens.h"
#include "test_move_callback.h"
#include <algorithm>
#include <array>
#include <string>
#include <string_view>
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

template <Side side>
std::vector<LegalMove> legalMoves(const BoardState& state) {
  MoveList moves;
  state.generateMoves<side>(moves);
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
  const auto& directions = piece == Rook ? rookDirections : bishopDirections;
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
    for (Piece piece : {Bishop, Rook}) {
      const Bitboard mask = internal::sliderAttackTables[square][piece - Bishop].maxAttackNoEdge;
      for (Bitboard occupancy = mask;; occupancy = (occupancy - 1) & mask) {
        const Bitboard actual = piece == Bishop ? getAttack<Bishop>(square, occupancy)
                                                  : getAttack<Rook>(square, occupancy);
        EXPECT_EQ(actual, slowSliderAttack(piece, square, occupancy));
        if (occupancy == 0) {
          break;
        }
      }
    }
  }
}

TEST(AttackTables, CornerLeapersDoNotWrapFiles) {
  EXPECT_EQ(getAttack<Knight>(A8), toBitboard(B6, C7));
  EXPECT_EQ(getAttack<Knight>(H1), toBitboard(F2, G3));
  EXPECT_EQ(getAttack<King>(A8), toBitboard(A7, B7, B8));
  EXPECT_EQ((getAttack<Pawn, White>(A7)), toBitboard(B8));
  EXPECT_EQ((getAttack<Pawn, Black>(H2)), toBitboard(G1));
}

TEST(LegalMoves, InitialPosition) {
  const auto state = BoardState{fen::kStartPosition};
  const auto& moves = legalMoves<White>(state);

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
  const auto state = BoardState{"4r2k/8/8/8/8/8/8/4K3 w - - 0 1"};
  const auto& moves = legalMoves<White>(state);

  EXPECT_EQ(moves.size(), 4u);
  EXPECT_TRUE(std::all_of(moves.begin(), moves.end(), [](const LegalMove& move) {
    return move.piece == King;
  }));
}

TEST(LegalMoves, PinnedKnightCannotMove) {
  const auto state = BoardState{"4r2k/8/8/8/8/8/4N3/4K3 w - - 0 1"};
  const auto& moves = legalMoves<White>(state);

  EXPECT_TRUE(std::none_of(moves.begin(), moves.end(), [](const LegalMove& move) {
    return move.srce == E2;
  }));
}

TEST(LegalMoves, CastlingMovesRookAndClearsRights) {
  const auto state = BoardState{"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"};
  const auto& moves = legalMoves<White>(state);

  EXPECT_EQ(countMoves(moves, E1, G1), 1u);
  EXPECT_EQ(countMoves(moves, E1, C1), 1u);
  auto afterCastle = state;
  afterCastle.makeMove<White>(Move(E1, G1, King, NoPiece, Move::kCastlingFlag), testing_support::NoOpMoveCallback{});
  EXPECT_EQ(afterCastle.getPieceAt(G1), (std::tuple<Side, Piece>{White, King}));
  EXPECT_EQ(afterCastle.getPieceAt(F1), (std::tuple<Side, Piece>{White, Rook}));
  EXPECT_FALSE(afterCastle.getPieceAt(H1).has_value());
  EXPECT_EQ(afterCastle.getCastlingRights(), BlackKingCastle | BlackQueenCastle);
}

TEST(LegalMoves, CannotCastleThroughAttack) {
  const auto state = BoardState{"4k3/8/8/8/8/5r2/8/R3K2R w KQ - 0 1"};
  const auto& moves = legalMoves<White>(state);

  EXPECT_EQ(countMoves(moves, E1, G1), 0u);
  EXPECT_EQ(countMoves(moves, E1, C1), 1u);
}

TEST(LegalMoves, CannotCastleOutOfCheck) {
  const auto state = BoardState{"4r2k/8/8/8/8/8/8/R3K2R w KQ - 0 1"};
  const auto& moves = legalMoves<White>(state);

  EXPECT_EQ(countMoves(moves, E1, G1), 0u);
  EXPECT_EQ(countMoves(moves, E1, C1), 0u);
}

TEST(LegalMoves, BlackCanCastleOnBothSides) {
  const auto state = BoardState{"r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1"};
  const auto& moves = legalMoves<Black>(state);

  EXPECT_EQ(countMoves(moves, E8, G8), 1u);
  EXPECT_EQ(countMoves(moves, E8, C8), 1u);
}

TEST(LegalMoves, QueenSideCastleMayPassAttackedRookSquare) {
  const auto state = BoardState{"1r5k/8/8/8/8/8/8/R3K2R w KQ - 0 1"};
  const auto& moves = legalMoves<White>(state);

  EXPECT_EQ(countMoves(moves, E1, C1), 1u);
}

TEST(LegalMoves, PinnedEnemyKnightStillAttacksCastlingPath) {
  const auto state = BoardState{"6k1/8/8/8/8/6n1/6Q1/4K2R w K - 0 1"};
  const auto& moves = legalMoves<White>(state);

  EXPECT_EQ(countMoves(moves, E1, G1), 0u);
}

TEST(BoardState, MovingRookPermanentlyRemovesItsCastlingRight) {
  const auto state = BoardState{"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"};
  auto afterRookReturns = state;
  afterRookReturns.makeMove<White>(Move(H1, H2, Rook), testing_support::NoOpMoveCallback{});
  afterRookReturns.makeMove<Black>(Move(E8, E7, King), testing_support::NoOpMoveCallback{});
  afterRookReturns.makeMove<White>(Move(H2, H1, Rook), testing_support::NoOpMoveCallback{});
  afterRookReturns.makeMove<Black>(Move(E7, E8, King), testing_support::NoOpMoveCallback{});
  const auto& moves = legalMoves<White>(afterRookReturns);

  EXPECT_EQ(countMoves(moves, E1, G1), 0u);
  EXPECT_EQ(countMoves(moves, E1, C1), 1u);
  EXPECT_EQ(afterRookReturns.getCastlingRights(), WhiteQueenCastle);
}

TEST(BoardState, CapturingCornerRookRemovesCastlingRight) {
  const auto state = BoardState{"r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"};
  auto afterCapture = state;
  afterCapture.makeMove<Black>(Move(A8, A1, Rook, NoPiece, Move::kCaptureFlag), testing_support::NoOpMoveCallback{});

  EXPECT_EQ(afterCapture.getCastlingRights(), WhiteKingCastle | BlackKingCastle);
}

TEST(LegalMoves, PawnHasFourPromotionChoices) {
  const auto state = BoardState{"7k/P7/8/8/8/8/8/7K w - - 0 1"};
  const auto& moves = legalMoves<White>(state);

  EXPECT_EQ(countMoves(moves, A7, A8), 4u);
  auto promoted = state;
  promoted.makeMove<White>(Move(A7, A8, Pawn, Queen), testing_support::NoOpMoveCallback{});
  EXPECT_FALSE(promoted.getPieceAt(A7).has_value());
  EXPECT_EQ(promoted.getPieceAt(A8), (std::tuple<Side, Piece>{White, Queen}));
}

TEST(BoardState, CaptureRemovesOpponentPiece) {
  const auto state = BoardState{"7k/8/8/8/8/8/4p3/4R2K w - - 17 1"};
  auto afterCapture = state;
  afterCapture.makeMove<White>(Move(E1, E2, Rook, NoPiece, Move::kCaptureFlag), testing_support::NoOpMoveCallback{});

  EXPECT_EQ(afterCapture.getPieceAt(E2), (std::tuple<Side, Piece>{White, Rook}));
  EXPECT_FALSE(afterCapture.getPieceAt(E1).has_value());
  EXPECT_EQ(afterCapture.getHalfmoveClock(), 0);
}

TEST(BoardState, EnPassantCaptureExpiresImmediately) {
  const auto state = BoardState{"7k/2pp4/8/4P3/8/8/8/7K b - - 0 1"};
  auto afterPush = state;
  afterPush.makeMove<Black>(Move(D7, D5, Pawn, NoPiece, Move::kDoublePushFlag), testing_support::NoOpMoveCallback{});
  EXPECT_EQ(afterPush.getEnpassantSquare(), D6);

  const auto& whiteMoves = legalMoves<White>(afterPush);
  const auto capture = std::find_if(whiteMoves.begin(), whiteMoves.end(), [](const LegalMove& move) {
    return move.srce == E5 && move.dest == D6 && move.enpassant;
  });
  ASSERT_NE(capture, whiteMoves.end());

  auto afterCapture = afterPush;
  afterCapture.makeMove<White>(Move(E5, D6, Pawn, NoPiece, Move::kEnpassantFlag), testing_support::NoOpMoveCallback{});
  EXPECT_EQ(afterCapture.getEnpassantSquare(), NoSquare);
  EXPECT_EQ(afterCapture.getPieceAt(D6), (std::tuple<Side, Piece>{White, Pawn}));
  EXPECT_FALSE(afterCapture.getPieceAt(D5).has_value());

  const auto& blackMoves = legalMoves<Black>(afterCapture);
  EXPECT_EQ(countMoves(blackMoves, C7, D6), 1u);
  EXPECT_TRUE(std::none_of(blackMoves.begin(), blackMoves.end(), [](const LegalMove& move) {
    return move.enpassant;
  }));
}

TEST(BoardState, EnPassantExpiresAfterQuietMove) {
  const auto state = BoardState{"7k/3p4/8/4P3/8/8/8/7K b - - 0 1"};
  auto afterQuietMove = state;
  afterQuietMove.makeMove<Black>(Move(D7, D5, Pawn, NoPiece, Move::kDoublePushFlag), testing_support::NoOpMoveCallback{});
  afterQuietMove.makeMove<White>(Move(H1, G1, King), testing_support::NoOpMoveCallback{});

  EXPECT_EQ(afterQuietMove.getEnpassantSquare(), NoSquare);
  EXPECT_EQ(afterQuietMove.getFullmoveNumber(), 2);
}

TEST(LegalMoves, DoubleCheckAllowsOnlyKingMoves) {
  const auto state = BoardState{"7k/8/8/3p1p2/4K3/8/8/R7 w - - 0 1"};
  const auto& moves = legalMoves<White>(state);

  EXPECT_FALSE(moves.empty());
  EXPECT_TRUE(std::all_of(moves.begin(), moves.end(), [](const LegalMove& move) {
    return move.piece == King;
  }));
}

TEST(LegalMoves, KingCannotRetreatIntoOpenedRookRay) {
  const auto state = BoardState{"4r2k/8/8/8/8/8/4K3/8 w - - 0 1"};
  const auto& moves = legalMoves<White>(state);

  EXPECT_EQ(countMoves(moves, E2, E1), 0u);
}

TEST(LegalMoves, KingCannotMoveNextToEnemyKing) {
  const auto state = BoardState{"8/8/8/8/8/8/6k1/4K3 w - - 0 1"};
  const auto& moves = legalMoves<White>(state);

  EXPECT_EQ(countMoves(moves, E1, F1), 0u);
  EXPECT_EQ(countMoves(moves, E1, D1), 1u);
}

TEST(LegalMoves, PinnedRookCanMoveOnlyOnPinLine) {
  const auto state = BoardState{"4r2k/8/8/8/8/8/4R3/4K3 w - - 0 1"};
  const auto& moves = legalMoves<White>(state);

  EXPECT_EQ(countMoves(moves, E2, E8), 1u);
  EXPECT_EQ(countMoves(moves, E2, D2), 0u);
  EXPECT_EQ(countMoves(moves, E2, F2), 0u);
}

TEST(LegalMoves, DiagonallyPinnedPawnCanCapturePinner) {
  const auto state = BoardState{"7k/8/8/8/8/4b3/3P4/2K5 w - - 0 1"};
  const auto& moves = legalMoves<White>(state);

  EXPECT_EQ(countMoves(moves, D2, E3), 1u);
  EXPECT_EQ(countMoves(moves, D2, D3), 0u);
  EXPECT_EQ(countMoves(moves, D2, D4), 0u);
}

TEST(LegalMoves, DoublePawnPushCanBlockDiagonalCheck) {
  const auto state = BoardState{"7k/8/1b6/8/8/8/3P4/6K1 w - - 0 1"};
  const auto& moves = legalMoves<White>(state);

  EXPECT_EQ(countMoves(moves, D2, D4), 1u);
  EXPECT_EQ(countMoves(moves, D2, D3), 0u);
}

TEST(LegalMoves, PawnCannotJumpOverOccupiedSquare) {
  const auto state = BoardState{"7k/8/8/8/8/4n3/4P3/7K w - - 0 1"};
  const auto& moves = legalMoves<White>(state);

  EXPECT_EQ(countMoves(moves, E2, E3), 0u);
  EXPECT_EQ(countMoves(moves, E2, E4), 0u);
}

TEST(LegalMoves, PawnDoublePushNeedsEmptyDestination) {
  const auto state = BoardState{"7k/8/8/8/4n3/8/4P3/7K w - - 0 1"};
  const auto& moves = legalMoves<White>(state);

  EXPECT_EQ(countMoves(moves, E2, E3), 1u);
  EXPECT_EQ(countMoves(moves, E2, E4), 0u);
}

TEST(LegalMoves, EdgePawnsDoNotWrapFiles) {
  const auto state = BoardState{"7k/8/8/P6P/8/8/8/K7 w - - 0 1"};
  const auto& moves = legalMoves<White>(state);

  EXPECT_EQ(countMoves(moves, A5, A6), 1u);
  EXPECT_EQ(countMoves(moves, H5, H6), 1u);
  EXPECT_TRUE(std::all_of(moves.begin(), moves.end(), [](const LegalMove& move) {
    return move.piece != Pawn || (move.srce == A5 && move.dest == A6) ||
           (move.srce == H5 && move.dest == H6);
  }));
}

TEST(LegalMoves, CapturePromotionHasFourChoices) {
  const auto state = BoardState{"1r5k/P7/8/8/8/8/8/7K w - - 0 1"};
  const auto& moves = legalMoves<White>(state);

  EXPECT_EQ(countMoves(moves, A7, A8), 4u);
  EXPECT_EQ(countMoves(moves, A7, B8), 4u);
}

TEST(LegalMoves, BlackCapturePromotionHasFourChoices) {
  const auto state = BoardState{"7k/8/8/8/8/8/p7/1R5K b - - 0 1"};
  const auto& moves = legalMoves<Black>(state);

  EXPECT_EQ(countMoves(moves, A2, A1), 4u);
  EXPECT_EQ(countMoves(moves, A2, B1), 4u);
}

TEST(LegalMoves, WhiteEnPassantCanCaptureCheckingPawn) {
  const auto state = BoardState{"7k/8/8/3pP3/4K3/8/8/8 w - d6 0 1"};
  const auto& moves = legalMoves<White>(state);

  EXPECT_EQ(countMoves(moves, E5, D6), 1u);
  EXPECT_TRUE(std::any_of(moves.begin(), moves.end(), [](const LegalMove& move) {
    return move.srce == E5 && move.dest == D6 && move.enpassant;
  }));
}

TEST(LegalMoves, WhiteEnPassantCanBlockBishopCheck) {
  const auto state = BoardState{"5b1k/8/8/3pP3/1K6/8/8/8 w - d6 0 1"};
  const auto& moves = legalMoves<White>(state);

  EXPECT_EQ(countMoves(moves, E5, D6), 1u);
}

TEST(LegalMoves, EnPassantCannotExposeHorizontalRook) {
  const auto state = BoardState{"7k/8/8/r1PpK3/8/8/8/8 w - d6 0 1"};
  const auto& moves = legalMoves<White>(state);

  EXPECT_EQ(countMoves(moves, C5, D6), 0u);
}

TEST(LegalMoves, EnPassantCannotExposeDiagonalBishop) {
  const auto state = BoardState{"7b/8/8/3pP3/8/8/8/K6k w - d6 0 1"};
  const auto& moves = legalMoves<White>(state);

  EXPECT_EQ(countMoves(moves, E5, D6), 0u);
}

TEST(LegalMoves, BlackEnPassantIsGenerated) {
  const auto state = BoardState{"7k/8/8/8/3Pp3/8/8/7K b - d3 0 1"};
  const auto& moves = legalMoves<Black>(state);

  EXPECT_EQ(countMoves(moves, E4, D3), 1u);
  EXPECT_TRUE(std::any_of(moves.begin(), moves.end(), [](const LegalMove& move) {
    return move.srce == E4 && move.dest == D3 && move.enpassant;
  }));
}

TEST(LegalMoves, BlackEnPassantCanCaptureCheckingPawn) {
  const auto state = BoardState{"8/8/8/4k3/3Pp3/8/8/7K b - d3 0 1"};
  const auto& moves = legalMoves<Black>(state);

  EXPECT_EQ(countMoves(moves, E4, D3), 1u);
  EXPECT_TRUE(std::any_of(moves.begin(), moves.end(), [](const LegalMove& move) {
    return move.srce == E4 && move.dest == D3 && move.enpassant;
  }));
}

TEST(LegalMoves, KingCannotCaptureDefendedCheckingPiece) {
  const auto state = BoardState{"7k/8/8/8/8/6n1/4r3/4K3 w - - 0 1"};
  const auto& moves = legalMoves<White>(state);

  EXPECT_EQ(countMoves(moves, E1, E2), 0u);
}

TEST(LegalMoves, CheckmateAndStalemateHaveNoMoves) {
  const auto mate = BoardState{"7k/6Q1/5K2/8/8/8/8/8 b - - 0 1"};
  const auto stalemate = BoardState{"7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"};

  EXPECT_TRUE(legalMoves<Black>(mate).empty());
  EXPECT_TRUE(legalMoves<Black>(stalemate).empty());
}

TEST(BoardState, EveryLegalMoveMakesAndUnmakesExactly) {
  // source: https://www.chessprogramming.org/Perft_Results
  const std::array<std::string_view, 6> positionsToCheck{{
    fen::kStartPosition,
    fen::kKiwipete,
    fen::kRookEndgame,
    fen::kMirrorView,
    fen::kTalkChessBug,
    fen::kStevenAlt
  }};
  for (const std::string_view fen : positionsToCheck) {
    SCOPED_TRACE(std::string(fen));
    BoardState state{fen};
    const BoardState original = state;
    MoveList moves;
    if (state.getSideToMove() == White) {
      state.generateMoves<White>(moves);
    } else {
      state.generateMoves<Black>(moves);
    }
    for (const Move& move : moves) {
      SCOPED_TRACE(std::to_string(move.getSource()) + " to " + std::to_string(move.getDest()));
      if (state.getSideToMove() == White) {
        const auto undo = state.makeMove<White>(move, testing_support::NoOpMoveCallback{});
        EXPECT_EQ(state.getSideToMove(), Black);
        state.unmakeMove<White>(move, undo, testing_support::NoOpMoveCallback{});
      } else {
        const auto undo = state.makeMove<Black>(move, testing_support::NoOpMoveCallback{});
        EXPECT_EQ(state.getSideToMove(), White);
        state.unmakeMove<Black>(move, undo, testing_support::NoOpMoveCallback{});
      }
      for (Square square = 0; square < kSquareSize; ++square) {
        EXPECT_EQ(state.getPieceAt(square), original.getPieceAt(square));
      }
      EXPECT_EQ(state.getSideToMove(), original.getSideToMove());
      EXPECT_EQ(state.getCastlingRights(), original.getCastlingRights());
      EXPECT_EQ(state.getEnpassantSquare(), original.getEnpassantSquare());
      EXPECT_EQ(state.getHalfmoveClock(), original.getHalfmoveClock());
      EXPECT_EQ(state.getFullmoveNumber(), original.getFullmoveNumber());
    }
  }
}

TEST(BoardState, BlackMoveAndEnPassantRestoreColor) {
  BoardState black{"7k/8/8/8/8/8/8/7K b - - 0 1"};
  const auto blackUndo = black.makeMove<Black>(Move(H8, G8, King), testing_support::NoOpMoveCallback{});
  EXPECT_EQ(black.getSideToMove(), White);
  black.unmakeMove<Black>(Move(H8, G8, King), blackUndo, testing_support::NoOpMoveCallback{});
  EXPECT_EQ(black.getSideToMove(), Black);

  BoardState enpassant{"7k/8/8/3pP3/8/8/8/7K w - d6 0 1"};
  const Move capture(E5, D6, Pawn, NoPiece, Move::kEnpassantFlag);
  const auto undo = enpassant.makeMove<White>(capture, testing_support::NoOpMoveCallback{});
  EXPECT_EQ(enpassant.getSideToMove(), Black);
  enpassant.unmakeMove<White>(capture, undo, testing_support::NoOpMoveCallback{});
  EXPECT_EQ(enpassant.getSideToMove(), White);
}
