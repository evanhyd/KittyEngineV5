#include "board.h"
#include "equal_percentage_time_control_policy.h"
#include "handcraft_evaluation_policy.h"
#include "negamax_search_policy.h"
#include "notation.h"
#include "position_fens.h"
#include "terminal_ui.h"
#include "test_move_callback.h"
#include "uci_protocol.h"
#include <array>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <gtest/gtest.h>

using namespace bb;

namespace {
  using UciSearch = searching::NegamaxSearchPolicy<evaluation::HandCraftEvaluationPolicy>;
  using TestTimePolicy = time_control::EqualPercentageTimeControlPolicy;
  using UciBoard = Board<UciSearch, TestTimePolicy>;
  constexpr int32_t kMateMagnitude = -evaluation::kCheckmateScore;
  constexpr int32_t kMateWindow = 100;
  constexpr size_t kTestTranspositionEntries = 1024;
  constexpr float kTestTimePercentage = 0.05f;

  struct RecordingSearchPolicy {
    int* requestedDepth;
    std::vector<searching::SearchResult>* results;
    std::vector<std::vector<Move>>* pvSnapshots = nullptr;
    int* resetCalls = nullptr;
    std::vector<size_t>* historySizes = nullptr;
    std::vector<size_t>* incomingPVSizes = nullptr;

    void reset() {
      if (resetCalls) {
        ++*resetCalls;
      }
    }

    void invalidateEvaluation() {}
    template <bool Add>
    void markPiece(Side, Piece, Square) {}
    void markCastle(CastlePermission) {}
    void markEnpassant(Square) {}

    template <Side ally>
    searching::SearchResult search(BoardState& state, const searching::SearchParam& param) {
      *requestedDepth = param.maxDepth;
      if (historySizes) {
        historySizes->push_back(param.positionHistory.size());
      }
      if (incomingPVSizes) {
        incomingPVSizes->push_back(param.pvLine.size());
      }
      MoveList moves;
      state.generateMoves<ally>(moves);
      searching::PVLine& pvLine = param.pvLine;
      pvLine.clear();
      if (!moves.empty()) {
        pvLine.push(moves[0]);
        if (param.maxDepth > 1) {
          BoardState nextState = state;
          nextState.makeMove<ally>(moves[0], testing_support::NoOpMoveCallback{});
          MoveList replies;
          nextState.generateMoves<getOtherSide(ally)>(replies);
          if (!replies.empty()) {
            pvLine.push(replies[0]);
          }
        }
      }
      const uint64_t nodes = param.maxDepth == 3 ? 100 : param.maxDepth == 2 ? 30 : 10;
      const int32_t score = param.maxDepth == 3 ? kMateMagnitude - 3 : param.maxDepth == 2 ? -12 : 34;
      searching::SearchResult result{
        score, moves.empty() ? std::nullopt : std::optional<Move>{moves[0]},
        nodes, std::chrono::milliseconds{10 * param.maxDepth}, &pvLine};
      if (pvSnapshots) {
        pvSnapshots->emplace_back(pvLine.begin(), pvLine.end());
      }
      results->push_back(result);
      return result;
    }
  };

  struct ZeroTimeSearchPolicy {
    void reset() {}
    void invalidateEvaluation() {}
    template <bool Add>
    void markPiece(Side, Piece, Square) {}
    void markCastle(CastlePermission) {}
    void markEnpassant(Square) {}

    template <Side ally>
    searching::SearchResult search(BoardState&, const searching::SearchParam& param) {
      param.pvLine.clear();
      return {0, std::nullopt, 1, std::chrono::nanoseconds{}, &param.pvLine};
    }
  };

  struct SlowSearchPolicy {
    int* requestedDepth;
    std::chrono::milliseconds pause;

    void reset() {}
    void invalidateEvaluation() {}
    template <bool Add>
    void markPiece(Side, Piece, Square) {}
    void markCastle(CastlePermission) {}
    void markEnpassant(Square) {}

    template <Side ally>
    searching::SearchResult search(BoardState& state, const searching::SearchParam& param) {
      *requestedDepth = param.maxDepth;
      std::this_thread::sleep_for(pause);
      MoveList moves;
      state.generateMoves<ally>(moves);
      param.pvLine.clear();
      return {0, moves.empty() ? std::nullopt : std::optional<Move>{moves[0]}, 1, pause, &param.pvLine};
    }
  };

  struct SubMillisecondSearchPolicy {
    static constexpr uint64_t kNodes = 100;
    static constexpr auto kDuration = std::chrono::microseconds{250};

    void reset() {}
    void invalidateEvaluation() {}
    template <bool Add>
    void markPiece(Side, Piece, Square) {}
    void markCastle(CastlePermission) {}
    void markEnpassant(Square) {}

    template <Side ally>
    searching::SearchResult search(BoardState&, const searching::SearchParam& param) {
      param.pvLine.clear();
      return {0, std::nullopt, kNodes, kDuration, &param.pvLine};
    }
  };

  struct ScoreSequenceSearchPolicy {
    static constexpr uint64_t kNodesPerDepth = 1;
    static constexpr auto kSearchDuration = std::chrono::milliseconds{1};
    const std::vector<int32_t>* scores;

    void reset() {}
    void invalidateEvaluation() {}
    template <bool Add>
    void markPiece(Side, Piece, Square) {}
    void markCastle(CastlePermission) {}
    void markEnpassant(Square) {}

    template <Side ally>
    searching::SearchResult search(BoardState& state, const searching::SearchParam param) {
      MoveList moves;
      state.generateMoves<ally>(moves);
      param.pvLine.clear();
      return {scores->at(param.maxDepth - 1), moves.empty() ? std::nullopt : std::optional<Move>{moves[0]},
              kNodesPerDepth, kSearchDuration, &param.pvLine};
    }
  };

  struct PVTrackingSearchPolicy {
    std::vector<std::vector<Move>>* incomingLines;
    bool seeded = false;
    size_t seedLength = 2;

    void reset() { seeded = false; }
    void invalidateEvaluation() {}
    template <bool Add>
    void markPiece(Side, Piece, Square) {}
    void markCastle(CastlePermission) {}
    void markEnpassant(Square) {}

    template <Side ally>
    searching::SearchResult search(BoardState& state, const searching::SearchParam& param) {
      incomingLines->emplace_back(param.pvLine.begin(), param.pvLine.end());
      searching::PVLine& pvLine = param.pvLine;
      MoveList moves;
      state.generateMoves<ally>(moves);
      if (!seeded) {
        BoardState nextState = state;
        Side side = ally;
        for (size_t i = 0; i < seedLength; ++i) {
          MoveList legalMoves;
          if (side == White) nextState.generateMoves<White>(legalMoves);
          else nextState.generateMoves<Black>(legalMoves);
          if (legalMoves.empty()) {
            break;
          }
          pvLine.push(legalMoves[0]);
          if (side == White) {
            nextState.makeMove<White>(legalMoves[0], testing_support::NoOpMoveCallback{});
          } else {
            nextState.makeMove<Black>(legalMoves[0], testing_support::NoOpMoveCallback{});
          }
          side = getOtherSide(side);
        }
        seeded = true;
      }
      return {0, moves.empty() ? std::nullopt : std::optional<Move>{moves[0]},
              1, std::chrono::nanoseconds{}, &pvLine};
    }
  };

  struct ParsedInfo {
    int depth;
    std::string scoreKind;
    int64_t score;
    int64_t timeMs;
    uint64_t nodes;
    uint64_t nps;
    uint64_t tablebaseHits = 0;
    std::vector<std::string> pv;
  };

  std::optional<ParsedInfo> parseInfoLine(const std::string& line) {
    std::istringstream stream{line};
    std::string info, depth, score, time, nodes, nps, extra;
    ParsedInfo parsed{};
    if (!(stream >> info >> depth >> parsed.depth >> score >> parsed.scoreKind >> parsed.score
                 >> time >> parsed.timeMs >> nodes >> parsed.nodes >> nps >> parsed.nps)) {
      return std::nullopt;
    }
    if (info != "info" || depth != "depth" || score != "score" ||
        time != "time" || nodes != "nodes" || nps != "nps") {
      return std::nullopt;
    }
    if (stream >> extra) {
      if (extra == "tbhits") {
        if (!(stream >> parsed.tablebaseHits)) {
          return std::nullopt;
        }
        if (!(stream >> extra)) {
          return parsed;
        }
      }
      if (extra != "pv") {
        return std::nullopt;
      }
      while (stream >> extra) {
        parsed.pv.push_back(extra);
      }
      if (parsed.pv.empty()) {
        return std::nullopt;
      }
    }
    return parsed;
  }
}

TEST(UciProtocol, DispatchesConstructorCallbacksWithoutBoard) {
  std::string calls;
  bool quit = false;
  uci::UciProtocol protocol{
    [&] { calls += 'u'; },
    [&] { calls += 'r'; },
    [&] { calls += 'n'; },
    [](std::string_view, std::span<const std::string_view>) {},
    [](std::span<const std::string_view>) {},
    [&] { quit = true; }
  };

  protocol.send(" unknown command\r");
  protocol.send("uci");
  protocol.send("  isready \r");
  protocol.send("ucinewgame");
  protocol.send("quit");
  EXPECT_EQ(calls, "urn");
  EXPECT_TRUE(quit);
}

TEST(UciProtocol, ParsesBothPositionFormsAndMoveLists) {
  std::vector<std::string> positions;
  uci::UciProtocol protocol{
    [] {},
    [] {},
    [] {},
    [&](std::string_view fen, std::span<const std::string_view> moves) {
      std::string received{fen};
      for (const auto move : moves) {
        received += " " + std::string(move);
      }
      positions.push_back(std::move(received));
    },
    [](std::span<const std::string_view>) {},
    [] {}
  };
  protocol.send("position startpos moves e2e4 e7e5");
  protocol.send("position fen 7k/8/8/8/8/8/8/7K b - - 4 12 moves h8g8");
  ASSERT_EQ(positions.size(), 2u);
  EXPECT_EQ(positions[0], std::string(fen::kStartPosition) + " e2e4 e7e5");
  EXPECT_EQ(positions[1], "7k/8/8/8/8/8/8/7K b - - 4 12 h8g8");
}

TEST(UciProtocol, PropagatesErrorsToItsCaller) {
  int readyCount = 0;
  uci::UciProtocol protocol{
    [] {},
    [&] { ++readyCount; },
    [] {},
    [](std::string_view, std::span<const std::string_view>) {},
    [](std::span<const std::string_view>) {
      throw std::logic_error("search is unfinished");
    },
    [] {}
  };

  EXPECT_THROW(protocol.send("position fen 8/8/8/8/8/8/8/8 w - -"), std::invalid_argument);
  EXPECT_THROW(protocol.send("go depth 1"), std::logic_error);
  protocol.send("isready");
  EXPECT_EQ(readyCount, 1);
}

TEST(UciProtocol, DispatchesPerftAndRejectsInvalidDepth) {
  std::vector<std::pair<uint32_t, bool>> requests;
  uci::UciProtocol protocol{
    [] {}, [] {}, [] {},
    [](std::string_view, std::span<const std::string_view>) {},
    [](std::span<const std::string_view>) {},
    [] {},
    {},
    [&](uint32_t depth, bool detail) { requests.emplace_back(depth, detail); }
  };

  protocol.send("perft depth 0");
  protocol.send("perft depth 3 detail");
  EXPECT_EQ(requests, (std::vector<std::pair<uint32_t, bool>>{{0, false}, {3, true}}));
  EXPECT_THROW(protocol.send("perft"), std::invalid_argument);
  EXPECT_THROW(protocol.send("perft depth"), std::invalid_argument);
  EXPECT_THROW(protocol.send("perft 3"), std::invalid_argument);
  EXPECT_THROW(protocol.send("perft depth -1"), std::invalid_argument);
  EXPECT_THROW(protocol.send("perft depth nope"), std::invalid_argument);
  EXPECT_THROW(protocol.send("perft depth 4294967296"), std::invalid_argument);
  EXPECT_THROW(protocol.send("perft depth 1 extra"), std::invalid_argument);
  EXPECT_THROW(protocol.send("perft depth 1 detail extra"), std::invalid_argument);
  EXPECT_EQ(requests.size(), 2u);
}

TEST(UciIntegration, KeepsHumanViewOffProtocolOutput) {
  std::istringstream input{"uci\nisready\nquit\nuci\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{UciSearch{evaluation::HandCraftEvaluationPolicy{}, 50, kTestTranspositionEntries}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(uciOutput.str(), "id name KittyEngineV5\nid author UnboxTheCat\nuciok\nreadyok\n");
  EXPECT_NE(humanOutput.str().find("Welcome to KittyEngineV5"), std::string::npos);
  EXPECT_NE(humanOutput.str().find("FEN: " + std::string(fen::kStartPosition)), std::string::npos);
  EXPECT_EQ(humanOutput.str().find("uciok"), std::string::npos);
}

TEST(UciIntegration, RunsFastAndDetailedPerftOnCurrentPosition) {
  std::istringstream input{
    "position fen " + std::string(fen::kKiwipete) + "\n"
    "perft depth 0\n"
    "perft depth 1\n"
    "perft depth 1 detail\n"
    "quit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{UciSearch{evaluation::HandCraftEvaluationPolicy{}, 50, kTestTranspositionEntries}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  std::istringstream output{uciOutput.str()};
  std::string line;
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(line.find("depth 0, nodes 1, time "), 0u);
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(line.find("depth 1, nodes 48, time "), 0u);
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(line.find("depth 1, nodes 48, time "), 0u);
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(line, "    captures 8 enpassants 0 castles 2 promotions 0");
  EXPECT_FALSE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(notation::boardToFen(board.getState()), fen::kKiwipete);
  EXPECT_EQ(humanOutput.str().find("depth 1, nodes"), std::string::npos);
}

TEST(SearchStatistics, CountsRootAndLeafPositions) {
  UciBoard board{UciSearch{evaluation::HandCraftEvaluationPolicy{}, 32'000, kTestTranspositionEntries}, TestTimePolicy{kTestTimePercentage}};
  MoveList legalMoves;
  board.getState().generateMoves<White>(legalMoves);

  const auto result = board.search(1, std::nullopt, [](const searching::SearchResult&) {});
  EXPECT_EQ(result.nodesSearched, legalMoves.size() + 1);
}

TEST(SearchStatistics, ReusesRootTranspositionWithNanosecondDuration) {
  UciBoard board{UciSearch{evaluation::HandCraftEvaluationPolicy{}, 50, kTestTranspositionEntries}, TestTimePolicy{kTestTimePercentage}};
  const auto first = board.search(1, std::nullopt, [](const searching::SearchResult&) {});
  const auto cached = board.search(1, std::nullopt, [](const searching::SearchResult&) {});

  ASSERT_TRUE(first.bestMove.has_value());
  ASSERT_TRUE(cached.bestMove.has_value());
  EXPECT_EQ(cached.score, first.score);
  EXPECT_EQ(notation::moveToString(*cached.bestMove), notation::moveToString(*first.bestMove));
  EXPECT_EQ(cached.nodesSearched, 1u);
  EXPECT_GE(cached.searchingTime.count(), 0);
}

TEST(UciIntegration, PassesGoDepthToSearch) {
  std::istringstream input{"go depth 3\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  int requestedDepth = 0;
  std::vector<searching::SearchResult> results;
  std::vector<std::vector<Move>> pvSnapshots;
  Board board{RecordingSearchPolicy{&requestedDepth, &results, &pvSnapshots}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(requestedDepth, 3);
  ASSERT_EQ(results.size(), 3u);
  std::istringstream output{uciOutput.str()};
  std::string line;
  uint64_t totalNodes = 0;
  int64_t previousTimeMs = 0;
  for (size_t i = 0; i < results.size(); ++i) {
    ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
    const auto info = parseInfoLine(line);
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->depth, i + 1);
    totalNodes += results[i].nodesSearched;
    EXPECT_EQ(info->nodes, totalNodes);
    EXPECT_GE(info->timeMs, previousTimeMs);
    if (info->timeMs > 0 && info->timeMs < 1000) {
      EXPECT_GT(info->nps, info->nodes);
    }
    previousTimeMs = info->timeMs;
    EXPECT_EQ(info->scoreKind, i + 1 == results.size() ? "mate" : "cp");
    if (info->scoreKind == "cp") {
      EXPECT_EQ(info->score, results[i].score);
    }
    const auto& expectedPV = pvSnapshots[i];
    ASSERT_EQ(info->pv.size(), expectedPV.size());
    for (size_t moveIndex = 0; moveIndex < expectedPV.size(); ++moveIndex) {
      EXPECT_EQ(info->pv[moveIndex], notation::moveToString(expectedPV[moveIndex]));
    }
  }
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  ASSERT_TRUE(results.back().bestMove.has_value());
  EXPECT_EQ(line, "bestmove " + notation::moveToString(*results.back().bestMove));
  EXPECT_FALSE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(notation::boardToFen(board.getState()), fen::kStartPosition);
}

TEST(UciIntegration, StopsTimedSearchAfterFirstCompletedDepthWhenBudgetIsZero) {
  std::istringstream input{"go wtime 0 btime 10000 winc 0 binc 1000 movestogo 9999\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  int requestedDepth = 0;
  std::vector<searching::SearchResult> results;
  Board board{RecordingSearchPolicy{&requestedDepth, &results}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(requestedDepth, 1);
  ASSERT_EQ(results.size(), 1u);
  std::istringstream output{uciOutput.str()};
  std::string line;
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  const auto info = parseInfoLine(line);
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->depth, 1);
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  ASSERT_TRUE(results.back().bestMove.has_value());
  EXPECT_EQ(line, "bestmove " + notation::moveToString(*results.back().bestMove));
  EXPECT_FALSE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(notation::boardToFen(board.getState()), fen::kStartPosition);
}

TEST(UciIntegration, CapsRequestedSearchDepth) {
  const int depthBeyondLimit = searching::kMaxDepthHardCutoff + 1;
  std::istringstream input{"go depth " + std::to_string(depthBeyondLimit) + "\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  int requestedDepth = 0;
  std::vector<searching::SearchResult> results;
  Board board{RecordingSearchPolicy{&requestedDepth, &results}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();

  EXPECT_EQ(requestedDepth, searching::kMaxDepthHardCutoff);
  ASSERT_EQ(results.size(), static_cast<size_t>(searching::kMaxDepthHardCutoff));
  EXPECT_EQ(uciOutput.str().find("info string error:"), std::string::npos);
  EXPECT_NE(uciOutput.str().find("bestmove "), std::string::npos);
}

TEST(BoardPV, DirectMovesConsumeMatchingPrefixAndClearOnDivergence) {
  std::vector<std::vector<Move>> incomingLines;
  Board board{PVTrackingSearchPolicy{&incomingLines}, TestTimePolicy{kTestTimePercentage}};
  const auto search = [&] {
    return board.search(1, std::nullopt, [](const searching::SearchResult&) {});
  };

  const auto first = search();
  const searching::PVLine firstPV = *first.pvLine;
  ASSERT_EQ(firstPV.size(), 2u);
  board.playMove(notation::moveToString(firstPV[0]));
  ASSERT_EQ(first.pvLine->size(), 1u);
  EXPECT_EQ(first.pvLine->front(), firstPV[1]);
  search();
  ASSERT_EQ(incomingLines.back().size(), 1u);
  EXPECT_EQ(incomingLines.back()[0], firstPV[1]);

  board.playMove(notation::moveToString(firstPV[1]));
  search();
  EXPECT_TRUE(incomingLines.back().empty());

  std::vector<std::vector<Move>> otherIncomingLines;
  Board otherBoard{PVTrackingSearchPolicy{&otherIncomingLines}, TestTimePolicy{kTestTimePercentage}};
  const auto otherResult = otherBoard.search(1, std::nullopt, [](const searching::SearchResult&) {});
  MoveList legalMoves;
  otherBoard.getState().generateMoves<White>(legalMoves);
  const auto alternative = std::find_if(legalMoves.begin(), legalMoves.end(), [&](const Move& move) {
    return move != otherResult.pvLine->front();
  });
  ASSERT_NE(alternative, legalMoves.end());
  otherBoard.playMove(notation::moveToString(*alternative));
  otherBoard.search(1, std::nullopt, [](const searching::SearchResult&) {});
  EXPECT_TRUE(otherIncomingLines.back().empty());
}

TEST(BoardPV, ReturnedLineFeedsTheNextDepth) {
  std::vector<std::vector<Move>> incomingLines;
  Board board{PVTrackingSearchPolicy{&incomingLines}, TestTimePolicy{kTestTimePercentage}};

  const auto result = board.search(2, std::nullopt, [](const searching::SearchResult&) {});

  ASSERT_EQ(incomingLines.size(), 2u);
  EXPECT_TRUE(incomingLines.front().empty());
  EXPECT_EQ(incomingLines.back().size(), result.pvLine->size());
  EXPECT_TRUE(std::equal(incomingLines.back().begin(), incomingLines.back().end(), result.pvLine->begin()));
}

TEST(BoardPV, ReplayedPositionMovesPreserveMatchingSuffix) {
  std::vector<std::vector<Move>> incomingLines;
  Board board{PVTrackingSearchPolicy{&incomingLines}, TestTimePolicy{kTestTimePercentage}};
  const auto first = board.search(1, std::nullopt, [](const searching::SearchResult&) {});
  const searching::PVLine firstPV = *first.pvLine;
  ASSERT_EQ(firstPV.size(), 2u);
  const std::string firstText = notation::moveToString(firstPV[0]);
  const std::string replyText = notation::moveToString(firstPV[1]);

  const std::array<std::string_view, 1> firstMove{firstText};
  board.setPosition(fen::kStartPosition, firstMove);
  board.search(1, std::nullopt, [](const searching::SearchResult&) {});
  ASSERT_EQ(incomingLines.back().size(), 1u);
  EXPECT_EQ(incomingLines.back()[0], firstPV[1]);

  const std::array<std::string_view, 2> bothMoves{firstText, replyText};
  board.setPosition(fen::kStartPosition, bothMoves);
  board.search(1, std::nullopt, [](const searching::SearchResult&) {});
  EXPECT_TRUE(incomingLines.back().empty());
}

TEST(BoardPV, ReplayedEngineMoveAndOpponentReplyPreserveRemainingPV) {
  std::vector<std::vector<Move>> incomingLines;
  Board board{PVTrackingSearchPolicy{&incomingLines, false, 4}, TestTimePolicy{kTestTimePercentage}};
  const searching::PVLine originalPV = *board.search(1, std::nullopt, [](const searching::SearchResult&) {}).pvLine;
  ASSERT_EQ(originalPV.size(), 4u);

  const std::string firstText = notation::moveToString(originalPV[0]);
  const std::string replyText = notation::moveToString(originalPV[1]);
  const std::array<std::string_view, 2> moves{firstText, replyText};
  board.setPosition(fen::kStartPosition, moves);
  board.search(1, std::nullopt, [](const searching::SearchResult&) {});

  ASSERT_EQ(incomingLines.back().size(), 2u);
  EXPECT_EQ(incomingLines.back()[0], originalPV[2]);
  EXPECT_EQ(incomingLines.back()[1], originalPV[3]);
}

TEST(BoardPV, FenOnlySuccessorAndFailedPositionUpdateKeepCorrectPV) {
  std::vector<std::vector<Move>> incomingLines;
  Board board{PVTrackingSearchPolicy{&incomingLines}, TestTimePolicy{kTestTimePercentage}};
  const auto first = board.search(1, std::nullopt, [](const searching::SearchResult&) {});
  const searching::PVLine firstPV = *first.pvLine;
  ASSERT_EQ(firstPV.size(), 2u);

  const std::array<std::string_view, 1> invalidMoves{"not-a-move"};
  EXPECT_THROW(board.setPosition(fen::kStartPosition, invalidMoves), std::invalid_argument);
  board.search(1, std::nullopt, [](const searching::SearchResult&) {});
  ASSERT_EQ(incomingLines.back().size(), firstPV.size());
  EXPECT_EQ(incomingLines.back()[0], firstPV[0]);

  BoardState successor = board.getState();
  successor.makeMove<White>(firstPV[0], testing_support::NoOpMoveCallback{});
  board.setPosition(notation::boardToFen(successor), {});
  board.search(1, std::nullopt, [](const searching::SearchResult&) {});
  ASSERT_EQ(incomingLines.back().size(), 1u);
  EXPECT_EQ(incomingLines.back()[0], firstPV[1]);
}

TEST(UciIntegration, SelectsBlackIncrementForTimedSearch) {
  std::istringstream input{
    "position fen 7k/8/8/8/8/8/8/7K b - - 0 1\n"
    "go depth 2 wtime 0 btime 0 winc 0 binc 1000\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  int requestedDepth = 0;
  std::vector<searching::SearchResult> results;
  Board board{RecordingSearchPolicy{&requestedDepth, &results}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(requestedDepth, 2);
  EXPECT_EQ(results.size(), 2u);
  EXPECT_EQ(uciOutput.str().find("info string error:"), std::string::npos);
}

TEST(UciIntegration, UsesLastDepthDurationToAvoidStartingAnotherDepth) {
  std::istringstream input{"go wtime 2000 btime 2000 winc 0 binc 0\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  int requestedDepth = 0;
  Board board{SlowSearchPolicy{&requestedDepth, std::chrono::milliseconds{60}}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(requestedDepth, 1);
  std::istringstream output{uciOutput.str()};
  std::string line;
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  const auto info = parseInfoLine(line);
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->depth, 1);
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(line.find("bestmove "), 0u);
  EXPECT_FALSE(static_cast<bool>(std::getline(output, line)));
}

TEST(UciIntegration, RejectsIncompleteOrInvalidTimeControls) {
  std::istringstream input{
    "go wtime 100\n"
    "go depth 2 winc 10\n"
    "go wtime -1 btime 100\n"
    "go wtime nope btime 100\n"
    "go wtime 100 btime 100 btime 200\n"
    "go wtime 100 btime 100 winc\n"
    "go wtime 100 btime 100 movestogo nope\n"
    "go wtime 100 btime 100 movestogo 1 movestogo 2\n"
    "quit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  int requestedDepth = 0;
  std::vector<searching::SearchResult> results;
  Board board{RecordingSearchPolicy{&requestedDepth, &results}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(requestedDepth, 0);
  EXPECT_TRUE(results.empty());
  std::istringstream output{uciOutput.str()};
  std::string line;
  for (int i = 0; i < 8; ++i) {
    ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
    EXPECT_EQ(line.find("info string error:"), 0u);
  }
  EXPECT_FALSE(static_cast<bool>(std::getline(output, line)));
}

TEST(UciIntegration, FormatsMateScoresWithinOneHundredPoints) {
  const std::vector<int32_t> scores{
    kMateMagnitude - kMateWindow, -(kMateMagnitude - kMateWindow),
    kMateMagnitude, -kMateMagnitude,
    kMateMagnitude - kMateWindow - 1, -(kMateMagnitude - kMateWindow - 1),
    kMateMagnitude + 1, -(kMateMagnitude + 1),
    std::numeric_limits<int32_t>::min()};
  const std::vector<int64_t> expectedScores{
    (kMateWindow + 1) / 2, -(kMateWindow + 1) / 2, 0, 0,
    kMateMagnitude - kMateWindow - 1, -(kMateMagnitude - kMateWindow - 1),
    kMateMagnitude + 1, -(kMateMagnitude + 1),
    std::numeric_limits<int32_t>::min()};
  std::istringstream input{"go depth 9\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  Board board{ScoreSequenceSearchPolicy{&scores}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();

  std::istringstream output{uciOutput.str()};
  std::string line;
  uint64_t totalNodes = 0;
  int64_t previousTimeMs = 0;
  for (size_t i = 0; i < scores.size(); ++i) {
    ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
    const auto info = parseInfoLine(line);
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->depth, i + 1);
    EXPECT_EQ(info->scoreKind, i < 4 ? "mate" : "cp");
    EXPECT_EQ(info->score, expectedScores[i]);
    EXPECT_TRUE(info->pv.empty());
    totalNodes += ScoreSequenceSearchPolicy::kNodesPerDepth;
    EXPECT_GE(info->timeMs, previousTimeMs);
    EXPECT_EQ(info->nodes, totalNodes);
    previousTimeMs = info->timeMs;
  }
  MoveList moves;
  board.getState().generateMoves<White>(moves);
  ASSERT_FALSE(moves.empty());
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(line, "bestmove " + notation::moveToString(moves[0]));
  EXPECT_FALSE(static_cast<bool>(std::getline(output, line)));
}

TEST(UciIntegration, ReplaysMovesAndResetsNewGame) {
  std::istringstream input{"position startpos moves e2e4 e7e5\nucinewgame\nposition startpos moves e2e4 e7e5\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{UciSearch{evaluation::HandCraftEvaluationPolicy{}, 50, kTestTranspositionEntries}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(notation::boardToFen(board.getState()),
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2");
  EXPECT_TRUE(uciOutput.str().empty());
  EXPECT_NE(humanOutput.str().find("FEN: " + std::string(fen::kStartPosition)), std::string::npos);
}

TEST(UciIntegration, NewGameKeepsBoardUntilNextPositionCommand) {
  std::istringstream input{"position startpos moves e2e4\nucinewgame\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{UciSearch{evaluation::HandCraftEvaluationPolicy{}, 50, kTestTranspositionEntries}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(notation::boardToFen(board.getState()),
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
  EXPECT_TRUE(uciOutput.str().empty());
}

TEST(UciIntegration, NewGameResetsSearchPolicyAndBoardHistory) {
  std::istringstream input{
    "position startpos moves e2e4\nucinewgame\ngo depth 1\nposition startpos\ngo depth 1\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  int requestedDepth = 0;
  int resetCalls = 0;
  std::vector<size_t> historySizes;
  std::vector<searching::SearchResult> results;
  Board board{RecordingSearchPolicy{&requestedDepth, &results, nullptr, &resetCalls, &historySizes},
              TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();

  EXPECT_EQ(resetCalls, 1);
  EXPECT_EQ(historySizes, (std::vector<size_t>{1, 1}));
  EXPECT_EQ(requestedDepth, 1);
  EXPECT_EQ(notation::boardToFen(board.getState()), fen::kStartPosition);
  EXPECT_NE(uciOutput.str().find("bestmove "), std::string::npos);
}

TEST(BoardReset, ClearsSearchHistoryAndPVButKeepsPosition) {
  int requestedDepth = 0;
  int resetCalls = 0;
  std::vector<searching::SearchResult> results;
  std::vector<size_t> historySizes;
  std::vector<size_t> incomingPVSizes;
  Board board{RecordingSearchPolicy{
    &requestedDepth, &results, nullptr, &resetCalls, &historySizes, &incomingPVSizes},
    TestTimePolicy{kTestTimePercentage}};

  board.playMove("e2e4");
  const auto first = board.search(1, std::nullopt, [](const searching::SearchResult&) {});
  ASSERT_FALSE(first.pvLine->empty());
  const std::string position = notation::boardToFen(board.getState());

  board.reset();
  board.search(1, std::nullopt, [](const searching::SearchResult&) {});

  EXPECT_EQ(resetCalls, 1);
  EXPECT_EQ(notation::boardToFen(board.getState()), position);
  EXPECT_EQ(historySizes, (std::vector<size_t>{2, 1}));
  EXPECT_EQ(incomingPVSizes, (std::vector<size_t>{0, 0}));
}

TEST(UciIntegration, RollsBackInvalidPositionAndSearchesToRequestedDepth) {
  std::istringstream input{
    "position startpos moves e2e4\n"
    "position startpos moves e2e4 e7e5 e4e8\n"
    "position fen broken w - - 0 1\n"
    "go depth 2\n"
    "isready\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{UciSearch{evaluation::HandCraftEvaluationPolicy{}, 50, kTestTranspositionEntries}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(notation::boardToFen(board.getState()),
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
  EXPECT_NE(uciOutput.str().find("info string error: Illegal UCI move: e4e8\n"), std::string::npos);
  EXPECT_NE(uciOutput.str().find("info string error: Invalid piece letter\n"), std::string::npos);
  const std::string output = uciOutput.str();
  const size_t moveStart = output.find("bestmove ");
  ASSERT_NE(moveStart, std::string::npos);
  const size_t moveEnd = output.find('\n', moveStart);
  ASSERT_NE(moveEnd, std::string::npos);
  const std::string bestMove = output.substr(moveStart + 9, moveEnd - moveStart - 9);
  MoveList legalMoves;
  board.getState().generateMoves<Black>(legalMoves);
  EXPECT_TRUE(std::any_of(legalMoves.begin(), legalMoves.end(), [&](const Move& move) {
    return notation::moveToString(move) == bestMove;
  }));
  EXPECT_NE(uciOutput.str().find("readyok\n"), std::string::npos);
}

TEST(UciIntegration, RejectsInvalidDepthAndReportsNoLegalMove) {
  std::istringstream input{
    "go\n"
    "go depth\n"
    "go depth 0\n"
    "go depth nope\n"
    "position fen 7k/5Q2/6K1/8/8/8/8/8 b - - 0 1\n"
    "go depth 2\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{UciSearch{evaluation::HandCraftEvaluationPolicy{}, 50, kTestTranspositionEntries}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  const std::string output = uciOutput.str();
  std::istringstream lines{output};
  std::string line;
  for (int i = 0; i < 4; ++i) {
    ASSERT_TRUE(static_cast<bool>(std::getline(lines, line)));
    EXPECT_EQ(line.find("info string error: "), 0u);
  }
  uint64_t previousNodes = 0;
  int64_t previousTime = 0;
  for (int depth = 1; depth <= 2; ++depth) {
    ASSERT_TRUE(static_cast<bool>(std::getline(lines, line)));
    const auto info = parseInfoLine(line);
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->depth, depth);
    EXPECT_GE(info->nodes, previousNodes);
    EXPECT_GE(info->timeMs, previousTime);
    EXPECT_TRUE(info->pv.empty());
    previousNodes = info->nodes;
    previousTime = info->timeMs;
  }
  ASSERT_TRUE(static_cast<bool>(std::getline(lines, line)));
  EXPECT_EQ(line, "bestmove 0000");
  EXPECT_FALSE(static_cast<bool>(std::getline(lines, line)));
  EXPECT_EQ(notation::boardToFen(board.getState()), "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
}

TEST(UciIntegration, ReportsCheckmateWithoutLegalMove) {
  std::istringstream input{"position fen 7k/6Q1/6K1/8/8/8/8/8 b - - 0 1\ngo depth 1\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{UciSearch{evaluation::HandCraftEvaluationPolicy{}, 50, kTestTranspositionEntries}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  std::istringstream output{uciOutput.str()};
  std::string line;
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  const auto info = parseInfoLine(line);
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->depth, 1);
  EXPECT_EQ(info->scoreKind, "mate");
  EXPECT_EQ(info->score, 0);
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(line, "bestmove 0000");
}

TEST(UciIntegration, ReportsNpsForFastSearch) {
  std::istringstream input{"go depth 1\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  Board board{ZeroTimeSearchPolicy{}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  std::istringstream output{uciOutput.str()};
  std::string line;
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  const auto info = parseInfoLine(line);
  ASSERT_TRUE(info.has_value());
  EXPECT_GE(info->timeMs, 0);
  EXPECT_GT(info->nps, info->nodes);
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(line, "bestmove 0000");
}

TEST(UciIntegration, ReportsElapsedTimeAndNps) {
  std::istringstream input{"go depth 1\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  Board board{SubMillisecondSearchPolicy{}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  std::istringstream output{uciOutput.str()};
  std::string line;
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  const auto info = parseInfoLine(line);
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->depth, 1);
  EXPECT_GE(info->timeMs, 0);
  EXPECT_EQ(info->nodes, SubMillisecondSearchPolicy::kNodes);
  EXPECT_GT(info->nps, info->nodes);
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(line, "bestmove 0000");
  EXPECT_FALSE(static_cast<bool>(std::getline(output, line)));
}

TEST(UciIntegration, PlayAcceptsLegalMoveAndRejectsIllegalMove) {
  std::istringstream input{"play e2e4\nplay e2e5\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{UciSearch{evaluation::HandCraftEvaluationPolicy{}, 50, kTestTranspositionEntries}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(notation::boardToFen(board.getState()),
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
  EXPECT_EQ(uciOutput.str(), "info string error: Illegal UCI move: e2e5\n");
}

TEST(UciIntegration, ReplaysPromotionCastlingAndEnPassant) {
  struct Case {
    const char* command;
    const char* expected;
  };
  const Case cases[] = {
    {"position fen 7k/P7/8/8/8/8/8/7K w - - 0 1 moves a7a8q\nquit\n",
     "Q6k/8/8/8/8/8/8/7K b - - 0 1"},
    {"position fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1 moves e1g1\nquit\n",
     "r3k2r/8/8/8/8/8/8/R4RK1 b kq - 1 1"},
    {"position fen 7k/8/8/3pP3/8/8/8/7K w - d6 0 1 moves e5d6\nquit\n",
     "7k/8/3P4/8/8/8/8/7K b - - 0 1"},
  };
  for (const Case& example : cases) {
    std::istringstream input{example.command};
    std::ostringstream uciOutput;
    std::ostringstream humanOutput;
    UciBoard board{UciSearch{evaluation::HandCraftEvaluationPolicy{}, 50, kTestTranspositionEntries}, TestTimePolicy{kTestTimePercentage}};
    user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};
    terminal.run();
    EXPECT_EQ(notation::boardToFen(board.getState()), example.expected);
    EXPECT_TRUE(uciOutput.str().empty());
  }
}

TEST(UciProtocol, SearchResultRetainsOriginalFiveFieldInterface) {
  const searching::SearchResult result{42, std::nullopt, 7, std::chrono::nanoseconds{9}, nullptr};
  const auto& [score, bestMove, nodes, time, pv] = result;
  EXPECT_EQ(score, 42);
  EXPECT_FALSE(bestMove);
  EXPECT_EQ(nodes, 7u);
  EXPECT_EQ(time.count(), 9);
  EXPECT_EQ(pv, nullptr);
}
