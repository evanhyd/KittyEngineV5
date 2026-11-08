#pragma once
#include "boardstate.h"
#include "evaluation_policy.h"
#include "notation.h"
#include "perft_driver.h"
#include "position_fens.h"
#include "time_control_policy.h"
#include "uci_protocol.h"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace bb::user_interface {
  template <typename BoardT>
  class TerminalUI {
    BoardT& board_;
    std::istream& uciInput_;
    std::ostream& uciOutput_;
    std::ostream& humanOutput_;
    std::string printBuffer_;
    std::string uciBuffer_;
    std::string lastFrame_;
    bool running_ = false;

    static void appendMove(std::string& out, Move move) noexcept {
      const Square from = move.from();
      const Square to = move.to();
      out.push_back(static_cast<char>('a' + getSquareFile(from)));
      out.push_back(static_cast<char>('8' - getSquareRank(from)));
      out.push_back(static_cast<char>('a' + getSquareFile(to)));
      out.push_back(static_cast<char>('8' - getSquareRank(to)));
      if (const Piece promo = move.getPromotedPieceType(); promo != NoPiece) {
        out.push_back(notation::pieceToAscii(Black, promo));
      }
    }

    static constexpr auto kSearchFlushInterval = std::chrono::seconds{1};

    void flushUci() {
      uciOutput_.flush();
    }

    void reportProtocolError(std::string_view message) {
      uciBuffer_.clear();
      uciBuffer_ += "info string error: ";
      for (const char ch : message) {
        uciBuffer_.push_back((ch == '\r' || ch == '\n') ? ' ' : ch);
      }
      uciBuffer_.push_back('\n');
      uciOutput_.write(uciBuffer_.data(), static_cast<std::streamsize>(uciBuffer_.size()));
      flushUci();
    }

    void renderState(const BoardState& state) {
      printBuffer_.clear();
      for (Square rank = 0; rank < kBoardLenSize; ++rank) {
        printBuffer_ += "  -------------------------------\n ";
        for (Square file = 0; file < kBoardLenSize; ++file) {
          const auto occupant = state.getPieceAt(rankFileToSquare(rank, file));
          const char symbol = occupant.piece != NoPiece
            ? notation::pieceToAsciiVisual(occupant.side, occupant.piece)
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
                                  notation::sideToString(state.getSide()),
                                  notation::castleToString(state.getCastlingRights()),
                                  notation::squareToString(state.getEnpassantSquare()),
                                  state.getHalfmove(), state.getFullmove());
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

    void render() {
      renderState(board_.getState());
    }

  public:
    TerminalUI(BoardT& board, std::istream& uciInput, std::ostream& uciOutput,
               std::ostream& humanOutput)
      : board_(board), uciInput_(uciInput), uciOutput_(uciOutput),
        humanOutput_(humanOutput) {
      uciInput_.tie(&uciOutput_);
      uciBuffer_.reserve(1024);
      printBuffer_.reserve(1024);
    }

    void run() {
      humanOutput_ << std::format("Welcome to KittyEngineV5.\nBuild Version: {}_{}\n", __DATE__, __TIME__);
      render();

      uci::UciProtocol protocol{
        [this] {
          static constexpr std::string_view kUciOk = "id name KittyEngineV5\nid author UnboxTheCat\nuciok\n";
          uciOutput_.write(kUciOk.data(), static_cast<std::streamsize>(kUciOk.size()));
          flushUci();
        },
        [this] {
          static constexpr std::string_view kReadyOk = "readyok\n";
          uciOutput_.write(kReadyOk.data(), static_cast<std::streamsize>(kReadyOk.size()));
          flushUci();
        },
        [this] {
          board_.newGame();
          render();
        },
        [this](std::string_view fen, std::span<const std::string_view> moves) {
          board_.setPosition(fen, moves);
          render();
        },
        [this](std::span<const std::string_view> args) {
          std::optional<int> depth;
          std::optional<int64_t> whiteTime;
          std::optional<int64_t> blackTime;
          std::optional<int64_t> whiteIncrement;
          std::optional<int64_t> blackIncrement;
          std::optional<int64_t> moveTime;
          std::optional<int64_t> movesToGo;
          for (size_t i = 0; i < args.size(); ++i) {
            if (++i == args.size()) {
              throw std::invalid_argument("go option needs a value");
            }
            const std::string_view value = args[i];
            int64_t parsed = 0;
            const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
            if (error != std::errc{} || end != value.data() + value.size() || parsed < 0) {
              throw std::invalid_argument("go option needs a nonnegative integer");
            }
            const std::string_view option = args[i - 1];
            if (option == "depth") {
              if (depth || parsed == 0 || parsed > std::numeric_limits<int>::max()) {
                throw std::invalid_argument("go depth needs a positive integer");
              }
              depth = static_cast<int>(parsed);
            } else {
              std::optional<int64_t>* target = nullptr;
              if (option == "wtime") target = &whiteTime;
              else if (option == "btime") target = &blackTime;
              else if (option == "winc") target = &whiteIncrement;
              else if (option == "binc") target = &blackIncrement;
              else if (option == "movetime") target = &moveTime;
              else if (option == "movestogo") target = &movesToGo;
              else throw std::invalid_argument("unsupported go option");
              if (*target) {
                throw std::invalid_argument("duplicate go option");
              }
              *target = parsed;
            }
          }
          const bool hasClock = whiteTime || blackTime;
          if (hasClock != (whiteTime && blackTime) ||
              (!hasClock && (whiteIncrement || blackIncrement))) {
            throw std::invalid_argument("go time control needs wtime and btime");
          }
          if (!depth && !hasClock && !moveTime) {
            throw std::invalid_argument("go depth or time control is required");
          }

          std::optional<time_control::TimeControl> timeControl;
          if (hasClock || moveTime) {
            if (movesToGo && *movesToGo > std::numeric_limits<int>::max()) {
              throw std::invalid_argument("go movestogo is too large");
            }
            timeControl.emplace();
            if (hasClock) {
              timeControl->wtime = std::chrono::milliseconds{*whiteTime};
              timeControl->btime = std::chrono::milliseconds{*blackTime};
            }
            if (whiteIncrement) timeControl->winc = std::chrono::milliseconds{*whiteIncrement};
            if (blackIncrement) timeControl->binc = std::chrono::milliseconds{*blackIncrement};
            if (moveTime) timeControl->moveTime = std::chrono::milliseconds{*moveTime};
            if (movesToGo) timeControl->movesToGo = static_cast<int>(*movesToGo);
          }

          int completedDepth = 0;
          uint64_t totalNodes = 0;
          uint64_t totalProbes = 0;
          uint64_t totalHits = 0;
          const auto searchStart = std::chrono::steady_clock::now();
          auto lastFlushTime = searchStart;
          const int searchDepth = std::min(depth.value_or(searching::kMaxDepthHardCutoff), searching::kMaxDepthHardCutoff);
          searching::SearchResult lastResult{};
          board_.search(searchDepth, timeControl, [this, &completedDepth, &totalNodes, &totalProbes, &totalHits, &lastResult, searchStart, &lastFlushTime](const auto& result) {
            lastResult = result;
            ++completedDepth;
            totalNodes += result.nodes;
            totalProbes += result.probes;
            totalHits += result.hits;
            const auto elapsed = std::chrono::steady_clock::now() - searchStart;
            const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
            const double elapsedSeconds = std::chrono::duration<double>(elapsed).count();
            const uint64_t nps = static_cast<uint64_t>(totalNodes / std::max(elapsedSeconds, 0.001));

            constexpr int64_t mateScore = evaluation::kCheckmateWinScore;
            constexpr int64_t mateWindow = 100;
            const int64_t score = static_cast<int64_t>(result.score);
            const int64_t magnitude = std::abs(score);
            const bool isMateScore = magnitude >= mateScore - mateWindow && magnitude <= mateScore;
            const uint64_t hashfull = totalProbes > 0 ? std::clamp((totalHits * 1000) / totalProbes, 0ULL, 1000ULL) : 0;

            uciBuffer_.clear();
            if (isMateScore) {
              const int64_t movesToMate = (mateScore - magnitude + 1) / 2;
              std::format_to(std::back_inserter(uciBuffer_),
                             "info depth {} score mate {} time {} nodes {} nps {} hashfull {}",
                             completedDepth, score < 0 ? -movesToMate : movesToMate, elapsedMs, totalNodes, nps, hashfull);
            } else {
              std::format_to(std::back_inserter(uciBuffer_),
                             "info depth {} score cp {} time {} nodes {} nps {} hashfull {}",
                             completedDepth, score, elapsedMs, totalNodes, nps, hashfull);
            }

            const auto& pv = result.pvLine;
            if (!pv.empty()) {
              uciBuffer_ += " pv";
              for (const Move& move : pv) {
                uciBuffer_.push_back(' ');
                appendMove(uciBuffer_, move);
              }
            }
            uciBuffer_.push_back('\n');
            uciOutput_.write(uciBuffer_.data(), static_cast<std::streamsize>(uciBuffer_.size()));

            // Intermediate flush point: throttle flushes during deep/long searches (>= 1s)
            // to keep GUIs updated without stalling fast microsecond depths.
            const auto now = std::chrono::steady_clock::now();
            if (now - lastFlushTime >= kSearchFlushInterval) {
              flushUci();
              lastFlushTime = now;
            }
          });

          // Definitive flush point: finished searching all depths / PV lines for this position.
          uciBuffer_.clear();
          uciBuffer_ += "bestmove ";
          if (lastResult.move) {
            appendMove(uciBuffer_, lastResult.move);
          } else {
            uciBuffer_ += "0000";
          }
          uciBuffer_.push_back('\n');
          uciOutput_.write(uciBuffer_.data(), static_cast<std::streamsize>(uciBuffer_.size()));
          flushUci();
        },
        [this] {
          running_ = false;
        },
        [this](std::string_view move) {
          board_.playMove(move);
          render();
        },
        [this](uint32_t depth, bool detail) {
          if (detail) {
            constexpr perft::Config config{false, false, true};
            perft::runPerft<config>(board_.getState(), depth, uciOutput_);
          } else {
            constexpr perft::Config config{false, true, false};
            perft::runPerft<config>(board_.getState(), depth, uciOutput_);
          }
        }
      };

      running_ = true;
      for (std::string line; running_ && std::getline(uciInput_, line); ) {
        try {
          protocol.send(line);
        } catch (const std::exception& error) {
          reportProtocolError(error.what());
        }
        flushUci();
      }
    }
  };

  template <typename BoardT>
  TerminalUI(BoardT&, std::istream&, std::ostream&, std::ostream&) -> TerminalUI<BoardT>;
}
