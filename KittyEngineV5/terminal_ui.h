#pragma once
#include "boardstate.h"
#include "notation.h"
#include "position_fens.h"
#include "uci_protocol.h"
#include <algorithm>
#include <charconv>
#include <format>
#include <iostream>
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
    std::string lastFrame_;
    bool running_ = false;

    void reportProtocolError(std::string_view message) {
      std::string safeMessage{message};
      std::replace_if(safeMessage.begin(), safeMessage.end(),
                      [](char ch) { return ch == '\r' || ch == '\n'; }, ' ');
      uciOutput_ << "info string error: " << safeMessage << '\n';
    }

    void renderState(const BoardState& state) {
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

    void render() {
      renderState(board_.getState());
    }

  public:
    TerminalUI(BoardT& board, std::istream& uciInput, std::ostream& uciOutput,
               std::ostream& humanOutput)
      : board_(board), uciInput_(uciInput), uciOutput_(uciOutput),
        humanOutput_(humanOutput) {}

    void run() {
      humanOutput_ << std::format("Welcome to KittyEngineV5.\nBuild Version: {}_{}\n", __DATE__, __TIME__);
      render();

      uci::UciProtocol protocol{
        [this] {
          uciOutput_ << "id name KittyEngineV5\nid author UnboxTheCat\nuciok\n";
        },
        [this] {
          uciOutput_ << "readyok\n";
        },
        [this] {
          board_.setPosition(fen::kStartPosition);
          render();
        },
        [this](std::string_view fen, std::span<const std::string_view> moves) {
          board_.setPosition(fen, moves);
          render();
        },
        [this](std::span<const std::string_view> args) {
          int depth = 0;
          for (size_t i = 0; i < args.size(); ++i) {
            if (args[i] != "depth") {
              continue;
            }
            if (++i == args.size()) {
              throw std::invalid_argument("go depth needs a positive integer");
            }
            const std::string_view value = args[i];
            const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), depth);
            if (error != std::errc{} || end != value.data() + value.size() || depth <= 0) {
              throw std::invalid_argument("go depth needs a positive integer");
            }
          }
          if (depth == 0) {
            throw std::invalid_argument("go depth is required");
          }
          const auto result = board_.search(depth);
          uciOutput_ << "bestmove "
                     << (result.bestMove ? notation::moveToString(*result.bestMove) : "0000")
                     << '\n';
        },
        [this] {
          running_ = false;
        },
        [this](std::string_view move) {
          board_.playMove(move);
          render();
        }
      };

      running_ = true;
      for (std::string line; running_ && std::getline(uciInput_, line); ) {
        try {
          protocol.send(line);
        } catch (const std::exception& error) {
          reportProtocolError(error.what());
        }
        uciOutput_.flush();
      }
    }
  };

  template <typename BoardT>
  TerminalUI(BoardT&, std::istream&, std::ostream&, std::ostream&) -> TerminalUI<BoardT>;
}
