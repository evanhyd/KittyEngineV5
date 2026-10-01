#pragma once
#include "board.h"
#include "uci_protocol.h"
#include <iostream>
#include <string>
#include <format>
#include <stdexcept>

namespace bb::user_interface {
    class TerminalInterfacePolicy {
      std::istream& uciInput_;
      std::ostream& uciOutput_;
      std::ostream& humanOutput_;
      std::string printBuffer_;
      std::string lastFrame_;
      bool running_ = false;

      void renderState(const BoardState& state);
      void reportProtocolError(std::string_view message);

    public:
      TerminalInterfacePolicy(std::istream& uciInput, std::ostream& uciOutput,
                              std::ostream& humanOutput);

      template <typename BoardInterface>
      void render(const BoardInterface& board) {
        renderState(board.getState());
      }

      template <typename BoardInterface>
      void run(BoardInterface& board) {
        humanOutput_ << std::format("Welcome to KittyEngineV5.\nBuild Version: {}_{}\n", __DATE__, __TIME__);
        render(board);

        uci::UciProtocol protocol{
          [this] {
            uciOutput_ << "id name KittyEngineV5\nid author UnboxTheCat\nuciok\n";
          },
          [this] {
            uciOutput_ << "readyok\n";
          },
          [this, &board] {
            board.setPosition(fen::kStartPosition);
            render(board);
          },
          [this, &board](std::string_view fen, std::span<const std::string_view> moves) {
            const BoardState original = board.getState();
            try {
              board.setPosition(fen);
              for (const std::string_view move : moves) {
                board.playMove(move);
              }
            } catch (...) {
              board.getState() = original;
              throw;
            }
            render(board);
          },
          [](std::span<const std::string_view>) {
            // TODO: connect the search policy and emit bestmove when search exists.
            throw std::logic_error("go is not implemented");
          },
          [this] {
            running_ = false;
          },
          [this, &board](std::string_view move) {
            board.playMove(move);
            render(board);
          }
        };

        running_ = true;
        for (std::string line; running_ && std::getline(uciInput_, line); ) {
          try {
            protocol.send(line);
          } catch (const std::exception& error) {
            reportProtocolError(error.what());
          }
          uciOutput_.flush(); // Must flush!.
        }
      }
    };
}
