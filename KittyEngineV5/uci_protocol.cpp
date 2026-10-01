#include "uci_protocol.h"
#include "position_fens.h"
#include <cctype>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bb::uci {
  namespace {
    std::vector<std::string_view> splitTokens(std::string_view line) {
      std::vector<std::string_view> words;
      for (size_t start = 0; start < line.size();) {
        while (start < line.size() && std::isspace(static_cast<unsigned char>(line[start]))) {
          ++start;
        }
        if (start == line.size()) {
          break;
        }
        size_t end = start;
        while (end < line.size() && !std::isspace(static_cast<unsigned char>(line[end]))) {
          ++end;
        }
        words.push_back(line.substr(start, end - start));
        start = end;
      }
      return words;
    }

  }

  UciProtocol::UciProtocol(
    SimpleCallback onUci, 
    SimpleCallback onIsReady,
    SimpleCallback onNewGame, 
    PositionCallback onPosition,
    GoCallback onGo, 
    SimpleCallback onQuit,
    PlayCallback onPlay)
    : 
    onUci_(std::move(onUci)), 
    onIsReady_(std::move(onIsReady)),
    onNewGame_(std::move(onNewGame)), 
    onPosition_(std::move(onPosition)),
    onGo_(std::move(onGo)), 
    onQuit_(std::move(onQuit)),
    onPlay_(std::move(onPlay)) {}

  void UciProtocol::send(std::string_view line) {
    const std::vector<std::string_view> words = splitTokens(line);
    if (words.empty()) {
      return;
    }

    const std::string_view command = words.front();
    const std::span<const std::string_view> args{words.begin() + 1, words.end()};
    if (command == "uci") {
      onUci_();
    } else if (command == "isready") {
      onIsReady_();
    } else if (command == "ucinewgame") {
      onNewGame_();
    } else if (command == "position") {
      if (args.empty()) {
        throw std::invalid_argument("position needs startpos or fen");
      }

      std::string fenStorage;
      std::string_view fen;
      size_t next = 0;
      if (args[0] == "startpos") {
        fen = fen::kStartPosition;
        next = 1;
      } else if (args[0] == "fen") {
        if (args.size() < 7) {
          throw std::invalid_argument("position fen needs six FEN fields");
        }
        for (size_t i = 1; i <= 6; ++i) {
          if (i != 1) {
            fenStorage += ' ';
          }
          fenStorage += args[i];
        }
        fen = fenStorage;
        next = 7;
      } else {
        throw std::invalid_argument("position needs startpos or fen");
      }

      if (next < args.size()) {
        if (args[next] != "moves") {
          throw std::invalid_argument("expected moves after position");
        }
        ++next;
      }
      onPosition_(fen, args.subspan(next));
    } else if (command == "go") {
      onGo_(args);
    } else if (command == "play") {
      if (args.size() != 1) {
        throw std::invalid_argument("play requires one move");
      }
      if (onPlay_) {
        onPlay_(args[0]);
      }
    } else if (command == "quit") {
      onQuit_();
    }
    // Unknown commands are ignored as required by UCI.
  }
}
