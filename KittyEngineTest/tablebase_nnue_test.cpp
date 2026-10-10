#include "board.h"
#include "negamax_search_policy.h"
#include "mlp_evaluation_policy.h"
#include "equal_percentage_time_control_policy.h"
#include "nnue_test_model.h"
#include "syzygy_test_path.h"
#include <memory>
#include <gtest/gtest.h>

using namespace bb;
using testing_support::syzygyTestPath;
using testing_support::TestNnueModelFile;

namespace {
  struct ObservableNnue {
    std::shared_ptr<evaluation::MLPEvaluationPolicy> model;
    void reset() { model->reset(); }
    void prepare(const BoardState& state) { model->prepare(state); }
    int32_t evaluate(const BoardState& state) { return model->evaluate(state); }
    template <bool Add>
    void markPiece(Side side, Piece piece, Square square) {
      model->markPiece<Add>(side, piece, square);
    }
    void markCastle(CastlePermission rights) { model->markCastle(rights); }
    void markEnpassant(Square square) { model->markEnpassant(square); }
  };
}

TEST(Nnue, TablebaseSearchRestoresIncrementalAccumulator) {
  const auto path = syzygyTestPath();
  if (path.empty()) {
    GTEST_SKIP() << "Set KITTY_SYZYGY_TEST_PATH for real probe tests";
  }
  tablebase::Service tables;
  tables.setOption("SyzygyPath", path);
  const TestNnueModelFile weights;
  auto model = std::make_shared<evaluation::MLPEvaluationPolicy>(weights.path);
  evaluation::MLPEvaluationPolicy fresh{weights.path};
  Board engine{searching::NegamaxSearchPolicy{ObservableNnue{model}, 80, 1u << 14, &tables},
    time_control::EqualPercentageTimeControlPolicy{0.05f}};
  for (const auto fen : {"8/4P3/4K3/8/8/8/k7/8 w - - 0 1",
                         "8/8/8/3pP3/8/4K3/8/k7 w - d6 0 1"}) {
    engine.setPosition(fen);
    uint64_t hits = 0;
    engine.search(3, std::nullopt, [&](const auto&) { hits += engine.tablebaseHits(); });
    EXPECT_GT(hits, 1u); // Exercise root ranking and interior early returns.
    fresh.reset();
    EXPECT_EQ(model->evaluate(engine.getState()), fresh.evaluate(engine.getState()));
  }
}
