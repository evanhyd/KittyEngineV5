#pragma once
#include "layer.h"
#include <array>
#include <cassert>
#include <span>
#include <vector>

namespace mlp {
  class NNUE {
    std::vector<Layer> layers_;
    Layer::Activator hiddenActivator_;
    Layer::Activator outputActivator_;
    struct Accumulator {
      std::vector<float> input;
      std::vector<float> firstLayerRawSums;
      bool initialized = false;
    };
    std::array<Accumulator, 2> accumulators_;

    std::span<const float> propagateFrom(size_t layerIndex) {
      // Run the remaining hidden layers, then the output layer.
      for (size_t i = layerIndex; i + 1 < layers_.size(); ++i) {
        layers_[i].propagateForward(layers_[i - 1], hiddenActivator_);
      }
      layers_.back().propagateForward(layers_[layers_.size() - 2], outputActivator_);
      return layers_.back().getOutput();
    }

  public:
    // Constructor that initializes the MLP with a given architecture.
    explicit NNUE(
      std::span<const size_t> layerSizes,
      Layer::Activator hiddenActivator,
      Layer::Activator outputActivator)
      : hiddenActivator_(hiddenActivator),
      outputActivator_(outputActivator) {

      assert(layerSizes.size() >= 3); // Input, hidden, and output layers.

      // Construct the layers.
      layers_.reserve(layerSizes.size());
      for (size_t i = 0; i < layerSizes.size(); ++i) {
        layers_.emplace_back(layerSizes[i]);
      }

      // Connect the layers.
      for (size_t i = 1; i < layerSizes.size(); ++i) {
        layers_[i].connect(layers_[i - 1]);
      }
    }

    // Run full inference from a new input.
    std::span<const float> infer(std::span<const float> input) {
      layers_.front().setOutput(input);
      return propagateFrom(1);
    }

    // Run full inference after changing one input.
    std::span<const float> infer(std::size_t index, float value) {
      layers_.front().setOutput(index, value);
      return propagateFrom(1);
    }

    void clearAccumulators() noexcept {
      // Mark both perspectives for rebuilding at the next search.
      for (Accumulator& accumulator : accumulators_) {
        accumulator.initialized = false;
      }
    }

    bool hasAccumulator(size_t perspective) const noexcept {
      return accumulators_[perspective].initialized;
    }

    void initializeAccumulator(size_t perspective, std::span<const float> input) {
      assert(layers_.size() >= 3 && input.size() == layers_.front().getNumNeurons());
      Accumulator& accumulator = accumulators_[perspective];
      accumulator.input.assign(input.begin(), input.end());
      const Layer& first = layers_[1];
      accumulator.firstLayerRawSums.assign(first.getBiases().begin(), first.getBiases().end());
      const Matrix& weights = first.getWeights();
      // Add each nonzero input's contribution to the first hidden layer.
      for (size_t index = 0; index < input.size(); ++index) {
        if (input[index] == 0.0f) {
          continue;
        }
        const auto column = weights.column(index);
        for (size_t neuron = 0; neuron < accumulator.firstLayerRawSums.size(); ++neuron) {
          accumulator.firstLayerRawSums[neuron] += input[index] * column[neuron];
        }
      }
      accumulator.initialized = true;
    }

    float getInput(size_t perspective, size_t index) const noexcept {
      return accumulators_[perspective].input[index];
    }

    void setInput(size_t perspective, size_t index, float value) noexcept {
      Accumulator& accumulator = accumulators_[perspective];
      if (!accumulator.initialized) {
        return;
      }
      const float delta = value - accumulator.input[index];
      if (delta == 0.0f) {
        return;
      }
      accumulator.input[index] = value;
      const Matrix& weights = layers_[1].getWeights();
      const auto column = weights.column(index);
      // Adjust the first-layer sums without recalculating every input.
      for (size_t neuron = 0; neuron < accumulator.firstLayerRawSums.size(); ++neuron) {
        accumulator.firstLayerRawSums[neuron] += delta * column[neuron];
      }
    }

    std::span<const float> inferPerspective(size_t perspective) {
      const Accumulator& accumulator = accumulators_[perspective];
      assert(accumulator.initialized);
      layers_[1].setActivatedOutput(accumulator.firstLayerRawSums, hiddenActivator_);
      return propagateFrom(2);
    }

    // Layers.
    size_t countLayer() const {
      return layers_.size();
    }

    Layer& getLayer(size_t index) {
      return layers_[index];
    }

    const Layer& getLayer(size_t index) const {
      return layers_[index];
    }

    std::span<Layer> getLayers() {
      return layers_;
    }

    std::span<const Layer> getLayers() const {
      return layers_;
    }
  };
}
