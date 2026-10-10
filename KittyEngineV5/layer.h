#pragma once
#include "matrix.h"
#include <algorithm>
#include <cassert>
#if defined(_M_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#endif
#include <span>
#include <vector>

namespace mlp {
  class Layer {
  private:
    std::vector<float> biases_;
    std::vector<float> weightedSums_;
    std::vector<float> output_;
    std::vector<size_t> activeOutputs_;
    Matrix weights_; // [output][input]

  public:
    using Activator = float(*)(float);

    explicit Layer(size_t numNeurons)
      : biases_(numNeurons), weightedSums_(numNeurons), output_(numNeurons), weights_(0, 0) {
      activeOutputs_.reserve(numNeurons);
    }

    void connect(const Layer& previousLayer) {
      weights_ = Matrix(getNumNeurons(), previousLayer.getNumNeurons());
    }

    size_t getNumNeurons() const { return output_.size(); }

    std::span<const float> getOutput() const {
      return std::span<const float>(output_);
    }

    // Set the entire layer's neuron.
    void setOutput(std::span<const float> output) {
      assert(output.size() == output_.size());
      std::copy(output.begin(), output.end(), output_.begin());
      activeOutputs_.clear();
      // Record the inputs that can affect the next layer.
      for (size_t i = 0; i < output_.size(); ++i) {
        if (output_[i] != 0.0f) {
          activeOutputs_.push_back(i);
        }
      }
    }

    // Set a single neuron.
    void setOutput(std::size_t index, float value) {
      output_[index] = value;
      activeOutputs_.clear();
      // Refresh the nonzero inputs after this value changes.
      for (size_t i = 0; i < output_.size(); ++i) {
        if (output_[i] != 0.0f) {
          activeOutputs_.push_back(i);
        }
      }
    }

    void setActivatedOutput(std::span<const float> rawSums, Activator activation) {
      assert(rawSums.size() == output_.size());
      activeOutputs_.clear();
      // Activate each cached sum and record nonzero outputs.
      for (size_t i = 0; i < rawSums.size(); ++i) {
        output_[i] = activation(rawSums[i]);
        if (output_[i] != 0.0f) {
          activeOutputs_.push_back(i);
        }
      }
    }

    const Matrix& getWeights() const {
      return weights_;
    }

    void setWeight(size_t outputNeuronIndex, size_t inputNeuronIndex, float weight) {
      weights_.set(outputNeuronIndex, inputNeuronIndex, weight);
    }

    std::span<const float> getBiases() const {
      return std::span<const float>(biases_);
    }

    void setBias(size_t neuronIndex, float bias) {
      biases_[neuronIndex] = bias;
    }

    void propagateForward(const Layer& lastLayer, Activator activation) {
#if defined(_M_AVX2) || defined(__AVX2__)
      if (output_.size() == 32) {
        // Keep the 32 sums in registers while adding each active input.
        __m256 sums[4];
        for (size_t block = 0; block < 4; ++block) {
          sums[block] = _mm256_loadu_ps(biases_.data() + block * 8);
        }

        for (size_t inputIndex : lastLayer.activeOutputs_) {
          const __m256 input = _mm256_set1_ps(lastLayer.output_[inputIndex]);
          const float* weights = weights_.column(inputIndex).data();
          for (size_t block = 0; block < 4; ++block) {
            const __m256 column = _mm256_loadu_ps(weights + block * 8);
            sums[block] = _mm256_add_ps(sums[block], _mm256_mul_ps(input, column));
          }
        }

        for (size_t block = 0; block < 4; ++block) {
          _mm256_storeu_ps(weightedSums_.data() + block * 8, sums[block]);
        }
      } else
#endif
      {
        std::copy(biases_.begin(), biases_.end(), weightedSums_.begin());
        // Add each active input to every neuron in this layer.
        for (size_t inputIndex : lastLayer.activeOutputs_) {
          const float input = lastLayer.output_[inputIndex];
          const auto weights = weights_.column(inputIndex);
          for (size_t neuron = 0; neuron < output_.size(); ++neuron) {
            weightedSums_[neuron] += input * weights[neuron];
          }
        }
      }

      activeOutputs_.clear();
      // Activate the completed sums for the next layer.
      for (size_t i = 0; i < output_.size(); ++i) {
        output_[i] = activation(weightedSums_[i]);
        if (output_[i] != 0.0f) {
          activeOutputs_.push_back(i);
        }
      }
    }
  };
}
