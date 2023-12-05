#pragma once

#include <iomanip>
#include <iostream>

#include <fstream>
#include <random>

#include "misc/typedefs.h"
#include "tensor/tensor.h"


class LinearLayer {

public:
    LinearLayer(size_t batch_size, size_t output_size, size_t input_size, real_t learn_rate) :
            batch_size_(batch_size), input_size_(input_size), output_size_(output_size), eta_(learn_rate) {

        weights_ = std::make_shared<Tensor<real_t>>(
                std::initializer_list<size_t>{input_size_ + 1, output_size_}, 0.0);

        in_transposed_ = std::make_shared<Tensor<real_t>>(std::initializer_list<size_t>{input_size_, batch_size_}, 0.0);
        weights_transposed_ = std::make_shared<Tensor<real_t>>(
                std::initializer_list<size_t>{output_size_, input_size_ + 1}, 0.0);

        std::mt19937_64 rng(0);
        std::uniform_real_distribution<real_t> unif(-1, 1);

        // generate random weights
        for (size_t i = 0; i < input_size_ + 1; ++i) { // one additional layer for bias
            for (size_t o = 0; o < output_size_; ++o) {
                (*weights_)({i, o}) = unif(rng) / (real_t) input_size_;
            }
        }
    }

    LinearLayer(size_t batch_size, size_t output_size, size_t input_size, real_t learn_rate,
                const std::shared_ptr<Tensor<real_t>> &weights) :
            weights_(weights), batch_size_(batch_size), input_size_(input_size), output_size_(output_size),
            eta_(learn_rate) {

        assert(weights->shape()[0] == input_size_ + 1);
        assert(weights->shape()[1] == output_size_);

        in_transposed_ = std::make_shared<Tensor<real_t>>(std::initializer_list<size_t>{input_size_, batch_size_}, 0.0);
        weights_transposed_ = std::make_shared<Tensor<real_t>>(
                std::initializer_list<size_t>{output_size_, input_size_ + 1}, 0.0);
    }

    virtual std::shared_ptr<Tensor<real_t>>
    forward(const std::shared_ptr<Tensor<real_t>> &in) {
        assert(in->shape()[0] == batch_size_);
        assert(in->shape()[1] == input_size_);

        in_ = in;
        out_ = std::make_shared<Tensor<real_t>>(std::initializer_list<size_t>{batch_size_, output_size_}, 0.0);

        // matrix-matrix mult for batched forward pass
        // ((N+1) x B)^T x ((N+1) x M)
        // (B x (N+1)) x ((N+1) x M) -> B x M
        for (size_t b = 0; b < batch_size_; ++b) {
            for (size_t o = 0; o < output_size_; ++o) {
                for (size_t i = 0; i < input_size_; ++i) {
                    (*out_)({b, o}) += (*in_)({b, i}) * (*weights_)({i, o});
                }

                // one additional layer for bias
                (*out_)({b, o}) += (*weights_)({input_size_, o});
            }
        }

        return out_;
    }

    virtual std::shared_ptr<Tensor<real_t>>
    backward(const std::shared_ptr<Tensor<real_t>> &grad) {
        assert(grad->shape()[0] == batch_size_);
        assert(grad->shape()[1] == output_size_);

        auto prev_layer_grad = std::make_shared<Tensor<real_t>>(std::initializer_list<size_t>{batch_size_, input_size_},
                                                                0.0);

        /* transpose matrix */
        for (size_t o = 0; o < output_size_; ++o) {
            for (size_t i = 0; i < input_size_ + 1; ++i) {
                (*weights_transposed_)({o, i}) = (*weights_)({i, o});
            }
        }

        /* compute derivatives wrt to input -> matrix-matrix mult */

        // (B x M) x (N x M)^T
        // (B x M) x (M x N) -> B x N

        for (size_t b = 0; b < batch_size_; ++b) {
            for (size_t i = 0; i < input_size_; ++i) {
                for (size_t o = 0; o < output_size_; ++o) {
                    (*prev_layer_grad)({b, i}) += (*grad)({b, o}) * (*weights_transposed_)({o, i});
                }
            }
        }


        /* SGD */

        // transpose input matrix
        for (size_t b = 0; b < batch_size_; ++b) {
            for (size_t i = 0; i < input_size_; ++i) {
                (*in_transposed_)({i, b}) = (*in_)({b, i});
            }
        }

        // weight update

        // (B x N)^T x (B x M)
        // (N x B) x (B x M) -> (N x M)
        for (size_t i = 0; i < input_size_; ++i) {
            for (size_t o = 0; o < output_size_; ++o) {
                for (size_t b = 0; b < batch_size_; ++b) {
                    (*weights_)({i, o}) -= eta_ * (*in_transposed_)({i, b}) * (*grad)({b, o});
                }
            }
        }

        // bias update
        for (size_t o = 0; o < output_size_; ++o) {
            for (size_t b = 0; b < batch_size_; ++b) {
                (*weights_)({input_size_, o}) -= eta_ * (*grad)({b, o});
            }
        }

        return prev_layer_grad;
    }

    std::shared_ptr<Tensor<real_t>> getWeights() {
        return weights_;
    }

    void printWeights() {
        std::cout << "Weights are: \n" << *weights_;
    }

protected:
    std::shared_ptr<Tensor<real_t>> in_;
    std::shared_ptr<Tensor<real_t>> out_;
    std::shared_ptr<Tensor<real_t>> weights_;

    std::shared_ptr<Tensor<real_t>> in_transposed_;
    std::shared_ptr<Tensor<real_t>> weights_transposed_;

    size_t batch_size_;
    size_t input_size_;
    size_t output_size_;

    real_t eta_; // learning rate
};