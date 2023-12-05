#pragma once

#include <iomanip>
#include <iostream>

#include <fstream>
#include <random>

#include "misc/typedefs.h"
#include "tensor/tensor.h"
#include "linear_layer.h"


class LinearLayerOptimized : public LinearLayer {

public:
    LinearLayerOptimized(size_t batch_size, size_t output_size, size_t input_size, real_t learn_rate) :
            LinearLayer(batch_size, output_size, input_size, learn_rate) {}

    LinearLayerOptimized(size_t batch_size, size_t output_size, size_t input_size, real_t learn_rate,
                         const std::shared_ptr<Tensor<real_t>> &weights) :
            LinearLayer(batch_size, output_size, input_size, learn_rate, weights) {

        assert(weights->shape()[0] == input_size_ + 1);
        assert(weights->shape()[1] == output_size_);
    }

    std::shared_ptr<Tensor<real_t>>
    forward(const std::shared_ptr<Tensor<real_t>> &in) override {
        assert(in->shape()[0] == batch_size_);
        assert(in->shape()[1] == input_size_);

        in_ = in;
        out_ = std::make_shared<Tensor<real_t>>(std::initializer_list<size_t>{batch_size_, output_size_}, 0.0);

        auto weights_preAddr = &((*weights_)({0, 0}));

        // matrix-matrix mult for batched forward pass
        // ((N+1) x B)^T x ((N+1) x M)
        // (B x (N+1)) x ((N+1) x M) -> B x M
#pragma omp parallel for default(none) shared(weights_preAddr)
        for (size_t b = 0; b < batch_size_; ++b) {
            auto in_preAddr = &((*in_)({b, 0}));
            auto out_preAddr = &((*out_)({b, 0}));

            for (size_t o = 0; o < output_size_; ++o) {
                for (size_t i = 0; i < input_size_; ++i) {
                    auto weights_preAddr2 = &(weights_preAddr[i * output_size_]);

                    out_preAddr[o] += in_preAddr[i] * weights_preAddr2[o];
                }

                // one additional layer for bias
                out_preAddr[o] += weights_preAddr[input_size_ * output_size_ + o];
            }
        }

        return out_;
    }

    std::shared_ptr<Tensor<real_t>>
    backward(const std::shared_ptr<Tensor<real_t>> &grad) override {
        assert(grad->shape()[0] == batch_size_);
        assert(grad->shape()[1] == output_size_);

        auto prev_layer_grad = std::make_shared<Tensor<real_t>>(std::initializer_list<size_t>{batch_size_, input_size_},
                                                                0.0);

        auto weights_preAddr = &((*weights_)({0, 0}));
        auto t_weights_preAddr = &((*weights_transposed_)({0, 0}));

        /* transpose matrix */

        for (size_t o = 0; o < output_size_; ++o) {
            auto t_weights_preAddr2 = &(t_weights_preAddr[o * (input_size_ + 1)]);

            for (size_t i = 0; i < input_size_ + 1; ++i) {
                auto weights_preAddr2 = &(weights_preAddr[i * output_size_]);

                t_weights_preAddr2[i] = weights_preAddr2[o];
            }
        }

        /* compute derivatives wrt to input -> matrix-matrix mult */

        // (B x M) x (N x M)^T
        // (B x M) x (M x N) -> B x N

        auto grad_preAddr = &((*grad)({0, 0}));
        auto prev_layer_grad_preAddr = &((*prev_layer_grad)({0, 0}));

#pragma omp parallel for default(none) shared(grad_preAddr, prev_layer_grad_preAddr, t_weights_preAddr)
        for (size_t b = 0; b < batch_size_; ++b) {
            auto grad_preAddr2 = &(grad_preAddr[b * (output_size_)]);
            auto prev_layer_grad_preAddr2 = &(prev_layer_grad_preAddr[b * (input_size_)]);

            for (size_t i = 0; i < input_size_; ++i) {
                for (size_t o = 0; o < output_size_; ++o) {
                    auto t_weights_preAddr2 = &(t_weights_preAddr[o * (input_size_ + 1)]);

                    prev_layer_grad_preAddr2[i] += grad_preAddr2[o] * t_weights_preAddr2[i];
                }
            }
        }


        /* SGD */

        auto in_preAddr = &((*in_)({0, 0}));
        auto t_in_preAddr = &((*in_transposed_)({0, 0}));

        // transpose input matrix
        for (size_t i = 0; i < input_size_; ++i) {
            auto t_in_preAddr2 = &(t_in_preAddr[i * batch_size_]);

            for (size_t b = 0; b < batch_size_; ++b) {
                auto in_preAddr2 = &(in_preAddr[b * input_size_]);

                t_in_preAddr2[b] = in_preAddr2[i];
            }
        }

        // weight update

        // (B x N)^T x (B x M)
        // (N x B) x (B x M) -> (N x M)

#pragma omp parallel for default(none) shared(weights_preAddr, t_in_preAddr, grad_preAddr)
        for (size_t i = 0; i < input_size_; ++i) {
            auto weights_preAddr2 = &(weights_preAddr[i * output_size_]);
            auto t_in_preAddr2 = &(t_in_preAddr[i * batch_size_]);

            for (size_t o = 0; o < output_size_; ++o) {
                for (size_t b = 0; b < batch_size_; ++b) {
                    auto grad_preAddr2 = &(grad_preAddr[b * (output_size_)]);

                    weights_preAddr2[o] -= eta_ * t_in_preAddr2[b] * grad_preAddr2[o];
                }
            }
        }

        // bias update
        auto weights_preAddr2 = &(weights_preAddr[input_size_ * output_size_]);

        for (size_t o = 0; o < output_size_; ++o) {
            for (size_t b = 0; b < batch_size_; ++b) {
                auto grad_preAddr2 = &(grad_preAddr[b * (output_size_)]);

                weights_preAddr2[o] -= eta_ * grad_preAddr2[o];
            }
        }

        return prev_layer_grad;
    }
};