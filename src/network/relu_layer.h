#pragma once

#include <iomanip>
#include <iostream>

#include <fstream>
#include <random>

#include "misc/typedefs.h"
#include "tensor/tensor.h"

class ReLULayer {
public:

    ReLULayer(size_t batch_size, size_t input_size) :
            batch_size_(batch_size), input_size_(input_size) {}

    std::shared_ptr<Tensor<real_t>>
    forward(const std::shared_ptr<Tensor<real_t>> &in) {
        assert(in->shape()[0] == batch_size_);
        assert(in->shape()[1] == input_size_);

        in_ = in;
        out_ = std::make_shared<Tensor<real_t>>(std::initializer_list<size_t>{batch_size_, input_size_}, 0.0);

        for (size_t b = 0; b < batch_size_; ++b) {
            for (size_t i = 0; i < input_size_; ++i) {
                real_t val = (*in_)({b, i});
                (*out_)({b, i}) = val > 0.0 ? val : 0.0;
            }
        }

        return out_;
    }

    std::shared_ptr<Tensor<real_t>>
    backward(const std::shared_ptr<Tensor<real_t>> &grad) {
        assert(grad->shape()[0] == batch_size_);
        assert(grad->shape()[1] == input_size_);

        auto prev_layer_grad = std::make_shared<Tensor<real_t>>(std::initializer_list<size_t>{batch_size_, input_size_},
                                                                0.0);

        for (size_t b = 0; b < batch_size_; ++b) {
            for (size_t i = 0; i < input_size_; ++i) {
                real_t val = (*in_)({b, i});
                real_t val2 = (*grad)({b, i});
                (*prev_layer_grad)({b, i}) = val > 0.0 ? val2 : 0.0;
            }
        }

        return prev_layer_grad;
    }

private:
    std::shared_ptr<Tensor<real_t>> in_;
    std::shared_ptr<Tensor<real_t>> out_;

    size_t batch_size_;
    size_t input_size_;
};
