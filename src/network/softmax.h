#pragma once

#include <iomanip>
#include <iostream>

#include <fstream>
#include <random>

#include "misc/typedefs.h"
#include "tensor/tensor.h"
#include "misc/util.h"

class SoftMaxLayer {
public:

    SoftMaxLayer(size_t batch_size, size_t labels_size) :
            batch_size_(batch_size), labels_size_(labels_size) {}

    std::shared_ptr<Tensor<real_t>>
    forward(const std::shared_ptr<Tensor<real_t>> &in) {
        assert(in->shape()[0] == batch_size_);
        assert(in->shape()[1] == labels_size_);

        in_ = in;
        out_ = std::make_shared<Tensor<real_t>>(std::initializer_list<size_t>{batch_size_, labels_size_}, 0.0);

        for (size_t b = 0; b < batch_size_; ++b) {
            real_t max = -std::numeric_limits<real_t>::infinity();
            for (size_t f = 0; f < labels_size_; ++f) {
                real_t val = (*in_)({b, f});
                if (val > max)
                    max = val;
            }

            real_t sum = 0.;
            for (size_t f = 0; f < labels_size_; ++f) {
                real_t val = exp((*in)({b, f}) - max);
                (*out_)({b, f}) = val;
                sum += val;
            }

            for (size_t f = 0; f < labels_size_; ++f) {
                (*out_)({b, f}) /= sum;
            }
        }

        return out_;
    }

    std::shared_ptr<Tensor<real_t>>
    backward(const std::shared_ptr<Tensor<real_t>> &grad) {
        assert(grad->shape()[0] == batch_size_);
        assert(grad->shape()[1] == labels_size_);

        auto prev_layer_grad = std::make_shared<Tensor<real_t>>(
                std::initializer_list<size_t>{batch_size_, labels_size_}, 0.0);

        for (size_t b = 0; b < batch_size_; ++b) {
            real_t sum = 0.;
            for (size_t f = 0; f < labels_size_; ++f) {
                sum += (*grad)({b, f}) * (*out_)({b, f});
            }

            for (size_t f = 0; f < labels_size_; ++f) {
                (*prev_layer_grad)({b, f}) = (*out_)({b, f}) * ((*grad)({b, f}) - sum);
            }
        }

        return prev_layer_grad;
    }

private:
    std::shared_ptr<Tensor<real_t>> in_;
    std::shared_ptr<Tensor<real_t>> out_;

    size_t batch_size_;
    size_t labels_size_;
};
