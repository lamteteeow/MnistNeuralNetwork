#pragma once

#include <iomanip>
#include <iostream>

#include <fstream>
#include <random>

#include "misc/typedefs.h"
#include "tensor/tensor.h"
#include "misc/util.h"

class CrossEntropyLossLayer {
public:

    CrossEntropyLossLayer(size_t batch_size, size_t labels_size) :
            batch_size_(batch_size), labels_size_(labels_size) {}

    real_t
    forward(const std::shared_ptr<Tensor<real_t>> &in, const std::shared_ptr<Tensor<real_t>> &labels) {
        assert(in->shape()[0] == batch_size_);
        assert(in->shape()[1] == labels_size_);
        assert(labels->shape()[0] == batch_size_);
        assert(labels->shape()[1] == labels_size_);

        in_ = in;

        real_t loss = 0.0;
        for (size_t b = 0; b < batch_size_; ++b) {
            // find index of one-hot-enc value
            size_t idx = 0;
            for (size_t f = 0; f < labels_size_; ++f) {
                auto val = (*labels)({b, f});

                if (fp_almost_equal(val, 1.0)) {
                    idx = f;
                    break;
                }
            }

            // compute loss
            real_t pred = (*in)({b, idx});
            loss += -log(pred + EPSILON);
        }

        return loss / (real_t) batch_size_;
    }

    std::shared_ptr<Tensor<real_t>>
    backward(const std::shared_ptr<Tensor<real_t>> &labels) {
        assert(labels->shape()[0] == batch_size_);
        assert(labels->shape()[1] == labels_size_);

        auto prev_layer_grad = std::make_shared<Tensor<real_t>>(
                std::initializer_list<size_t>{batch_size_, labels_size_}, 0.0);

        for (size_t b = 0; b < batch_size_; ++b) {
            for (size_t f = 0; f < labels_size_; f++) {
                (*prev_layer_grad)({b, f}) = -((*labels)({b, f}) / (*in_)({b, f}));
            }
        }

        return prev_layer_grad;
    }

private:
    std::shared_ptr<Tensor<real_t>> in_;

    size_t batch_size_;
    size_t labels_size_;

    const real_t EPSILON = 1e-9;
};