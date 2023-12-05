#pragma once

#include <iomanip>
#include <iostream>

#include <fstream>

#include "misc/typedefs.h"

// convert MNIST image data from characters to floating point numbers
std::vector<std::shared_ptr<Tensor<real_t>>>
encode_mnist_images(const std::vector<std::shared_ptr<Tensor<uchar>>> &in, size_t number_of_batches, size_t batch_size,
                    size_t nrows,
                    size_t ncols) {
    std::vector<std::shared_ptr<Tensor<real_t>>> out;
    out.reserve(number_of_batches);

    assert(in[0]->shape()[0] == batch_size);
    assert(in[0]->shape()[1] == nrows);
    assert(in[0]->shape()[2] == ncols);

    for (size_t k = 0; k < number_of_batches; ++k) {
        auto tensor = std::make_shared<Tensor<real_t>>(std::initializer_list<size_t>{batch_size, nrows, ncols}, 0.0);

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t j = 0; j < nrows; ++j) {
                for (size_t i = 0; i < ncols; ++i) {
                    (*tensor)({b, j, i}) = (*(in[k]))({b, j, i}) / 255.0;
                }
            }
        }

        out.push_back(tensor);
    }

    return out;
}

// one-hot-encoding for label data
std::vector<std::shared_ptr<Tensor<real_t>>>
encode_mnist_labels(const std::vector<std::shared_ptr<Tensor<uchar>>> &in,
                    size_t number_of_batches, size_t batch_size) {

    std::vector<std::shared_ptr<Tensor<real_t>>> out;
    out.reserve(number_of_batches);

    assert(in[0]->shape()[0] == batch_size);

    for (size_t k = 0; k < number_of_batches; ++k) {
        auto tensor = std::make_shared<Tensor<real_t>>(std::initializer_list<size_t>{batch_size, 10}, 0.0);

        for (size_t b = 0; b < batch_size; ++b) {
            auto idx = static_cast<size_t>((*(in[k]))({b}));

            (*tensor)({b, idx}) = 1.0;
        }

        out.push_back(tensor);
    }

    return out;
}
