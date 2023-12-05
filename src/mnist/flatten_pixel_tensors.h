#pragma once

#include <iomanip>
#include <iostream>

#include <fstream>

#include "misc/typedefs.h"
#include "tensor/tensor.h"


template<typename T>
static std::vector<std::shared_ptr<Tensor<T>>>
flatten_pixel_tensors(const std::vector<std::shared_ptr<Tensor<T>>> &in, size_t num_batches, size_t batch_size,
                      size_t num_rows, size_t num_cols) {

    // init vector
    std::vector<std::shared_ptr<Tensor<T>>> out;
    out.reserve(num_batches);

    // linearize
    for (size_t k = 0; k < num_batches; ++k) {
        auto tensor = std::make_shared<Tensor<T>>(std::initializer_list<size_t>{batch_size, num_rows * num_cols},
                                                  (T) 0);

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t j = 0; j < num_rows; ++j) {
                for (size_t i = 0; i < num_cols; ++i) {
                    (*tensor)({b, j * num_cols + i}) = (*(in[k]))({b, j, i});
                }
            }
        }

        out.push_back(tensor);
    }

    return out;
}
