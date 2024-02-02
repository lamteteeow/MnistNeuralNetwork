#pragma once

#include <memory>
#include <vector>

#include "misc/typedefs.h"
#include "tensor/tensor.h"

template<typename T>
static std::vector<std::shared_ptr<Tensor<T>>>
drop_batch_rank_images(const std::vector<std::shared_ptr<Tensor<T>>> &in,
                       size_t num_batches, size_t batch_size, size_t num_rows, size_t num_cols) {

    // sanity checks
    assert(batch_size == 1);
    assert(in[0]->shape()[0] == batch_size);
    assert(in[0]->shape()[1] == num_rows);
    assert(in[0]->shape()[2] == num_cols);

    (void) batch_size;

    // drop rank for batch dimension as it's size is set to 1 anyways
    std::vector<std::shared_ptr<Tensor<T>>> unbatched_tensors;

    for (size_t k = 0; k < num_batches; ++k) {
        auto tensor = std::make_shared<Tensor<T>>(std::initializer_list<size_t>{num_rows, num_cols}, (T) 0);

        for (size_t j = 0; j < num_rows; ++j) {
            for (size_t i = 0; i < num_cols; ++i) {
                (*tensor)({j, i}) = (*(in[k]))({0, j, i});
            }
        }

        unbatched_tensors.push_back(tensor);
    }

    return unbatched_tensors;
}
