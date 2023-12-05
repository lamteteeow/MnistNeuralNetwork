#pragma once

#include <cstddef>

#include "misc/typedefs.h"
#include "tensor/tensor.h"
#include "misc/util.h"

[[maybe_unused]]
static size_t
count_correct_predictions(const std::shared_ptr<Tensor<real_t>> &in, const std::shared_ptr<Tensor<real_t>> &labels,
                          size_t batch_size, size_t labels_size) {

    assert(in->shape()[0] == batch_size);
    assert(in->shape()[1] == labels_size);
    assert(labels->shape()[0] == batch_size);
    assert(labels->shape()[1] == labels_size);

    size_t correct = 0;

    for (size_t b = 0; b < batch_size; ++b) {
        // search for entry with max probability
        size_t maxIdx = 0;
        for (size_t f = 0; f < labels_size; ++f) {
            if ((*in)({b, f}) > (*in)({b, maxIdx})) {
                maxIdx = f;
            }
        }

        // check if prediction was correct by checking the label entry
        if ((*labels)({b, maxIdx}) > 0.0) {
            ++correct;
        }
    }

    return correct;
}

void
log_predictions(const std::shared_ptr<Tensor<real_t>> &in, const std::shared_ptr<Tensor<real_t>> &labels,
                std::ofstream& outstream, size_t cur_batch, size_t batch_size, size_t labels_size) {

    assert(in->shape()[0] == batch_size);
    assert(in->shape()[1] == labels_size);
    assert(labels->shape()[0] == batch_size);
    assert(labels->shape()[1] == labels_size);

    outstream << "Current batch: " << cur_batch << std::endl;
    for (size_t b = 0; b < batch_size; ++b) {
        // search for entry with max probability
        size_t pred_idx = 0;
        for (size_t f = 0; f < labels_size; ++f) {
            if ((*in)({b, f}) > (*in)({b, pred_idx})) {
                pred_idx = f;
            }
        }

        // find index of one-hot-enc value
        size_t truth_idx = 0;
        for (size_t f = 0; f < labels_size; ++f) {
            auto val = (*labels)({b, f});

            if (fp_almost_equal(val, 1.0)) {
                truth_idx = f;
                break;
            }
        }

        // check if prediction was correct by checking the label entry
        outstream << " - image " << cur_batch * batch_size + b << ": Prediction=" << pred_idx << ". Label=" << truth_idx
                  << std::endl;
    }
}
