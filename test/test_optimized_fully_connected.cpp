#include <iostream>
#include <cmath>
#include <filesystem>
#include <cfenv>

#include "mnist/read_dataset.h"
#include "mnist/convert_dataset.h"
#include "network/linear_layer.h"
#include "mnist/flatten_pixel_tensors.h"
#include "network/linear_layer_optimized.h"
#include "misc/util.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cout
                << "Invalid arguments. Program must be run as: "
                   "./test_optimized_fully_connected <filepath in (string)> <batch_size (int)>"
                << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string in_file = argv[1];
    size_t batch_size = std::stoul(argv[2]);

    size_t num_images, image_size;
    auto images = read_mnist_images(in_file, batch_size, num_images, image_size);

    int ncols = static_cast<int>(std::sqrt(image_size));
    int nrows = ncols;

    size_t hidden_size = 500;
    size_t num_batches = num_images / batch_size;
    real_t learning_rate = 1E-3;

    auto converted_images = encode_mnist_images(images, num_batches, batch_size, nrows, ncols);

    // linearize pixels to neurons
    auto flattened_image_tensors = flatten_pixel_tensors<real_t>(converted_images, num_batches, batch_size,
                                                                 nrows, ncols);

    // init weights

    auto weights = std::make_shared<Tensor<real_t>>(std::initializer_list<size_t>{image_size + 1, hidden_size}, 0.0);
    auto weightsDup = std::make_shared<Tensor<real_t>>(std::initializer_list<size_t>{image_size + 1, hidden_size}, 0.0);

    std::mt19937_64 rng(0);
    std::uniform_real_distribution<real_t> unif(-1, 1);
    for (size_t j = 0; j < hidden_size; ++j) {
        for (size_t i = 0; i < image_size + 1; ++i) { // one additional layer for bias
            auto val = unif(rng) / (real_t) image_size;
            (*weights)({j, i}) = val;
            (*weightsDup)({j, i}) = val;
        }
    }

    // init both layers with identical weights
    LinearLayer linear_layer(batch_size, hidden_size, image_size, learning_rate, weights);

    LinearLayerOptimized linear_layer_opti(batch_size, hidden_size, image_size, learning_rate, weightsDup);

    // check for 100 samples if both linear layers compute the same results
    for (size_t cur_batch = 0; cur_batch < num_batches; ++cur_batch) {
        auto image_tensor_per_batch = flattened_image_tensors[cur_batch];

        // test forward pass
        auto res_fwd = linear_layer.forward(image_tensor_per_batch);
        auto res_opti_fwd = linear_layer_opti.forward(image_tensor_per_batch);

        assert(res_fwd->shape()[0] == batch_size);
        assert(res_fwd->shape()[1] == hidden_size);
        assert(res_opti_fwd->shape()[0] == batch_size);
        assert(res_opti_fwd->shape()[1] == hidden_size);

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < hidden_size; ++i) {
                if (!fp_almost_equal((*res_fwd)({b, i}), (*res_opti_fwd)({b, i}))) {
                    std::cout << "Forward: Optimized and non-optimized linear layers obtain different results: " <<
                              (*res_fwd)({b, i}) << " vs " << (*res_opti_fwd)({b, i}) << std::endl;

                    return EXIT_FAILURE;
                }
            }
        }

        // test backward pass
        auto tensor = std::make_shared<Tensor<real_t>>(std::initializer_list<size_t>{batch_size, hidden_size}, 1.0);
        auto tensorDup = std::make_shared<Tensor<real_t>>(std::initializer_list<size_t>{batch_size, hidden_size}, 1.0);

        auto res_bwd = linear_layer.backward(tensor);
        auto res_opti_bwd = linear_layer_opti.backward(tensorDup);

        assert(res_bwd->shape()[0] == batch_size);
        assert(res_bwd->shape()[1] == image_size);
        assert(res_opti_bwd->shape()[0] == batch_size);
        assert(res_opti_bwd->shape()[1] == image_size);

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < image_size; ++i) {
                if (!fp_almost_equal((*res_bwd)({b, i}), (*res_opti_bwd)({b, i}))) {
                    std::cout << "Backward: Optimized and non-optimized linear layers obtain different results: " <<
                              (*res_bwd)({b, i}) << " vs " << (*res_opti_bwd)({b, i}) << std::endl;

                    return EXIT_FAILURE;
                }
            }
        }
    }

    return EXIT_SUCCESS;
}
