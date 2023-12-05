#include <cmath>
#include <iostream>
#include <filesystem>

#include "mnist/read_dataset.h"
#include "mnist/convert_dataset.h"
#include "mnist/flatten_pixel_tensors.h"
#include "misc/util.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cout
                << "Invalid arguments. Program must be run as: "
                   "./test_flatten_pixels <filepath in (string)> <batch_size (int)>"
                << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string in_file = argv[1];
    auto batch_size = std::stoul(argv[2]);

    size_t num_images, image_size;
    auto images = read_mnist_images(in_file, batch_size, num_images, image_size);

    auto num_batches = num_images / batch_size;

    auto ncols = static_cast<size_t>(std::sqrt(image_size));
    auto nrows = ncols;

    auto converted_images = encode_mnist_images(images, num_batches, batch_size, nrows, ncols);

    auto flattened_image_tensors = flatten_pixel_tensors<real_t>(converted_images, num_batches, batch_size, nrows,
                                                                 ncols);

    // output should only diff in tensor shape
    writeTensorToFile(*(converted_images[0]), "a.txt");
    writeTensorToFile(*(flattened_image_tensors[0]), "b.txt");

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t j = 0; j < nrows; ++j) {
            for (size_t i = 0; i < ncols; ++i) {
                if (!fp_almost_equal((*(converted_images[0]))({b, j, i}), (*(flattened_image_tensors[0]))({b, j * ncols + i}))) {
                    exit(EXIT_FAILURE);
                }
            }
        }
    }

    return EXIT_SUCCESS;
}
