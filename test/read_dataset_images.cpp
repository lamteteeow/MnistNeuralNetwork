#include <iostream>
#include <cmath>
#include <cassert>

#include "mnist/read_dataset.h"
#include "mnist/convert_dataset.h"
#include "mnist/drop_batch_rank.h"

int main(int argc, char *argv[]) {

    if (argc != 4) {
        std::cout
                << "Invalid arguments. Program must be run as: "
                   "./read_dataset_images <filepath in (string)> <filepath out (string)> <image idx (int)>"
                << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string in_file = argv[1];
    std::string out_file = argv[2];
    int image_idx = std::stoi(argv[3]);

    size_t num_images, img_size;
    size_t batch_size = 1; // we are only interested in a single image
    auto images = read_mnist_images(in_file, batch_size, num_images, img_size);

    assert(img_size == 784);
    //assert(num_images == 60000);
    assert(num_images % batch_size == 0);

    auto ncols = static_cast<size_t>(std::sqrt(img_size));
    auto nrows = ncols;

    // compute number of batches
    auto num_batches = num_images / batch_size;

    // convert images from uchar tensors to float tensors
    auto converted_images = encode_mnist_images(images, num_batches, batch_size, nrows, ncols);

    // drop rank for batch dimension as we have a batch size of 1
    auto unbatched_images = drop_batch_rank_images(converted_images, num_batches, batch_size, nrows, ncols);

    // write tensor to file for validation
    writeTensorToFile(*(unbatched_images[image_idx]), out_file);

    // customized output for testing
    std::ofstream outstream("test.txt");
    for (size_t j = 0; j < nrows; ++j) {
        for (size_t i = 0; i < ncols; ++i) {
            if ((*(images[image_idx]))({0, j, i}) == 0)
                outstream << 0;
            else
                outstream << 1;
        }
        outstream << std::endl;
    }
    outstream.close();

    return 0;
}
