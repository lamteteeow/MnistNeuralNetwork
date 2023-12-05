#include <cassert>
#include <cmath>
#include <iostream>

#include "mnist/read_dataset.h"
#include "mnist/flatten_pixel_tensors.h"
#include "mnist/drop_batch_rank.h"

int main(int argc, char *argv[]) {

    if (argc != 7) {
        std::cout
                << "Invalid arguments. Program must be run as: "
                   "./create_sub_dataset <rel_path_images in (string)> <rel_path_labels in (string)> "
                   "<rel_path_images out (string)> <rel_path_labels out (string)>"
                   "<idx start [inclusive] (int)> <idx end [exclusive] (int)>"
                << std::endl;
        return EXIT_FAILURE;
    }

    std::string img_file_in = argv[1];
    std::string label_file_in = argv[2];
    std::string img_file_out = argv[3];
    std::string label_file_out = argv[4];
    int start_idx = std::stoi(argv[5]);
    int end_idx = std::stoi(argv[6]);

    if (end_idx - start_idx <= 0) {
        std::cout << "Invalid arguments: <idx end (int)> must be greater than <idx start (int)>." << std::endl;
        return EXIT_FAILURE;
    }

    size_t num_images, num_labels, img_size;
    size_t batch_size = 1; // we are only interested in a single image
    auto images = read_mnist_images(img_file_in, batch_size, num_images, img_size);

    auto ncols = static_cast<int>(std::sqrt(img_size));
    auto nrows = ncols;

    /* output image subset */

    auto image_tensors = drop_batch_rank_images<uchar>(images, num_images, batch_size, nrows, ncols);

    std::ofstream ofs_img(img_file_out, std::ios::out | std::ios::binary);

    int magic_number_img_rev = reverse_int(2051);
    ofs_img.write((char *) &magic_number_img_rev, sizeof(int));

    int new_num_samples_rev = reverse_int(end_idx - start_idx);
    ofs_img.write((char *) &new_num_samples_rev, sizeof(int));

    int nrows_rev = reverse_int(nrows);
    int ncols_rev = reverse_int(ncols);
    ofs_img.write((char *) &nrows_rev, sizeof(int));
    ofs_img.write((char *) &ncols_rev, sizeof(int));

    for (auto cur_sample = (size_t) start_idx; cur_sample < (size_t) end_idx; ++cur_sample) {
        auto image_tensor_per_sample = image_tensors[cur_sample];

        ofs_img.write((char *) &((*image_tensor_per_sample)({0, 0})), static_cast<long>(img_size * sizeof(uchar)));
    }

    ofs_img.close();

    /* output label subset */

    auto label_tensors = read_mnist_labels(label_file_in, batch_size, num_labels);

    std::ofstream ofs_label(label_file_out, std::ios::out | std::ios::binary);

    int magic_number_label_rev = reverse_int(2049);
    ofs_label.write((char *) &magic_number_label_rev, sizeof(int));

    ofs_label.write((char *) &new_num_samples_rev, sizeof(int));

    for (auto cur_sample = (size_t) start_idx; cur_sample < (size_t) end_idx; ++cur_sample) {
        auto label_tensor_per_sample = label_tensors[cur_sample];

        ofs_label.write((char *) &((*label_tensor_per_sample)({0})), sizeof(uchar));
    }

    ofs_label.close();
}