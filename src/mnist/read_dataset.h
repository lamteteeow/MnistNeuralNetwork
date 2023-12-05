#pragma once

#include <iomanip>
#include <iostream>

#include <cstdint>

#include <fstream>

#include "misc/typedefs.h"
#include "tensor/tensor.h"

static int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}

std::vector<std::shared_ptr<Tensor<uchar>>>
read_mnist_images(const std::string &full_path, const size_t batch_size, size_t &number_of_images, size_t &image_size) {
    std::ifstream file(full_path, std::ios::binary);

    if (file.is_open()) {
        int magic_number = 0, nimages = 0, nrows = 0, ncols = 0;

        file.read((char *) &magic_number, sizeof(int));
        magic_number = reverse_int(magic_number);

        if (magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

        file.read((char *) &nimages, sizeof(int));
        file.read((char *) &nrows, sizeof(int));
        file.read((char *) &ncols, sizeof(int));

        size_t number_of_rows, number_of_cols;
        number_of_images = static_cast<size_t>(reverse_int(nimages));
        number_of_rows = static_cast<size_t>(reverse_int(nrows));
        number_of_cols = static_cast<size_t>(reverse_int(ncols));

        image_size = number_of_rows * number_of_cols;

        assert(number_of_images % batch_size == 0);

        auto number_of_batches = number_of_images / batch_size;

        std::vector<std::shared_ptr<Tensor<uchar>>> dataset;
        dataset.reserve(number_of_batches);

        for (size_t i = 0; i < number_of_batches; ++i) {
            auto tensor = std::make_shared<Tensor<uchar>>(
                    std::initializer_list<size_t>{batch_size, number_of_rows, number_of_cols}, 0);

            auto address = &((*tensor)({0, 0, 0}));
            file.read((char *) address, static_cast<long>(batch_size * image_size * sizeof(uchar)));

            dataset.push_back(tensor);
        }

        return dataset;
    } else {
        throw std::runtime_error("Cannot open file `" + full_path + "`!");
    }
}

std::vector<std::shared_ptr<Tensor<uchar>>>
read_mnist_labels(const std::string &full_path, const size_t batch_size, size_t &number_of_labels) {
    std::ifstream file(full_path, std::ios::binary);

    if (file.is_open()) {
        int magic_number = 0, nlabels = 0;

        file.read((char *) &magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);

        if (magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

        file.read((char *) &nlabels, sizeof(int));
        number_of_labels = static_cast<size_t>(reverse_int(nlabels) * sizeof(uchar));

        assert(number_of_labels % batch_size == 0);

        auto number_of_batches = number_of_labels / batch_size;

        std::vector<std::shared_ptr<Tensor<uchar>>> dataset;
        dataset.reserve(number_of_batches);

        for (size_t i = 0; i < number_of_batches; ++i) {
            auto tensor = std::make_shared<Tensor<uchar>>(std::initializer_list<size_t>{batch_size}, 0);
            auto address = &((*tensor)({0}));

            file.read((char *) address, static_cast<long>(batch_size * sizeof(uchar)));

            dataset.push_back(tensor);
        }

        return dataset;
    } else {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}