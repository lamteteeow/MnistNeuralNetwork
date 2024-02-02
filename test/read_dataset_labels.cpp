#include <cassert>
#include <cmath>
#include <iostream>

#include "mnist/read_dataset.h"

int main(int argc, char *argv[]) {

    if (argc != 4) {
        std::cout
                << "Invalid arguments. Program must be run as: "
                   "./read_dataset_labels <filepath in (string)> <filepath out (string)> <label idx (int)>"
                << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string in_file = argv[1];
    std::string out_file = argv[2];
    int label_idx = std::stoi(argv[3]);

    size_t num_labels;
    size_t batch_size = 1; // we are only interested in a single label
    auto labels = read_mnist_labels(in_file, batch_size, num_labels);

    size_t num_batches = num_labels / batch_size;

    std::vector<std::shared_ptr<Tensor<double>>> label_tensors;
    label_tensors.reserve(num_batches);

    assert(labels[0]->shape()[0] == 1);

    for (size_t k = 0; k < num_batches; ++k) {
        auto tensor = std::make_shared<Tensor<double>>(std::initializer_list<size_t>{10}, 0.0);
        auto idx = ((*(labels[k]))({0}));
        (*tensor)({idx}) = 1.0;

        label_tensors.push_back(tensor);
    }

    //assert(num_labels == 60000);

    writeTensorToFile((*(label_tensors[label_idx])), out_file);

    return 0;
}
