#include <cassert>
#include <cmath>
#include <iostream>

#include "network/linear_layer.h"

int main(int argc, char *argv[]) {
    (void) argc;
    (void) argv;

    size_t batch_size = 10;
    size_t ncols = 2;
    size_t nrows = 2;

    size_t input_size = nrows * ncols;
    size_t output_size = input_size;

    auto weights = std::make_shared<Tensor<real_t>>(std::initializer_list<size_t>{input_size + 1, output_size}, 0.0);

    for (size_t i = 0; i < input_size; ++i) {
        int c = 1;
        for (size_t o = 0; o < output_size; ++o) {
            (*weights)({o, i}) = pow(0.5, c);
            (*weights)({input_size, o}) = 1.0;

            c++;
        }
    }

    auto inputTensor = std::make_shared<Tensor<real_t>>(std::initializer_list<size_t>{batch_size, input_size}, 0.0);
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < input_size; ++i) {
            (*inputTensor)({b, i}) = (real_t) (b * batch_size + i);
        }
    }

    std::cout << "Input tensor: \n" << *inputTensor;

    auto fc = LinearLayer(batch_size, output_size, input_size, 0.0, weights);
    fc.printWeights();

    auto outTensorForward = fc.forward(inputTensor);

    std::cout << "Tensor after forward: \n" << *outTensorForward;

    auto outTensorBackward = fc.backward(inputTensor);

    std::cout << "Tensor after backward: \n" << *outTensorBackward;

    return EXIT_SUCCESS;
}