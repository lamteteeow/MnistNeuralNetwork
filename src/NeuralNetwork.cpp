#include <iostream>
#include "Eigen/Dense"
#include "Loss.hpp"
#include "Optimizers.hpp"
#include "ReLU.hpp"
#include "SoftMax.hpp"
#include "FullyConnected.hpp"

using Tensor = Eigen::MatrixXd;

class NeuralNetwork
{
};

// Currently for testing accessibility
int main()
{
    unsigned int seed = 123;
    Initializer *weights_initializer = new Xavier(seed);
    Initializer *bias_initializer = new Xavier(seed);
    double muy = 0.001; // tunable hyperparameter
    Optimizer *optimizer = new SGD(muy);
    unsigned int s = 50; // tunable hyperparameter
    FullyConnected fc1 = FullyConnected(784, s, optimizer);
    fc1.initialize(weights_initializer, bias_initializer);
    FullyConnected *fc2 = new FullyConnected(s, 10, optimizer);
    fc2->initialize(weights_initializer, bias_initializer);
    ReLU relu;
    SoftMax softmax;
    CrossEntropyLoss ce_loss;

    ce_loss.trainable = false;
    relu.trainable = false;

    fc1.trainable = true;
    fc2->trainable = true;

    // Clean up dynamically allocated memory
    delete weights_initializer;
    delete bias_initializer;
    delete optimizer;
    delete fc2;

    return 0;
}