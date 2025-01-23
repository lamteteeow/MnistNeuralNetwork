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

void main()
{
    unsigned int seed = 123;
    Initializer *weights_initializer = new Xavier(seed);
    Initializer *bias_initializer = new Xavier(seed);
    double muy = 0.001; // tunable hyperparameter
    Optimizer *optimizer = new SGD(muy);
    unsigned int s = 50; // tunable hyperparameter
    FullyConnected fc1(784, s, optimizer);
    fc1.initialize(weights_initializer, bias_initializer);
    ReLU relu;
    FullyConnected fc2(s, 10, optimizer);
    fc2.initialize(weights_initializer, bias_initializer);
    SoftMax softmax;
    CrossEntropyLoss ce_loss;
}