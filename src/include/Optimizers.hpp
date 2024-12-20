#pragma once

#include "Eigen/Dense"

using Tensor = Eigen::MatrixXd;

class SGD
{
private:
    double learningRate;

public:
    SGD() : learningRate(0.001) {}
    SGD(double learningRate) : learningRate(learningRate) {}
    ~SGD() {}

    /**
     * @author Lam Tran
     * @since 20-12-2024
     * @brief Adjust the weights using Stochastic Gradient Descent between iterations (not epochs)
     * @param weights
     * @param gradient
     * @return Tensor
     */
    Tensor updateWeights(Tensor &weights, Tensor &gradient)
    {
        return (weights - learningRate * gradient);
    }
};

// TODO: Maybe implement SGD with momentum or Adam in the future