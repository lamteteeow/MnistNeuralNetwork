#pragma once

#include "BaseLayer.hpp"
#include "Eigen/Dense"

#define EPSILON 1e-10

using Tensor = Eigen::MatrixXd;

class CrossEntropyLoss : public BaseLayer
{
private:
    Tensor prediction_tensor;

public:
    CrossEntropyLoss() {}
    ~CrossEntropyLoss() {}

    /**
     * @author Lam Tran
     * @since 20-12-2024
     * @brief Compute the loss at the end of forward pass via cross entropy function
     * @param prediction_tensor Prediction tensor from the predecessor layer
     * @param label_tensor
     * @return double
     */
    double forward(const Tensor &prediction_tensor, const Tensor &label_tensor)
    {
        this->prediction_tensor = prediction_tensor;
        double loss = -(label_tensor.array() * (prediction_tensor.array() + EPSILON).log()).sum();
        return loss;
    }

    /**
     * @author Lam Tran
     * @since 20-12-2024
     * @brief Compute the initial error tensor to start backward pass
     * @param label_tensor Label tensor from successor layer
     * @return Tensor
     */
    Tensor backward(const Tensor &label_tensor)
    {
        return -label_tensor.array() / (this->prediction_tensor.array() + EPSILON);
    }
};