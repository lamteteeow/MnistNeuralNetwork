#pragma once

#include "BaseLayer.hpp"
#include "Eigen/Dense"

#define EPSILON 1e-10

using Tensor = Eigen::MatrixXd;

class CrossEntropyLoss final : public BaseLayer
{
private:
    bool trainable = true;
    Tensor prediction_tensor;

public:
    CrossEntropyLoss() {}
    ~CrossEntropyLoss() {}

    /**
     * @author Lam Tran
     * @since 23-01-2025
     * @brief Compute the loss at the end of forward pass via cross entropy function, internally calling `computed_loss`, convert double to Tensor(1x1) and return `loss_tensor`
     * @param label_tensor
     * @return Tensor
     */
    Tensor forward(const Tensor &label_tensor) override
    {
        double loss = computed_loss(this->prediction_tensor, label_tensor);
        Tensor loss_tensor(1, 1);
        loss_tensor(0, 0) = loss;
        return loss_tensor;
    }

    /**
     * @author Lam Tran
     * @since 20-12-2024
     * @brief Compute the loss at the end of forward pass via cross entropy function
     * @param prediction_tensor Prediction tensor from the predecessor layer
     * @param label_tensor
     * @return double
     */
    double computed_loss(const Tensor &prediction_tensor, const Tensor &label_tensor)
    {
        this->prediction_tensor = prediction_tensor;
        return -(label_tensor.array() * (prediction_tensor.array() + EPSILON).log()).sum();
    }

    /**
     * @author Lam Tran
     * @since 20-12-2024
     * @brief Compute the initial error tensor to start backward pass
     * @param label_tensor Label tensor from successor layer
     * @return Tensor
     */
    Tensor backward(const Tensor &label_tensor) override
    {
        return -label_tensor.array() / (this->prediction_tensor.array() + EPSILON);
    }
};