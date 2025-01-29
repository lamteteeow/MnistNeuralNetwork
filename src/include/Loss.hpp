#pragma once

#include "BaseLayer.hpp"
#include "Eigen/Dense"

#define EPSILON 1e-12

using Tensor = Eigen::MatrixXd;

class CrossEntropyLoss final : public BaseLayer
{
private:
    Tensor prediction_tensor;

public:
    CrossEntropyLoss() : BaseLayer() {}
    ~CrossEntropyLoss() {}

    /**
     * @author Lam Tran
     * @since 23-01-2025
     * @brief Wrapper of `computed_loss` which converts double to Tensor(1x1), prints `loss` and returns `loss_tensor`
     * @param label_tensor (rows, cols) = (batch_size, output_size)
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
     * @param prediction_tensor Prediction tensor from the predecessor layer; (rows, cols) = (batch_size, output_size)
     * @param label_tensor (rows, cols) = (batch_size, output_size)
     * @return double
     */
    double computed_loss(const Tensor &prediction_tensor, const Tensor &label_tensor)
    {
        // label_tensor.(rows, cols) = prediction_tensor.(rows, cols) = (batch_size, output_size)
        this->prediction_tensor = prediction_tensor;
        return -(label_tensor.array() * (prediction_tensor.array() + EPSILON).log()).sum();
    }

    /**
     * @author Lam Tran
     * @since 20-12-2024
     * @brief Compute the initial error tensor to start backward pass
     * @param label_tensor Label tensor from successor layer
     * @return Tensor (rows, cols) = (batch_size, output_size)
     */
    Tensor backward(const Tensor &label_tensor) override
    {
        return -label_tensor.array() / (this->prediction_tensor.array() + EPSILON);
    }
};