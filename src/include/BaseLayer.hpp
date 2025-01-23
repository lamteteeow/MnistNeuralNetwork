#pragma once

#include "Eigen/Dense"

using Tensor = Eigen::MatrixXd;

class BaseLayer
{
public:
    BaseLayer() : trainable(false), weights(Tensor()) {}
    // BaseLayer(bool is_trainable) : trainable(is_trainable), weights(Tensor()) {}
    virtual ~BaseLayer() = default;

    /**
     * @author Lam Tran
     * @since 20.12.2024
     *
     * @brief Forward pass from this layer
     *
     * @param input_tensor Input tensor from the predecessor layer
     * @return Tensor
     */
    virtual Tensor forward(const Tensor &input_tensor) = 0;

    /**
     * @author Lam Tran
     * @since 20.12.2024
     *
     * @brief Backward pass from this layer
     *
     * @param error_tensor Error tensor from the successor layer
     * @return Tensor
     */
    virtual Tensor backward(const Tensor &error_tensor) = 0;

    bool trainable;
    Tensor weights;
};