#pragma once

#include "Eigen/Dense"

class BaseLayer
{
public:
    BaseLayer() : trainable(false), weights(Eigen::MatrixXd()) {}
    virtual ~BaseLayer() = default;

    /**
     * @author Lam Tran
     * @since 20.12.2024
     *
     * @brief Forward pass from this layer
     *
     * @param input_tensor Input tensor from the preceding layer
     * @return Eigen::MatrixXd
     */
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd &input_tensor) = 0;

    /**
     * @author Lam Tran
     * @since 20.12.2024
     *
     * @brief Backward pass from this layer
     *
     * @param error_tensor Error tensor from the successive layer
     * @return Eigen::MatrixXd
     */
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd &error_tensor) = 0;

    bool trainable;
    Eigen::MatrixXd weights;
};