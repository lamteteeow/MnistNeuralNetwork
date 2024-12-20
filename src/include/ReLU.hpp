#pragma once

#include "BaseLayer.hpp"
#include "Eigen/Dense"

class ReLU : public BaseLayer
{
private:
    Eigen::MatrixXd relu_gradient;

public:
    ReLU() : relu_gradient(Eigen::MatrixXd()) {}
    ~ReLU() {}

    /**
     * @author Lam Tran
     * @since 20.12.2024
     *
     * @brief Forward pass gradient from the rectified linear unit layer
     *
     * @param input_tensor Input tensor from the preceding layer
     * @return
     */
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input_tensor)
    {
        relu_gradient = (input_tensor.array() > 0).cast<double>();
        return input_tensor.cwiseMax(0.0);
    }

    /**
     * @author Lam Tran
     * @since 20.12.2024
     *
     * @brief Backward pass gradient from the rectified linear unit layer
     *
     * @param error_tensor Error tensor from the successive layer
     * @return
     */
    Eigen::MatrixXd backward(const Eigen::MatrixXd &error_tensor)
    {
        return error_tensor.cwiseProduct(relu_gradient);
    }
};