#pragma once

#include "BaseLayer.hpp"
#include "Eigen/Dense"

using Tensor = Eigen::MatrixXd;

class SoftMax final : public BaseLayer
{
private:
  Tensor y_hat;

public:
    SoftMax() : BaseLayer() {}
    ~SoftMax() {}

    /**
     * @author Lam Tran
     * @since 20.12.2024
     *
     * @brief Encode input neurons x to PDF y_hat via softmax, into the successor layer
     *
     * @param input_tensor Input tensor from the predecessor layer
     * @return Tensor
     */
    Tensor forward(const Tensor &input_tensor) override
    {
        // Shift for numerical stability
        Tensor x_shifted = input_tensor.colwise() - input_tensor.rowwise().maxCoeff();
        Tensor e_x = x_shifted.array().exp();
        // y_hat.(rows, cols) = (batch_size, output_size)
        this->y_hat = e_x.array().colwise() / e_x.array().rowwise().sum();
        return this->y_hat;
    }

    /**
     * @author Lam Tran
     * @since 20.12.2024
     *
     * @brief Compute next error tensor e_{n−1} for the predecessor layer
     *
     * @param error_tensor Error tensor from the successor layer
     * @return Tensor
     */
    Tensor backward(const Tensor &error_tensor) override
    {
        // error_tensor.(rows, cols) = (batch_size, output_size)

        // Calculate the gradient of the loss with respect to input
        // weighted_sum_error.(rows, cols) = (batch_size, 1)
        Tensor weighted_sum_error = (error_tensor.array() * y_hat.array()).rowwise().sum();

        // Transform the sum back to the forwarded shape and subtract it from each error_tensor element
        Tensor adjusted_error = error_tensor - (weighted_sum_error.replicate(1, error_tensor.cols()));

        // Perform the final element-wise multiplication with y_hat
        // return.(rows, cols) = (batch_size, output_size)
        return y_hat.array() * adjusted_error.array();
    }
};