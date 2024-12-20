#pragma once

#include "BaseLayer.hpp"
#include "Eigen/Dense"

using Tensor = Eigen::MatrixXd;

class SoftMax : public BaseLayer
{
private:
    Tensor input_tensor_cache;

public:
    SoftMax() : input_tensor_cache(Tensor()) {}
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
        Tensor input_tensor_exp = x_shifted.array().exp();
        Tensor input_tensor_pre = input_tensor_exp.array().colwise() / input_tensor_exp.array().rowwise().sum();

        this->input_tensor_cache = input_tensor_pre;
        return input_tensor_pre;
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
    Tensor backward(const Tensor &error_tensor)
    {
        // Calculate the gradient of the loss with respect to input
        Tensor weighted_sum_error = (error_tensor.array() * input_tensor_cache.array()).rowwise().sum();

        // Transform the sum back to the forwarded shape and subtract it from each error_tensor element
        Tensor adjusted_error = error_tensor.array() - (weighted_sum_error.replicate(1, error_tensor.cols())).array();

        // Perform the final element-wise multiplication with input_tensor_cache y_hat
        return input_tensor_cache.array() * adjusted_error.array();
    }
};