#pragma once

#include "BaseLayer.hpp"
#include "Eigen/Dense"

class SoftMax : public BaseLayer
{
private:
    // Eigen::MatrixXd input_tensor_cache;
    Eigen::MatrixXd input_tensor_suc;

public:
    SoftMax() {}
    ~SoftMax() {}

    /**
     * @author Lam Tran
     * @since 20.12.2024
     *
     * @brief Forward pass from the SoftMax layer
     *
     * @param input_tensor Input tensor from the preceding layer
     * @return Eigen::MatrixXd
     */
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input_tensor) override
    {
        // Shift for numerical stability
        Eigen::MatrixXd x_shifted = input_tensor.colwise() - input_tensor.rowwise().maxCoeff();
        Eigen::MatrixXd input_tensor_exp = x_shifted.array().exp();
        Eigen::MatrixXd next_input_tensor = input_tensor_exp.array().colwise() / input_tensor_exp.array().rowwise().sum();

        this->input_tensor_suc = next_input_tensor;
        return next_input_tensor;
    }

    /**
     * @author Lam Tran
     * @since 20.12.2024
     *
     * @brief Backward pass from the SoftMax layer
     *
     * @param error_tensor Error tensor from the successive layer
     * @return Eigen::MatrixXd
     */
    Eigen::MatrixXd backward(const Eigen::MatrixXd &error_tensor)
    {
        // Calculate the gradient of the loss with respect to input
        Eigen::MatrixXd weighted_error_sum = (error_tensor.array() * input_tensor_suc.array()).rowwise().sum();

        // Transform the sum back to the original shape for it to be subtracted from each error_tensor element
        Eigen::MatrixXd adjusted_error = error_tensor.array() - (weighted_error_sum.replicate(1, error_tensor.cols())).array();

        // Perform the final element-wise multiplication with input_tensor_suc
        Eigen::MatrixXd error_tensor_pre = input_tensor_suc.array() * adjusted_error.array();

        return error_tensor_pre;
    }
};