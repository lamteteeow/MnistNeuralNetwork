#pragma once

#include "BaseLayer.hpp"
#include "Eigen/Dense"

using Tensor = Eigen::MatrixXd;

class ReLU : public BaseLayer
{
private:
    Tensor relu_cache;

public:
    ReLU() : relu_cache(Tensor()) {}
    ~ReLU() {}

    /**
     * @author Lam Tran
     * @since 20-12-2024
     * @brief : Introduces non-linearity to each individual input neuron x_i of x via ReLU
     * @param input_tensor Input tensor from the predecessor layer
     * @return Tensor
     */
    Tensor forward(const Tensor &input_tensor)
    {
        this->relu_cache = (input_tensor.array() > 0).cast<double>();
        return input_tensor.cwiseMax(0.0);
    }

    /**
     * @author Lam Tran
     * @since 20.12.2024
     *
     * @brief Compute the next error tensor from the rectified linear unit layer
     *
     * @param error_tensor Error tensor from the successor layer
     * @return Tensor
     */
    Tensor backward(const Tensor &error_tensor)
    {
        return error_tensor.cwiseProduct(this->relu_cache);
    }
};