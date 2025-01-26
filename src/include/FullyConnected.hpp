#pragma once

#include "BaseLayer.hpp"
#include "Optimizers.hpp"
#include "Initializers.hpp"
#include "Eigen/Dense"

using Tensor = Eigen::MatrixXd;

class FullyConnected final : public BaseLayer
{
private:
    unsigned int input_size;
    unsigned int output_size;
    Optimizer *optimizer;
    Tensor input_tensor_cache;
    Tensor input_tensor_w_bias;

public:
    FullyConnected(unsigned int input_size, unsigned int output_size, Optimizer *optimizer)
    {
        this->input_size = input_size;
        this->output_size = output_size;
        // Declare weights tensor size to include bias
        this->weights = Tensor::Zero(this->input_size + 1, this->output_size);
        this->trainable = true;
        this->optimizer = optimizer;
    };

    ~FullyConnected() {}

    /**
     * @author Hamiz Ali, Lam Tran
     * @since 24-01-2025
     * @brief Initialize the weights and bias using the provided initializer
     * @param weights_initializer
     * @param bias_initializer
     */
    void initialize(Initializer *weights_initializer, Initializer *bias_initializer)
    {
        weights_initializer->initialize(input_size, output_size);
        bias_initializer->initialize(1, output_size);

        this->weights.topRows(input_size) = weights_initializer->getWeights();
        this->weights.bottomRows(1) = bias_initializer->getWeights();
    }

    /**
     * @author Hamiz Ali, Lam Tran
     * @since 24-01-2025
     * @brief Forward pass through the fully connected layer
     * @param input_tensor
     * @return Tensor
     */
    Tensor forward(const Tensor &input_tensor) override
    {
        input_tensor_cache = input_tensor;

        input_tensor_w_bias = Tensor::Zero(input_tensor.rows(), input_tensor.cols() + 1);
        input_tensor_w_bias.leftCols(input_tensor.cols()) = input_tensor;
        input_tensor_w_bias.rightCols(1).setOnes();

        return input_tensor_w_bias * this->weights;
    }

    /**
     * @author Hamiz Ali, , Lam Tran
     * @since 24-01-2025
     * @brief Backward pass through the fully connected layer
     * @param error_tensor
     * @return Tensor
     */
    Tensor backward(const Tensor &error_tensor) override
    {
        Tensor gradient_weights = input_tensor_w_bias.transpose() * error_tensor;
        this->weights = optimizer->updateWeights(this->weights, gradient_weights);
        Tensor weights_no_bias = this->weights.topRows(this->weights.rows() - 1);
        return error_tensor * weights_no_bias.transpose();
    }
};
