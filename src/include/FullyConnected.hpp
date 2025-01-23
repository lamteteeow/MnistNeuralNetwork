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
     * @author Lam Tran
     * @since 23-01-2025
     * @brief Initialize the weights and bias using the provided initializer
     * @param weights_initializer
     * @param bias_initializer
     */
    void initialize(Initializer *weights_initializer, Initializer *bias_initializer)
    {
        weights_initializer->initialize(input_size - 1, output_size);
        bias_initializer->initialize(1, output_size);

        this->weights.topRows(input_size) = weights_initializer->getWeights();
        this->weights.bottomRows(1) = bias_initializer->getWeights();
    }

    /**
     * @author Lam Tran
     * @since 23-01-2025
     * @brief Forward pass through the fully connected layer
     * @param input_tensor
     * @return Tensor
     */
    Tensor forward(const Tensor &input_tensor) override
    {
        input_tensor_cache = Tensor::Zero(input_tensor.rows(), input_tensor.cols() + 1);
        input_tensor_cache.leftCols(input_tensor.cols()) = input_tensor;
        input_tensor_cache.rightCols(1).setOnes();
        return input_tensor_cache * this->weights;
    }

    /**
     * @author Lam Tran
     * @since 23-01-2025
     * @brief Backward pass through the fully connected layer
     * @param error_tensor
     * @return Tensor
     */
    Tensor backward(const Tensor &error_tensor) override
    {
        // Calculate gradients with respect to weights
        Tensor gradient_weights = input_tensor_cache.transpose() * error_tensor;

        // Update weights using the optimizer
        // TODO: check if fc1 only or both fc1 and fc2?
        this->weights = optimizer->updateWeights(weights, gradient_weights);

        // Calculate and return gradient with respect to inputs
        return error_tensor * this->weights.transpose();
    }
};
