#pragma once

#include "Eigen/Dense"

#include <random>

using Tensor = Eigen::MatrixXd;

class Xavier
{
public:
    Tensor weights;
    unsigned int seed;

    Xavier(unsigned int seed = 0) : seed(seed) {}

    /**
     * @author Lam Tran
     * @since 22-01-2025
     * @brief Initialize the weights using Xavier's initialization technique in combination with a seed for reproducibility using Mersenne twister engine
     * @param weights_shape
     * @param fan_in
     * @param fan_out
     * @return Tensor
     */
    Tensor initialize(const std::vector<int> &weights_shape, int fan_in, int fan_out)
    {
        double sigma = std::sqrt(2.0 / (fan_in + fan_out));
        std::mt19937 generator(this->seed);
        std::normal_distribution<double> distribution(0.0, sigma);

        this->weights.resize(weights_shape[0], weights_shape[1]);
        for (int i = 0; i < weights_shape[0]; ++i)
        {
            for (int j = 0; j < weights_shape[1]; ++j)
            {
                this->weights(i, j) = distribution(generator);
            }
        }
        return this->weights;
    }
};

class He
{
public:
    Tensor weights;
    unsigned int seed;

    He(unsigned int seed = 0) : seed(seed) {}

    /**
     * @author Lam Tran
     * @since 22-01-2025
     * @brief Initialize the weights using He's initialization technique in combination with a seed for reproducibility using Mersenne twister engine
     * @param weights_shape
     * @param fan_in
     * @param fan_out
     * @return Tensor
     */
    Tensor initialize(const std::vector<int> &weights_shape, int fan_in)
    {
        double sigma = std::sqrt(2.0 / fan_in);
        std::mt19937 generator(this->seed);
        std::normal_distribution<double> distribution(0.0, sigma);

        this->weights.resize(weights_shape[0], weights_shape[1]);
        for (int i = 0; i < weights_shape[0]; ++i)
        {
            for (int j = 0; j < weights_shape[1]; ++j)
            {
                this->weights(i, j) = distribution(generator);
            }
        }
        return this->weights;
    }
};