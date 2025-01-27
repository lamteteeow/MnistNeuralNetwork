#pragma once

#include "Eigen/Dense"
#include <random>

using Tensor = Eigen::MatrixXd;

class Initializer
{
protected:
    unsigned int seed;
    Tensor weights;

public:
    Initializer(unsigned int seed = 0) : seed(seed) {}
    virtual ~Initializer() = default;

    // Pure virtual function for weight initialization
    virtual void initialize(unsigned int fan_in, unsigned int fan_out) = 0;

    const Tensor &getWeights() const
    {
        return weights;
    }
};

class Xavier final : public Initializer
{
private:
    std::mt19937 gen;

public:
    Xavier(unsigned int seed = 0) : Initializer(seed), gen(seed) {}

    /**
     * @author Lam Tran, Hamiz Ali
     * @since 24-01-2025
     * @brief Initialize weights using Xavier initialization
     * @param rows
     * @param cols
     * @param fan_in
     * @param fan_out
     */
    void initialize(unsigned int fan_in, unsigned int fan_out) override
    {
        weights = Tensor::Zero(fan_in, fan_out);
        const double sigma = std::sqrt(2.0 / (fan_in + fan_out));
        std::normal_distribution<double> distribution(0.0, sigma);

        // Random number generators are not thread-safe
        // #pragma omp parallel for collapse(2)
        for (unsigned int i = 0; i < fan_in; i++)
        {
            for (unsigned int j = 0; j < fan_out; j++)
            {
                weights(i, j) = distribution(gen);
            }
        }
    }
};

class He final : public Initializer
{
private:
    std::mt19937 gen;

public:
    He(unsigned int seed = 0) : Initializer(seed), gen(seed) {}

    /**
     * @author Lam Tran, Hamiz Ali
     * @since 24-01-2025
     * @brief Initialize weights using He initialization
     * @param rows
     * @param cols
     * @param fan_in
     * @param fan_out
     */
    void initialize(unsigned int fan_in, unsigned int fan_out) override
    {
        weights = Tensor::Zero(fan_in, fan_out);
        const double sigma = std::sqrt(2.0 / fan_in);
        std::normal_distribution<double> distribution(0.0, sigma);

        // Random number generators are not thread-safe
        // #pragma omp parallel for collapse(2)
        for (unsigned int i = 0; i < fan_in; i++)
        {
            for (unsigned int j = 0; j < fan_out; j++)
            {
                weights(i, j) = distribution(gen);
            }
        }
    }
};