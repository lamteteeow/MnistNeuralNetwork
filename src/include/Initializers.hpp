#pragma once

#include "Eigen/Dense"
#include <random>

using Tensor = Eigen::MatrixXd;

class Initializer
{
protected:
  unsigned long seed;
  Tensor weights;

public:
  Initializer(unsigned long seed = 0) : seed(seed) {}
  virtual ~Initializer() = default;

  // Pure virtual function for weight initialization
  virtual void initialize(unsigned int fan_in, unsigned int fan_out) = 0;

  const Tensor &getWeights() const { return weights; }
};

class Xavier final : public Initializer
{
private:
    std::mt19937 gen;

public:
  Xavier(unsigned long seed = 0) : Initializer(seed), gen(seed) {}

    /**
     * @author Lam Tran, Hamiz Ali
     * @since 28-01-2025
     * @brief Initialize weights using Xavier initialization
     * @param fan_in
     * @param fan_out
     */
    void initialize(unsigned int fan_in, unsigned int fan_out) override
    {
        weights = Tensor::Zero(fan_in, fan_out);
        const double sigma = std::sqrt(2.0 / (fan_in + fan_out));
        std::normal_distribution<double> distribution(0.0, sigma);

        weights = Tensor::NullaryExpr(fan_in, fan_out, [&]() { return distribution(gen); });
    }
};

class He final : public Initializer
{
private:
    std::mt19937 gen;

public:
    He(unsigned long seed = 0) : Initializer(seed), gen(seed) {}

    /**
     * @author Lam Tran, Hamiz Ali
     * @since 28-01-2025
     * @brief Initialize weights using He initialization
     * @param fan_in
     * @param fan_out
     */
    void initialize(unsigned int fan_in, unsigned int fan_out) override
    {
        weights = Tensor::Zero(fan_in, fan_out);
        const double sigma = std::sqrt(2.0 / fan_in);
        std::normal_distribution<double> distribution(0.0, sigma);

        weights = Tensor::NullaryExpr(fan_in, fan_out, [&]() { return distribution(gen); });
    }
};