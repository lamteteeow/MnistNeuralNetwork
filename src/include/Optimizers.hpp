#pragma once

#include "Eigen/Dense"

using Tensor = Eigen::MatrixXd;

class Optimizer
{
private:
    bool trainable = false;

public:
    virtual ~Optimizer() = default;

    virtual Tensor updateWeights(Tensor &weights, Tensor &gradient) = 0;
};

class SGD final : public Optimizer
{
private:
    double learningRate;

public:
    SGD() : learningRate(0.001) {}
    SGD(double learningRate) : learningRate(learningRate) {}
    ~SGD() override {}

    /**
     * @author Lam Tran
     * @since 20-12-2024
     * @brief Adjust the weights using Stochastic Gradient Descent between iterations (not epochs)
     * @param weights
     * @param gradient
     * @return Tensor
     */
    Tensor updateWeights(Tensor &weights, Tensor &gradient) override
    {
        return (weights - learningRate * gradient);
    }
};

// TODO: Finish ADAM implementation
class ADAM final : public Optimizer
{
private:
    double learningRate;
    double beta1;
    double beta2;
    double epsilon;
    Tensor m;
    Tensor v;
    int t;

public:
    ADAM() : learningRate(0.001), beta1(0.9), beta2(0.999), epsilon(1e-8), t(0) {}
    ADAM(double learningRate, double beta1, double beta2, double epsilon) : learningRate(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}
    ~ADAM() override {}

    /**
     * @author Lam Tran
     * @since 20-12-2024
     * @brief Adjust the weights using ADAM optimization algorithm
     * @param weights
     * @param gradient
     * @return Tensor
     */
    Tensor updateWeights(Tensor &weights, Tensor &gradient) override
    {
        t++;
        m = beta1 * m + (1 - beta1) * gradient;
        v = beta2 * v + (1 - beta2) * gradient.cwiseProduct(gradient);
        Tensor m_hat = m / (1 - std::pow(beta1, t));
        Tensor v_hat = v / (1 - std::pow(beta2, t));
        return weights - learningRate * (m_hat.array() / (v_hat.array().sqrt() + epsilon)).matrix();
    }
};

// TODO: Maybe implement SGD with momentum