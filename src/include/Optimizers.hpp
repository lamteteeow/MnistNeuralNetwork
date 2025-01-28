#pragma once

#include "Eigen/Dense"
#include <mutex>

using Tensor = Eigen::MatrixXd;

class Optimizer
{
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

class ADAM final : public Optimizer
{
private:
    double learningRate;
    double beta1;
    double beta2;
    double epsilon;
    Tensor m;
    Tensor v;
    bool uninitialized = true;
    static inline std::mutex mtx;
    static inline int t = 0;

  public:
    ADAM() : learningRate(0.001), beta1(0.9), beta2(0.999), epsilon(1e-8), uninitialized(true) {}
    /**
     * @author Lam Tran
     * @since 20-12-2024
     * @brief Construct ADAM optimizer with learning rate, beta1, beta2, and epsilon
     * @param learningRate aka step size
     * @param beta1 Exponential decay rate for momentum term aka first moment estimates
     * @param beta2 Exponential decay rate for velocity term aka second-moment estimates
     * @param epsilon Small value to prevent division by zero
     * @param t Number of iterations (statically managed)
     * @param lambda Rate of decay for the moment estimates (not implemented)
     */
    ADAM(double learningRate, double beta1, double beta2, double epsilon)
        : learningRate(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon), uninitialized(true) {}
    ~ADAM() override {}

    /**
     * @author Lam Tran
     * @since 20-12-2024
     * @brief Adjust the weights using ADAM optimization algorithm
     * @param weights
     * @param gradient
     * @return Tensor
     */
    Tensor updateWeights(Tensor &weights, Tensor &gradient) override {
        // Initialize m and v if not initialized
        if (uninitialized) {
            m = Tensor::Zero(weights.rows(), weights.cols());
            v = Tensor::Zero(weights.rows(), weights.cols());
            uninitialized = false;
        }

        int t_temp;
        {
            std::lock_guard<std::mutex> lock(mtx);
            t_temp = ++t;
        }

        m = beta1 * m + (1 - beta1) * gradient;
        v = beta2 * v + (1 - beta2) * gradient.cwiseProduct(gradient);
        Tensor m_hat = m / (1 - std::pow(beta1, t_temp));
        Tensor v_hat = v / (1 - std::pow(beta2, t_temp));
        return weights - learningRate * (m_hat.array() / (v_hat.array().sqrt() + epsilon)).matrix();
    }
};

// TODO: Maybe implement SGD with momentum