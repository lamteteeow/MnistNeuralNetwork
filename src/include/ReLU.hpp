#pragma once

#include "BaseLayer.hpp"
#include <Eigen/Dense>

class ReLU : public BaseLayer
{
private:
    Eigen::MatrixXd relu_gradient;

public:
    ReLU() {}
    ~ReLU() {}

    Eigen::MatrixXd forward(const Eigen::MatrixXd &input_tensor)
    {
        relu_gradient = (input_tensor.array() > 0).cast<double>();
        return input_tensor.cwiseMax(0.0);
    }

    Eigen::MatrixXd backward(const Eigen::MatrixXd &error_tensor)
    {
        return error_tensor.cwiseProduct(relu_gradient);
    }
};