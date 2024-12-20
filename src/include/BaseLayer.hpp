#pragma once

#include <Eigen/Dense>

class BaseLayer
{
public:
    BaseLayer() : trainable(false), weights(Eigen::MatrixXd()) {}
    virtual ~BaseLayer() = default;

    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd &input_tensor) = 0;
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd &error_tensor) = 0;

    bool trainable;
    Eigen::MatrixXd weights;
};