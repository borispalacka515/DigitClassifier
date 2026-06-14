#pragma once

#include <vector>

enum class ActivationType
{
    None,
    ReLU,
    Softmax
};

namespace Activation
{
    double ReLU(double x);
    double ReLUDerivative(double x);

    std::vector<double> ReLU(const std::vector<double>& x);
    std::vector<double> softmax(const std::vector<double>& x);
}