#pragma once

#include <vector>
#include <string>

enum class ActivationType
{
    None,
    ReLU,
    Softmax
};

namespace Activation
{
    std::string toString(ActivationType type);
    ActivationType fromString(const std::string& name);

    double ReLU(double x);
    double ReLUDerivative(double x);

    std::vector<double> ReLU(const std::vector<double>& x);
    std::vector<double> softmax(const std::vector<double>& x);
}