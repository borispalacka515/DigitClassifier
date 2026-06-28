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
    std::vector<double> ReLUDerivative(const std::vector<double>& x);

    std::vector<double> softmax(const std::vector<double>& x);
    std::vector<std::vector<double>> softmaxDerivative(const std::vector<double>& x);

    std::vector<double> activate(const std::vector<double>& x, ActivationType type);
    std::vector<double> activateDerivative(const std::vector<double>& x, ActivationType type);

    double activateDerivative(double x, ActivationType type);
}