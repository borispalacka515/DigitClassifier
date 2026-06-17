#pragma once

#include "Activation.h"

#include <vector>

class DenseLayer
{
private:
    int m_inputSize = 0;
    int m_outputSize = 0;

    std::vector<double> m_weights;
    std::vector<double> m_biases;

    ActivationType m_activation = ActivationType::None;

public:
    DenseLayer(
        int inputSize,
        int outputSize,
        ActivationType activation
    );

    int inputSize() const;
    int outputSize() const;
    ActivationType activation() const;

    const std::vector<double>& weights() const;
    const std::vector<double>& biases() const;

    void setParameters(
        const std::vector<double>& weights,
        const std::vector<double>& biases
    );

    std::vector<double> forward(
        const std::vector<double>& input
    ) const;
};
