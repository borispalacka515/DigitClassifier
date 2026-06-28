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

    // forward cache
    std::vector<double> m_lastInput;
    std::vector<double> m_lastZ;
    std::vector<double> m_lastOutput;
    bool m_hasCachedForward;

    // gradient cache
    std::vector<std::vector<double>> m_weightGradients;
    std::vector<double> m_biasGradients;
    bool m_hasGradients;

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

    bool hasCachedForward() const;

    void setParameters(
        const std::vector<double>& weights,
        const std::vector<double>& biases
    );

    std::vector<double> forward(
        const std::vector<double>& input
    );

    std::vector<double> backward(
        const std::vector<double>& outputGradient
    );
};
