#pragma once

#include "Activation.h"

#include <vector>

enum class WeightInitializationType
{
    None,
    He
};

class DenseLayer
{
private:
    int m_inputSize;
    int m_outputSize;

    std::vector<double> m_weights;
    std::vector<double> m_biases;

    ActivationType m_activation;

    WeightInitializationType m_weightInitializationType;

    // forward cache
    std::vector<double> m_lastInput;
    std::vector<double> m_lastZ;
    std::vector<double> m_lastOutput;
    bool m_hasCachedForward;

    // gradient cache
    std::vector<std::vector<double>> m_accumulatedWeightGradients;
    std::vector<double> m_accumulatedBiasGradients;
    bool m_hasAccumulatedGradients;
    size_t m_accumulatedGradientCount;

public:
    DenseLayer(
        int inputSize,
        int outputSize,
        ActivationType activation,
        WeightInitializationType initType = WeightInitializationType::He
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

    void clearCache();
    void clearGradients();

    // Temporary solution

    void updateParameters(double learningRate);

private:
    void initializeWeights();
    void initializeBiases();
};
