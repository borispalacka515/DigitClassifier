#include "DenseLayer.h"

#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <random>


DenseLayer::DenseLayer(
    int inputSize,
    int outputSize,
    ActivationType activation,
    WeightInitializationType initType
)
    : m_inputSize(inputSize),
    m_outputSize(outputSize),
    m_activation(activation),
    m_weightInitializationType(initType),
    m_weights(inputSize* outputSize),
    m_biases(outputSize),
    m_hasCachedForward(false),
    m_hasAccumulatedGradients(false),
    m_accumulatedGradientCount(0)
{
    m_accumulatedBiasGradients = std::vector<double>(m_outputSize);
    m_accumulatedWeightGradients = std::vector<std::vector<double>>(m_outputSize, std::vector<double>(m_inputSize));

    initializeWeights();
    initializeBiases();
}

void DenseLayer::initializeWeights()
{
    std::mt19937 rng(std::random_device{}());

    double standardDeviation = 0.0;

    switch (m_weightInitializationType)
    {
    case WeightInitializationType::None:
        return;

    case WeightInitializationType::He:
        standardDeviation =
            std::sqrt(2.0 / static_cast<double>(m_inputSize));
        break;
    default:
        throw std::logic_error(
            "Unsupported weith initialization type."
        );
    }

    std::normal_distribution<double> distribution(0.0, standardDeviation);

    for (double& weight : m_weights)
    {
        weight = distribution(rng);
    }
}

void DenseLayer::initializeBiases()
{
    m_biases = std::vector<double>(m_outputSize, 0.0);
}


std::vector<double> DenseLayer::forward(
    const std::vector<double>& input
)
{
    if (input.size() != m_inputSize)
    {
        throw std::invalid_argument(
            "Input size does not match DenseLayer input size."
        );
    }

    std::vector<double> z(m_outputSize);

    for (size_t i = 0; i < m_outputSize; ++i)
    {
        z[i] = m_biases[i];

        for (size_t j = 0; j < m_inputSize; ++j)
        {
            z[i] += m_weights[i * m_inputSize + j] * input[j];
        }
    }

    m_lastInput = input;
    m_lastZ = z;
    m_lastOutput = Activation::activate(z, m_activation);
    m_hasCachedForward = true;

    return m_lastOutput;
}

std::vector<double> DenseLayer::backward(
    const std::vector<double>& outputGradient
)
{
    if (!m_hasCachedForward)
    {
        throw std::logic_error(
            "Layer backward propagation requires a valid cached forward propagation."
        );
    }

    std::vector<double> delta(m_outputSize);

    if(m_activation == ActivationType::Softmax)
    {
        std::vector < std::vector<double>> jacobian = Activation::softmaxDerivative(m_lastZ);
        for (size_t i = 0; i < m_outputSize; i++)
        {
            delta[i] = 0;

            for (size_t j = 0; j < m_outputSize; j++)
            {
                delta[i] += jacobian[i][j] * outputGradient[j];
            }
        }

    }
    else
    {
        for (size_t i = 0; i < m_outputSize; i++)
        {
            delta[i] = outputGradient[i] * Activation::activateDerivative(m_lastZ[i], m_activation);
        }
    }

    for (size_t i = 0; i < m_outputSize; i++)
    {
        m_accumulatedBiasGradients[i] += delta[i];

        for (size_t j = 0; j < m_inputSize; j++)
        {
            m_accumulatedWeightGradients[i][j] += delta[i] * m_lastInput[j];
        }
    }

    m_hasAccumulatedGradients = true;
    m_accumulatedGradientCount++;

    std::vector<double> inputGradient(m_inputSize);

    for (size_t i = 0; i < m_inputSize; i++)
    {
        inputGradient[i] = 0;

        for (size_t j = 0; j < m_outputSize; j++)
        {
            inputGradient[i] += m_weights[j * m_inputSize + i] * delta[j];
        }
    }

    return inputGradient;
}

void DenseLayer::setParameters(
    const std::vector<double>& weights,
    const std::vector<double>& biases
)
{
    if (weights.size() != m_inputSize * m_outputSize)
    {
        throw std::invalid_argument(
            "Invalid weights size for DenseLayer."
        );
    }

    if (biases.size() != m_outputSize)
    {
        throw std::invalid_argument(
            "Invalid biases size for DenseLayer."
        );
    }

    m_weights = weights;
    m_biases = biases;

    m_hasCachedForward = false;
}

int DenseLayer::inputSize() const
{
    return m_inputSize;
}

int DenseLayer::outputSize() const
{
    return m_outputSize;
}

ActivationType DenseLayer::activation() const
{
    return m_activation;
}

const std::vector<double>& DenseLayer::weights() const
{
    return m_weights;
}

const std::vector<double>& DenseLayer::biases() const
{
    return m_biases;
}

bool DenseLayer::hasCachedForward() const
{
    return m_hasCachedForward;
}

void DenseLayer::clearCache()
{
    m_lastInput.clear();
    m_lastZ.clear();
    m_lastOutput.clear();
    m_hasCachedForward = false;
}

void DenseLayer::clearGradients()
{
    m_accumulatedBiasGradients = std::vector<double>(m_outputSize);
    m_accumulatedWeightGradients = std::vector<std::vector<double>>(m_outputSize, std::vector<double>(m_inputSize));
    m_hasAccumulatedGradients = false;
    m_accumulatedGradientCount = 0;
}

// Temporary solution

void DenseLayer::updateParameters(double learningRate)
{
    if (!m_hasAccumulatedGradients)
    {
        throw std::logic_error(
            "Parameter update requires accumulated gradients."
        );
    }

    for (size_t i = 0; i < m_outputSize; i++)
    {
        m_biases[i] -= learningRate * m_accumulatedBiasGradients[i] / m_accumulatedGradientCount;

        for (size_t j = 0; j < m_inputSize; j++)
        {
            m_weights[i * m_inputSize + j] -= learningRate * m_accumulatedWeightGradients[i][j] / m_accumulatedGradientCount;
        }
    }

    clearCache();
    clearGradients();
}