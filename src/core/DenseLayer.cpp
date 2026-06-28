#include "DenseLayer.h"

#include <stdexcept>


DenseLayer::DenseLayer(
    int inputSize,
    int outputSize,
    ActivationType activation
)
    : m_inputSize(inputSize),
    m_outputSize(outputSize),
    m_activation(activation),
    m_weights(inputSize* outputSize),
    m_biases(outputSize),
    m_hasCachedForward(false),
    m_hasGradients(false)
{
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
    m_hasCachedForward = true;

    switch (m_activation)
    {
    case ActivationType::None:
        return m_lastOutput = z;

    case ActivationType::ReLU:
        return m_lastOutput = Activation::ReLU(z);

    case ActivationType::Softmax:
        return m_lastOutput = Activation::softmax(z);

    default:
        throw std::runtime_error("Unsupported activation type.");
    }
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

    for (size_t i = 0; i < m_outputSize; i++)
    {
        delta[i] = outputGradient[i]; //* activation derivative(m_lastZ[i])
    }
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