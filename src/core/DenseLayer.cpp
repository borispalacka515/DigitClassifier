#include "DenseLayer.h"

#include <stdexcept>


DenseLayer::DenseLayer(
    int inputSize,
    int outputSize,
    ActivationType activation
)
    : inputSize(inputSize),
    outputSize(outputSize),
    activation(activation),
    weights(inputSize* outputSize),
    biases(outputSize)
{}

std::vector<double> DenseLayer::forward(
    const std::vector<double>& input
) const
{
    if (input.size() != inputSize)
    {
        throw std::invalid_argument(
            "Input size does not match DenseLayer input size."
        );
    }

    std::vector<double> z(outputSize);

    for (size_t i = 0; i < outputSize; ++i)
    {
        z[i] = biases[i];

        for (size_t j = 0; j < inputSize; ++j)
        {
            z[i] += weights[i * inputSize + j] * input[j];
        }
    }

    switch (activation)
    {
    case ActivationType::None:
        return z;

    case ActivationType::ReLU:
        return Activation::ReLU(z);

    case ActivationType::Softmax:
        return Activation::softmax(z);

    default:
        throw std::runtime_error("Unsupported activation type.");
    }
}