#pragma once

#include "Activation.h"

#include <vector>

struct DenseLayer
{
    int inputSize = 0;
    int outputSize = 0;

    std::vector<double> weights;
    std::vector<double> biases;

    ActivationType activation = ActivationType::None;

    DenseLayer(
        int inputSize,
        int outputSize,
        ActivationType activation
    );

    std::vector<double> forward(
        const std::vector<double>& input
    ) const;
};
