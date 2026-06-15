#pragma once

#include "Activation.h"
#include "DenseLayer.h"
#include "ModelConfig.h"

#include <vector>
#include <stdexcept>

class Model
{
private:
    std::vector<DenseLayer> m_layers;

public:
    explicit Model(
        const ModelConfig& config
    );

    std::vector<double> forward(const std::vector<double>& input);
    int predict(const std::vector<double>& input);
};