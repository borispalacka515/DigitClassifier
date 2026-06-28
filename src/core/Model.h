#pragma once

#include "Activation.h"
#include "DenseLayer.h"
#include "ModelConfig.h"

#include <vector>
#include <stdexcept>

class Model
{
private:
    ModelConfig m_config;
    std::vector<DenseLayer> m_layers;

public:
    explicit Model(
        const ModelConfig& config
    );

    const ModelConfig& config() const;

    const std::vector<DenseLayer>& layers() const;
    const DenseLayer& layerAt(size_t index) const;

    void setLayerParameters(
        size_t index,
        const std::vector<double>& weights,
        const std::vector<double>& biases
    );


    std::vector<double> forward(
        const std::vector<double>& input
    );

    void backward(
        const std::vector<double>& outputGradient
    );

    int predict(const std::vector<double>& input);

    // Temporary Solution

    void updateLayerParameters(double learningRate);
};