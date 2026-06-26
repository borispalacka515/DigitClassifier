#pragma once

#include "Activation.h"

#include <cstddef>
#include <vector>

class ModelConfig
{
private:
    std::vector<int> m_layerSizes;
    std::vector<ActivationType> m_layerActivations;

public:
    ModelConfig(
        const std::vector<int>& layerSizes,
        const std::vector<ActivationType>& layerActivations
    );

    int denseLayerCount() const;

    int layerSizeAt(size_t index) const;
    const std::vector<int>& layerSizes() const;

    int inputSize() const;
    int outputSize() const;

    ActivationType layerActivationAt(size_t index) const;
    const std::vector<ActivationType>& layerActivations() const;

    void validate() const;
};