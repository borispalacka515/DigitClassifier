#include "ModelConfig.h"

#include <stdexcept>

ModelConfig::ModelConfig(
    const std::vector<int>& layerSizes,
    const std::vector<ActivationType>& layerActivations
)
    : m_layerSizes(layerSizes),
    m_layerActivations(layerActivations)
{
    validate();
}

void ModelConfig::validate() const
{
    if (m_layerSizes.size() < 2)
    {
        throw std::invalid_argument(
            "ModelConfig must contain at least input and output layer sizes."
        );
    }

    for (int layerSize : m_layerSizes)
    {
        if (layerSize <= 0)
        {
            throw std::invalid_argument(
                "All layer sizes in model must be greater than zero."
            );
        }
    }

    if (m_layerActivations.size() != m_layerSizes.size() - 1)
    {
        throw std::invalid_argument(
            "Number of activations must match number of dense layers."
        );
    }
}

int ModelConfig::denseLayerCount() const
{
    return static_cast<int>(m_layerSizes.size()) - 1;
}

int ModelConfig::layerSizeAt(size_t index) const
{
    return m_layerSizes.at(index);
}

const std::vector<int>& ModelConfig::layerSizes() const
{
    return m_layerSizes;
}

int ModelConfig::inputSize() const
{
    return m_layerSizes.front();
}

int ModelConfig::outputSize() const
{
    return m_layerSizes.back();
}

ActivationType ModelConfig::layerActivationAt(size_t index) const
{
    return m_layerActivations.at(index);
}

const std::vector<ActivationType>& ModelConfig::layerActivations() const
{
    return m_layerActivations;
}