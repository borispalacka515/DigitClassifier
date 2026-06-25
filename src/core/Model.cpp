#include "Model.h"
#include "Activation.h"
#include <stdexcept>

Model::Model(const ModelConfig& config) 
    : m_config(config)
{
    m_layers.clear();

    for (size_t i = 0; i < config.denseLayerCount(); ++i)
    {
        int layerInputSize = config.layerSizeAt(i);
        int layerOutputSize = config.layerSizeAt(i + 1);
        ActivationType layerActivation = config.layerActivationAt(i);
        
        m_layers.emplace_back(layerInputSize, layerOutputSize, layerActivation);
    }
}

const ModelConfig& Model::config() const
{
    return m_config;
}

const std::vector<DenseLayer>& Model::layers() const
{
    return m_layers;
}

const DenseLayer& Model::layerAt(size_t index) const
{
    return m_layers.at(index);
}

void Model::setLayerParameters(
    size_t index,
    const std::vector<double>& weights,
    const std::vector<double>& biases
)
{
    m_layers.at(index).setParameters(weights, biases);
}

std::vector<double> Model::forward(const std::vector<double>& input)
{   
    std::vector<double> output = input;

    for (const auto& layer : m_layers)
    {
        output = layer.forward(output);
    }

    return output;
}

int Model::predict(const std::vector<double>& input)
{
    std::vector<double> probabilities = forward(input);

    auto it = std::max_element(probabilities.begin(), probabilities.end());

    int predicted_digit = static_cast<int>(it - probabilities.begin());

    return predicted_digit;
}