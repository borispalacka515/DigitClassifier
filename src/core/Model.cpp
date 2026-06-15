#include "Model.h"
#include "Activation.h"

Model::Model(const ModelConfig& config)
{
    m_layers.clear();

    for (size_t i = 0; i < config.denseLayerCount(); i++)
    {
        int layerInputSize = config.layerSizeAt(i);
        int layerOutputSize = config.layerSizeAt(i + 1);
        ActivationType layerActivation = config.layerActivationAt(i);
        
        m_layers.emplace_back(layerInputSize, layerOutputSize, layerActivation);
    }
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