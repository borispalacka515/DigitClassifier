#pragma once

#include "Activation.h"

#include <vector>
#include <stdexcept>

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

	ActivationType layerActivationAt(size_t index) const;
	const std::vector<ActivationType>& layerActivations() const;

	void validate() const;
};

ModelConfig::ModelConfig(
	const std::vector<int>& layerSizes,
	const std::vector<ActivationType>& layerActivations
) :
	m_layerSizes(layerSizes),
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
				"All layer sizes in model must be greater the zero"
			);
		}
	}
}

int ModelConfig::denseLayerCount() const 
{
	return m_layerSizes.size() - 1;
}

int ModelConfig::layerSizeAt(size_t index) const
{
	return m_layerSizes[index];
}

const std::vector<int>& ModelConfig::layerSizes() const
{
	return m_layerSizes;
}

ActivationType ModelConfig::layerActivationAt(size_t index) const
{
	return m_layerActivations[index];
}

const std::vector<ActivationType>& ModelConfig::layerActivations() const
{
	return m_layerActivations;
}