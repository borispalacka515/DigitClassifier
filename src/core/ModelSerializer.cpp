#include "ModelSerializer.h"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace ModelSerializer {

	void saveToFile(
		const Model& model,
		const std::filesystem::path& filePath)
	{
		nlohmann::json root;

		root["format"] = "DigitClassifierModel";
		
		nlohmann::json configJson;

		configJson["layerSizes"] = model.config().layerSizes();

		nlohmann::json activationJson = nlohmann::json::array();

		for (ActivationType type : model.config().layerActivations())
		{
			activationJson.push_back(
				Activation::toString(type)
			);
		}

		configJson["layerActivations"] = activationJson;

		root["config"] = configJson;

		nlohmann::json layersJson = nlohmann::json::array();

		for (size_t i = 0; i < model.layers().size(); i++)
		{
			nlohmann::json layerJson;
			layerJson["index"] = i;
			layerJson["weights"] = model.layerAt(i).weights;
			layerJson["biases"] = model.layerAt(i).biases;

			layersJson.push_back(layerJson);
		}

		root["layers"] = layersJson;

		std::ofstream file(filePath);

		if (!file)
		{
			throw std::runtime_error(
				"Could not open file for writing."
			);
		}

		file << root.dump(4);
	}

	Model loadFromFile(const std::filesystem::path& filePath)
	{

	}
}