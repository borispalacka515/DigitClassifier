#include "ModelSerializer.h"

#include <fstream>
#include <stdexcept>

namespace 
{
	
	void applyParametersFromJson(
		Model& model,
		const nlohmann::json& layersJson
	)
	{
		if (!layersJson.is_array() || layersJson.size() == 0)
		{
			throw std::runtime_error(
				"Invalid Model configuration: incorrect layer structure."
			);
		}

		for (const auto& layerJson : layersJson) {
			if (!layerJson.is_object() ||
				!layerJson.contains("index") ||
				!layerJson.at("index").is_number_integer() ||
				!layerJson.contains("weights") ||
				!layerJson.at("weights").is_array() ||
				!layerJson.contains("biases") ||
				!layerJson.at("biases").is_array())
			{
				throw std::runtime_error(
					"Invalid Model configuration: incorrect layer structure"
				);
			}

			int index = layerJson.at("index").get<int>();

			std::vector<double> weights;

			for (const auto& weightJson : layerJson.at("weights"))
			{
				if (!weightJson.is_number())
				{
					throw std::runtime_error(
						"Invalid Model configuration: incorrect weight data type"
					);
				}


				weights.push_back(weightJson.get<double>());
			}

			std::vector<double> biases;

			for (const auto& biasJson : layerJson.at("biases"))
			{
				if (!biasJson.is_number())
				{
					throw std::runtime_error(
						"Invalid Model configuration: incorrect bias data type"
					);
				}

				biases.push_back(biasJson.get<double>());
			}

			model.setLayerParameters(index, weights, biases);
		}
	}

	ModelConfig configFromJson(
		const nlohmann::json& configJson
	)
	{
		if (!configJson.is_object() ||
			!configJson.contains("layerSizes") ||
			!configJson.at("layerSizes").is_array() ||
			!configJson.contains("layerActivations") ||
			!configJson.at("layerActivations").is_array())
		{
			throw std::runtime_error(
				"Invalid Model configuration."
			);
		}

		const nlohmann::json& layerSizesJson = configJson.at("layerSizes");

		std::vector<int> layerSizes;

		for (const auto& layerSizeJson : layerSizesJson)
		{
			if (!layerSizeJson.is_number_integer())
			{
				throw std::runtime_error(
					"Invalid Model configuration: incorrect layer size."
				);
			}

			layerSizes.push_back(layerSizeJson.get<int>());
		}

		const nlohmann::json& layerActivationsJson = configJson.at("layerActivations");

		std::vector<ActivationType> layerActivations;

		for (const auto& layerActivationJson : layerActivationsJson)
		{
			if (!layerActivationJson.is_string())
			{
				throw std::runtime_error(
					"Invalid Model configuration: incorrect layer activation."
				);
			}

			std::string layerActivation = layerActivationJson.get<std::string>();

			layerActivations.push_back(Activation::fromString(layerActivation));
		}

		return ModelConfig(layerSizes, layerActivations);
	}
}

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
			layerJson["weights"] = model.layerAt(i).weights();
			layerJson["biases"] = model.layerAt(i).biases();

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
		std::ifstream file(filePath); 

		if (!file)
		{
			throw std::runtime_error(
				"Could not open file for reading."
			);
		}

		nlohmann::json root;
		file >> root;

		if (!root.contains("format") ||
			!root.at("format").is_string() ||
			root.at("format").get<std::string>() != "DigitClassifierModel")
		{
			throw std::runtime_error(
				"Invalid data format in file."
			);
		}

		if (!root.contains("config") ||
			!root.at("config").is_object())
		{
			throw std::runtime_error(
				"Invalid Model configuration in file."
			);
		}

		if (!root.contains("layers") ||
			!root.at("layers").is_array())
		{
			throw std::runtime_error(
				"Invalid Model configuration in file."
			);
		}

		nlohmann::json configJson = root.at("config");

		ModelConfig config = configFromJson(configJson);

		nlohmann::json layersJson = root.at("layers");

		Model model(config);
		applyParametersFromJson(model, layersJson);

		return model;
	}
}