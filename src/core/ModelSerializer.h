#pragma once

#include "Model.h"
#include "ModelConfig.h"

#include <string>
#include <filesystem>

#include <nlohmann/json.hpp>

namespace ModelSerializer
{
	void saveToFile(
		const Model& model,
		const std::filesystem::path& filePath
	);

	Model loadFromFile(const std::filesystem::path& filePath);
}