#pragma once

#include "Model.h"

#include <string>
#include <filesystem>

namespace ModelSerializer
{
	void saveToFile(const Model& model, const std::filesystem::path& filePath);

	Model loadFromFile(const std::filesystem::path& filePath);
}