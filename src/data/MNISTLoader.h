#pragma once

#include "Dataset.h"

#include <string>
#include <functional>

namespace MNISTLoader 
{
	Dataset load(
		const std::string& imagesPath,
		const std::string& labelsPath,
		std::function<void(int current, int total)> progressCallback = nullptr);
}