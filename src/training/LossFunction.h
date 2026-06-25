#pragma once

#include <vector>

namespace LossFunction 
{
	double crossEntropy(
		const std::vector<double>& predicted,
		int targetLabel
	);

	double MeanSquareError(
		const std::vector<double>& predicted,
		int targetLabel
	);
}