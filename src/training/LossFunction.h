#pragma once

#include <vector>

enum class LossFunctionType
{
	CrossEntropy,
	MeanSquareError,
	None
};

namespace LossFunction
{
	double crossEntropy(
		const std::vector<double>& predicted,
		int targetLabel
	);

	std::vector<double> crossEntropyGradient(
		const std::vector<double>& predicted,
		int targetLabel
	);

	double meanSquaredError(
		const std::vector<double>& predicted,
		int targetLabel
	);

	std::vector<double> meanSquaredErrorGradient(
		const std::vector<double>& predicted,
		int targetLabel
	);

	std::vector<double> gradient(
		const std::vector<double>& predicted,
		int targetLabel,
		LossFunctionType type
	);
}