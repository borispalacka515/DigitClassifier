#pragma once

#include "LossFunction.h"

#include <stdexcept>

class TrainingConfig
{
private:

	int m_batchSize;
	int m_epochCount;
	double m_learningRate;
	LossFunctionType m_lossFunctionType = LossFunctionType::None;

public:

	TrainingConfig(
		int batchSize,
		int epochCount,
		double learningRate,
		LossFunctionType lossFunctionType
	);

	int batchSize() const;
	int epochCount() const;
	double learningRate() const;
	LossFunctionType lossFunctionType() const;

private:

	void validate() const;
};