#pragma once

#include <stdexcept>

class TrainingConfig
{
private:

	int m_batchSize;
	int m_epochCount;
	double m_learningRate;

public:

	TrainingConfig(
		int batchSize,
		int epochCount,
		double learningRate
	);

	int batchSize() const;
	int epochCount() const;
	double learningRate() const;

private:

	void validate() const;
};