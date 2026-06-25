#include "TrainingConfig.h"

#include <stdexcept>


TrainingConfig::TrainingConfig(
	int batchSize,
	int epochCount,
	double learningRate
) :
	m_batchSize(batchSize),
	m_epochCount(epochCount),
	m_learningRate(learningRate)
{
	TrainingConfig::validate();
}

int TrainingConfig::batchSize() const
{
	return m_batchSize;
}

int TrainingConfig::epochCount() const
{
	return m_epochCount;
}

double TrainingConfig::learningRate() const
{
	return m_learningRate;
}

void TrainingConfig::validate() const
{
	if (m_batchSize <= 0 ||
		m_epochCount <= 0 ||
		m_learningRate <= 0)
	{
		throw std::invalid_argument(
			"Training configuration parameters must be positive."
		);
	}
}