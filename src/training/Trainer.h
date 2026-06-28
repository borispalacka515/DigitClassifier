#pragma once

#include "Model.h"
#include "TrainingConfig.h"
#include "Dataset.h"

#include <random>

struct TrainingBatchResult
{
	double averageLoss = 0.0;
	double accuracy = 0.0;
};

class Trainer
{
private:
	std::mt19937 m_rng;
public: 
	Trainer();

	void train(
		Model& model,
		const TrainingConfig& config,
		const Dataset& dataset);

private:
	void validateTrainingSetup(
		const Model& model,
		const TrainingConfig& config,
		const Dataset& dataset
	) const;

	TrainingBatchResult trainBatch(
		Model& model,
		const TrainingConfig& config,
		const std::vector<std::reference_wrapper<const Sample>>& batch
	) const;

	std::vector<size_t> createShuffledIndices(size_t indicesSize);

	std::vector<std::vector<std::reference_wrapper<const Sample>>> createRandomBatches(
		const Dataset& dataset,
		size_t batchSize
	);
};