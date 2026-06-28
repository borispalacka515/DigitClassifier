#include "Trainer.h"
#include "LossFunction.h"

#include <vector>
#include <stdexcept>
#include <numeric>

Trainer::Trainer()
	: m_rng(std::random_device{}())
{
}

std::vector<size_t> Trainer::createShuffledIndices(size_t indicesSize)
{
	std::vector<size_t> indices(indicesSize);

	std::iota(indices.begin(), indices.end(), 0);

	std::shuffle(indices.begin(), indices.end(), m_rng);

	return indices;
}

std::vector<std::vector<std::reference_wrapper<const Sample>>>
Trainer::createRandomBatches(
	const Dataset& dataset,
	size_t batchSize
)
{
	std::vector<size_t> shuffledIndices =
		createShuffledIndices(dataset.size());

	const size_t batchCount =
		(dataset.size() + batchSize - 1) / batchSize;

	std::vector<std::vector<std::reference_wrapper<const Sample>>> batches;
	batches.reserve(batchCount);

	for (size_t batchStart = 0; batchStart < dataset.size(); batchStart += batchSize)
	{
		const size_t batchEnd = std::min(
			batchStart + batchSize,
			dataset.size()
		);

		std::vector<std::reference_wrapper<const Sample>> batch;
		batch.reserve(batchEnd - batchStart);

		for (size_t i = batchStart; i < batchEnd; ++i)
		{
			const size_t sampleIndex = shuffledIndices[i];

			batch.push_back(std::cref(dataset.getSample(sampleIndex)));
		}

		batches.push_back(std::move(batch));
	}

	return batches;
}

void Trainer::train(
	Model& model,
	const TrainingConfig& config,
	const Dataset& dataset
)
{
	validateTrainingSetup(model, config, dataset);

	for (size_t epoch = 0; epoch < config.epochCount(); ++epoch)
	{
		const auto& batches = createRandomBatches(dataset, config.batchSize());
		
		for (const auto& batch : batches)
		{
			trainBatch(model, config, batch);
		}
	}
}

void Trainer::trainBatch(
	Model& model,
	const TrainingConfig& config,
	const std::vector<std::reference_wrapper<const Sample>>& batch
) const
{
	double totalLoss = 0.0;

	for (const auto& sampleReference : batch)
	{
		//forward
		const Sample& sample = sampleReference.get();
		
		std::vector<double> output = model.forward(sample.input);

		//loss 
		double loss = LossFunction::crossEntropy(output, sample.label);
		totalLoss += loss;

		//model.backward(sample);
	}
}


void Trainer::validateTrainingSetup(
	const Model& model,
	const TrainingConfig& config,
	const Dataset& dataset
) const 
{
	if (config.batchSize() >= dataset.size())
	{
		throw std::invalid_argument(
			"Batch size can't be greater then size of the dataset."
		);
	}

	if (model.config().inputSize() !=
		dataset.getSample(0).input.size())
	{
		throw std::invalid_argument(
			"Model input layer size doesn't match dataset sample size."
		);
	}

}