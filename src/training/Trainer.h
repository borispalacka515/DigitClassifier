#pragma once

#include "Model.h"
#include "TrainingConfig.h"
#include "Dataset.h"

class Trainer
{
public: 
	void train(
		const Model& model,
		const TrainingConfig& config,
		const Dataset& dataset);
};