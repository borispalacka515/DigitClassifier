#include "Dataset.h"

Dataset::Dataset()
{
	samples = std::vector<Sample>();
}

void Dataset::addSample(const Sample& sample) {
	samples.push_back(sample);
}

const Sample& Dataset::getSample(size_t index) const
{
	return samples.at(index);
}

size_t Dataset::size() const
{
	return samples.size();
}

bool Dataset::empty() const
{
	return samples.empty();
}