#include "Dataset.h"

Dataset::Dataset()
{
	m_samples = std::vector<Sample>();
}

void Dataset::addSample(const Sample& sample) {
	m_samples.push_back(sample);
}

const Sample& Dataset::getSample(size_t index) const
{
	return m_samples.at(index);
}

size_t Dataset::size() const
{
	return m_samples.size();
}

bool Dataset::empty() const
{
	return m_samples.empty();
}