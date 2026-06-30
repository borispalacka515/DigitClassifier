#pragma once
#include <vector>

struct Sample
{
	std::vector<double> input;
	int label = -1;
	int width = 0;
	int height = 0;
};

class Dataset
{
private:
	std::vector<Sample> m_samples;
public:
	Dataset();
	void addSample(const Sample& sample);
	const Sample& getSample(size_t index) const;

	size_t size() const;
	bool empty() const;
};