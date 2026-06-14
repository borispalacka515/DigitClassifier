#include "MNISTLoader.h"
#include "MNIST.h"

#include <fstream>
#include <stdexcept>
#include <iostream>

namespace
{
	template <typename T>
	T readBigEndian(std::ifstream& file)
	{
		unsigned char bytes[sizeof(T)];
		file.read(reinterpret_cast<char*>(bytes), sizeof(T));

		if (!file)
		{
			throw std::runtime_error("Failed to read bytes from file.");
		}

		T value = 0;

		for (size_t i = 0; i < sizeof(T); ++i)
		{
			value += static_cast<T>(bytes[i] * pow(256, sizeof(T) - i - 1));
		}

		return value;
	}
}

namespace MNISTLoader
{
    Dataset load(
        const std::string& imagesPath,
        const std::string& labelsPath,
        std::function<void(int current, int total)> progressCallback)
    {
        Dataset MNISTDataset;

        std::ifstream imagesFile(imagesPath, std::ios::binary);
        std::ifstream labelsFile(labelsPath, std::ios::binary);

        if (!imagesFile.is_open())
        {
            throw std::runtime_error("Could not open MNIST images file: " + imagesPath);
        }

        if (!labelsFile.is_open())
        {
            throw std::runtime_error("Could not open MNIST labels file: " + labelsPath);
        }

        int imagesMagicNumber = readBigEndian<int>(imagesFile);
        int numberOfSamples = readBigEndian<int>(imagesFile);
        int numberOfRows = readBigEndian<int>(imagesFile);
        int numberOfColumns = readBigEndian<int>(imagesFile);

        int labelsMagicNumber = readBigEndian<int>(labelsFile);
        int numberOfLabels = readBigEndian<int>(labelsFile);

        if (imagesMagicNumber != MNIST::ImagesMagicNumber)
        {
            throw std::runtime_error("Invalid MNIST images file: wrong magic number.");
        }

        if (labelsMagicNumber != MNIST::LabelsMagicNumber)
        {
            throw std::runtime_error("Invalid MNIST labels file: wrong magic number.");
        }

        if (numberOfRows != MNIST::ImageHeight || numberOfColumns != MNIST::ImageWidth)
        {
            throw std::runtime_error("Unsupported MNIST image size. Expected 28x28.");
        }

        for (int sampleIndex = 0; sampleIndex < numberOfSamples; sampleIndex++)
        {
            Sample sample;
            sample.input.resize(MNIST::ImageSize);
            sample.height = MNIST::ImageHeight;
            sample.width = MNIST::ImageWidth;

            for (int pixelIndex = 0; pixelIndex < MNIST::ImageSize; ++pixelIndex)
            {
                unsigned char pixel = 0;

                imagesFile.read(reinterpret_cast<char*>(&pixel), sizeof(unsigned char));

                if (!imagesFile)
                {
                    throw std::runtime_error("Failed to read pixel from file: " + imagesPath);
                }

                sample.input[pixelIndex] = static_cast<double>(pixel) / 255.0;
            }

            unsigned char label = 0;

            labelsFile.read(reinterpret_cast<char*>(&label), sizeof(unsigned char));

            if (!labelsFile)
            {
                throw std::runtime_error("Failed to read label from file: " + labelsPath);
            }

            sample.label = static_cast<int>(label);

            MNISTDataset.addSample(sample);

            if (progressCallback)
            {
                progressCallback(sampleIndex + 1, numberOfSamples);
            }
        }

        return MNISTDataset;
    }
}

