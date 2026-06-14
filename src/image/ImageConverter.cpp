#include "ImageConverter.h"
#include "MNIST.h"

#include <algorithm>

namespace ImageConverter
{
	QImage sampleToImage(const Sample& sample)
	{
		QImage image(
			sample.width, 
			sample.height, 
			QImage::Format_Grayscale8
		);

		for (size_t y = 0; y < sample.height; ++y)
		{
			unsigned char* row = image.scanLine(y);

			for (size_t x = 0; x < sample.width; ++x)
			{
				int index = y * sample.width + x;

				double value = std::clamp(sample.input[index], 0.0, 1.0);

				row[x] = static_cast<unsigned char>(value * 255);
			}
		}

		return image;
	}
}