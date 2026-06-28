#include "Activation.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace Activation {

	ActivationType fromString(const std::string& name)
	{
		if (name == "None")
		{
			return ActivationType::None;
		}

		if (name == "ReLU")
		{
			return ActivationType::ReLU;
		}

		if (name == "Softmax")
		{
			return ActivationType::Softmax;
		}

		throw std::runtime_error("Uknown activation type.");
	}

	std::string toString(ActivationType type)
	{
		switch (type)
		{
		case ActivationType::None:
			return "None";

		case ActivationType::ReLU:
			return "ReLU";

		case ActivationType::Softmax:
			return "Softmax";

		default:
			throw std::runtime_error("Uknown activation type.");
		}
	}

	double ReLU(double x)
	{
		return std::max(0.0, x);
	}

	double ReLUDerivative(double x)
	{
		return (x > 0.0) ? 1.0 : 0.0;
	}

	std::vector<double> ReLU(const std::vector<double>& x)
	{
		std::vector<double> result(x.size());

		for (size_t i = 0; i < x.size(); ++i)
		{
			result[i] = ReLU(x[i]);
		}
		return result;
	}

	std::vector<double> ReLUDerivative(const std::vector<double>& x)
	{
		std::vector<double> result(x.size());

		for (size_t i = 0; i < x.size(); ++i)
		{
			result[i] = ReLUDerivative(x[i]);
		}
		return result;
	}
	
	std::vector<double> softmax(const std::vector<double>& x) 
	{
		std::vector<double> result(x.size());
		double maxValue = *std::max_element(x.begin(), x.end());
		double sum = 0.0;

		for (size_t i = 0; i < x.size(); ++i)
		{
			result[i] = std::exp(x[i] - maxValue);
			sum += result[i];
		}
		for (size_t i = 0; i < x.size(); ++i)
		{
			result[i] /= sum;
		}
		
		return result;
	}

	std::vector<std::vector<double>> softmaxDerivative(
		const std::vector<double>& x
	)
	{
		const std::vector<double> a = softmax(x);
		const size_t size = a.size();

		std::vector<std::vector<double>> jacobian(
			size,
			std::vector<double>(size, 0.0)
		);

		for (size_t i = 0; i < size; ++i)
		{
			for (size_t j = 0; j < size; ++j)
			{
				const double kroneckerDelta = (i == j) ? 1.0 : 0.0;

				jacobian[i][j] = a[i] * (kroneckerDelta - a[j]);
			}
		}

		return jacobian;
	}

	std::vector<double> activate(const std::vector<double>& x, ActivationType type)
	{
		switch (type)
		{
		case ActivationType::None:
			return x;

		case ActivationType::ReLU:
			return Activation::ReLU(x);

		case ActivationType::Softmax:
			return Activation::softmax(x);

		default:
			throw std::runtime_error("Unsupported activation type.");
		}
	}

	std::vector<double> activateDerivative(const std::vector<double>& x, ActivationType type)
	{
		switch (type)
		{
		case ActivationType::None:
			return std::vector<double>(x.size(), 1.0);

		case ActivationType::ReLU:
			return Activation::ReLUDerivative(x);

		default:
			throw std::runtime_error("Unsupported activation type.");
		}
	}

	double activateDerivative(double x, ActivationType type)
	{
		switch (type)
		{
		case ActivationType::None:
			return 1;

		case ActivationType::ReLU:
			return Activation::ReLUDerivative(x);

		default:
			throw std::runtime_error("Unsupported activation type.");
		}
	}
}