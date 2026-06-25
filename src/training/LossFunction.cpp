#include "LossFunction.h"

#include <cmath>
#include <stdexcept>

namespace LossFunction
{
    double crossEntropy(
        const std::vector<double>& predicted,
        int targetLabel
    )
    {
        if (predicted.empty())
        {
            throw std::invalid_argument(
                "Predicted vector must not be empty."
            );
        }

        if (targetLabel < 0 ||
            targetLabel >= predicted.size())
        {
            throw std::out_of_range(
                "Target label is out of range."
            );
        }

        const double epsilon = 1e-12;

        double probability = predicted[targetLabel];

        if (probability < epsilon)
        {
            probability = epsilon;
        }

        return -std::log(probability);
    }

    double MeanSquareError(
        const std::vector<double>& predicted,
        int targetLabel
    )
    {
        if (predicted.empty())
        {
            throw std::invalid_argument(
                "Predicted vector must not be empty."
            );
        }

        std::vector<double> oneHot(predicted.size());
        oneHot.at(targetLabel) = 1.0;

        double mseSum = 0.0;

        for (size_t i = 0; i < predicted.size(); i++)
        {
            double error = predicted[i] - oneHot[i];
            mseSum += error * error;
        }

        mseSum /= predicted.size();

        return mseSum;
    }
}