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

    std::vector<double> crossEntropyGradient(
        const std::vector<double>& predicted,
        int targetLabel
    )
    {
        const double epsilon = 1e-12;

        std::vector<double> gradient(predicted.size(), 0);
        gradient.at(targetLabel) = -1 / std::max(predicted.at(targetLabel), epsilon);

        return gradient;
    }

    double meanSquaredError(
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

    std::vector<double> meanSquaredErrorGradient(
        const std::vector<double>& predicted,
        int targetLabel
    )
    {
        std::vector<double> gradient(predicted.size());

        for (size_t i = 0; i < predicted.size(); ++i)
        {
            const double target = (i == targetLabel) ? 1.0 : 0.0;

            gradient[i] = 2.0 * (predicted.at(i) - target);
        }

        return gradient;
    }

}