#pragma once

#include "Activation.h"
#include "DenseLayer.h"

#include <vector>
#include <stdexcept>

class Model
{
private:
    std::vector<DenseLayer> layers;

public:
    explicit Model(
        const std::vector<double>& b1,
        const std::vector<double>& b2,
        const std::vector<std::vector<double>>& W1,
        const std::vector<std::vector<double>>& W2
    );

    std::vector<double> forward(const std::vector<double>& input);
    int predict(const std::vector<double>& input);
};