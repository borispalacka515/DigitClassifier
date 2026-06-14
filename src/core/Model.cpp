#include "Model.h"
#include "Activation.h"

Model::Model()
{
    W1.resize(HIDDEN_SIZE, std::vector<double>(INPUT_SIZE, 0.0));
    b1.resize(HIDDEN_SIZE, 0.0);

    W2.resize(OUTPUT_SIZE, std::vector<double>(HIDDEN_SIZE, 0.0));
    b2.resize(OUTPUT_SIZE, 0.0);
}

Model::Model(
    const std::vector<double>& b1,
    const std::vector<double>& b2,
    const std::vector<std::vector<double>>& W1,
    const std::vector<std::vector<double>>& W2
)
{
    this->b1 = b1;
    this->b2 = b2;
    this->W1 = W1;
    this->W2 = W2;
}

std::vector<double> Model::forward(const std::vector<double>& input)
{
    std::vector<double> z1(HIDDEN_SIZE);            // výstup W1*x + b1
    std::vector<double> a1(HIDDEN_SIZE);            // ReLU(z1)

    std::vector<double> z2(OUTPUT_SIZE);            // výstup W2*a1 + b2
    std::vector<double> p(OUTPUT_SIZE); // softmax(z2)

    // z1 = W1 * input + b1
    for (size_t i = 0; i < HIDDEN_SIZE; ++i)
    {
        z1[i] = 0;
        for (size_t j = 0; j < INPUT_SIZE; ++j)
        {
            z1[i] += W1[i][j] * input[j];
        }
        z1[i] += b1[i];
    }

    // a1 = ReLU(z1)
    a1 = Activation::ReLU(z1);

    // z2 = W2 * a1 + b2
    for (size_t i = 0; i < OUTPUT_SIZE; ++i)
    {
        z2[i] = 0;
        for (size_t j = 0; j < HIDDEN_SIZE; ++j)
        {
            z2[i] += W2[i][j] * a1[j];
        }
        z2[i] += b2[i];
    }

    // P = softmax(z2)
    p = Activation::softmax(z2);

    return p;
}

int Model::predict(const std::vector<double>& input)
{
    std::vector<double> probabilities = forward(input);

    auto it = std::max_element(probabilities.begin(), probabilities.end());

    int predicted_digit = static_cast<int>(it - probabilities.begin());

    return predicted_digit;
}