#pragma once

namespace MNIST
{
    constexpr int ImageWidth = 28;
    constexpr int ImageHeight = 28;
    constexpr int ImageSize = ImageWidth * ImageHeight;
    constexpr int ClassCount = 10;

    constexpr int ImagesMagicNumber = 2051;
    constexpr int LabelsMagicNumber = 2049;
}