#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_DigitClassifier.h"
#include "Dataset.h"
#include "SampleThumbnailModel.h"
#include "Model.h"
#include "DatasetViewerPage.h"

#include <memory>
#include <optional>

class DigitClassifier : public QMainWindow
{
    Q_OBJECT

private:
    Ui::DigitClassifierClass ui;

    std::optional<Model> m_model;

public:
    DigitClassifier(QWidget *parent = nullptr);
    ~DigitClassifier();

public slots:
    void openDataset();

    void onPredictSelectedSampleRequested(const Sample& sample);

private:

};

