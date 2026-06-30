#include "DigitClassifier.h"
#include "MNISTLoader.h"
#include "DatasetLoadWorker.h"
#include "ImageConverter.h"
#include "DatasetOpenDialog.h"
#include "ModelSerializer.h"
#include "Trainer.h"
#include "DatasetViewerPage.h"

#include <QThread>
#include <QMessageBox>
#include <QSpinBox>
#include <iostream>

DigitClassifier::DigitClassifier(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    /*loadDataset(
        t10kImagesPath,
        t10kLabelsPath
    );*/

    m_model = ModelSerializer::loadFromFile("../models/model01.json");

    connect(ui.openDatasetAction, &QAction::triggered,
        this, &DigitClassifier::openDataset);

    connect(ui.datasetViewerPage, &DatasetViewerPage::predictSelectedSampleRequest,
        this, &DigitClassifier::onPredictSelectedSampleRequested);
}

DigitClassifier::~DigitClassifier()
{}


void trainingTest(Dataset trainingDataset)
{
    try
    {
        // 784 vstupov = 28x28 MNIST obrázok
        // 128 hidden neurónov
        // 10 výstupov = číslice 0 až 9
        ModelConfig modelConfig(
            { 784, 128, 10 },
            { ActivationType::ReLU, ActivationType::Softmax }
        );

        Model model(modelConfig);

        TrainingConfig trainingConfig(
            256,                              // batch size
            2,                               // epoch count
            0.5,                            // learning rate
            LossFunctionType::CrossEntropy
        );


        Trainer trainer;
        trainer.train(model, trainingConfig, trainingDataset);

        std::cout << "Training finished successfully.\n";

        ModelSerializer::saveToFile(model, "../models/model01.json");
    }
    catch (const std::exception& exception)
    {
        std::cerr << "Training failed: "
            << exception.what()
            << '\n';
    }
}

void DigitClassifier::openDataset()
{
    DatasetOpenDialog dialog(this);

    if (dialog.exec() != QDialog::Accepted)
    {
        return;
    }

    const QString imagesPath = dialog.imagesPath();
    const QString labelsPath = dialog.labelsPath();

    if (imagesPath.isEmpty() || labelsPath.isEmpty())
    {
        return;
    }

    //loadDataset(imagesPath, labelsPath);
}

void DigitClassifier::onPredictSelectedSampleRequested(const Sample& sample)
{
    if (!m_model.has_value())
    {
        QMessageBox::warning(
            this,
            "Prediction error",
            "Model is not loaded."
        );
        return;
    }

    try
    {

        const std::vector<double> output = m_model->forward(sample.input);

        int predictedLabel = m_model->predict(output);

        ui.datasetViewerPage->showPrediction(output);

        qDebug() << "Predicted label is" << predictedLabel;

    }
    catch (const std::exception& exception)
    {
        QMessageBox::critical(
            this,
            "Prediction error",
            exception.what()
        );
    }
}

