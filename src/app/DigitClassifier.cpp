#include "DigitClassifier.h"
#include "MNISTLoader.h"
#include "DatasetLoadWorker.h"
#include "ImageConverter.h"
#include "DatasetOpenDialog.h"
#include "ModelSerializer.h"
#include "Trainer.h"

#include <QThread>
#include <QMessageBox>
#include <QSpinBox>
#include <iostream>

const QString t10kImagesPath =
QStringLiteral("../datasets/t10k-dataset/t10k-images.idx3-ubyte");

const QString t10kLabelsPath =
QStringLiteral("../datasets/t10k-dataset/t10k-labels.idx1-ubyte");

DigitClassifier::DigitClassifier(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    
    ui.progressBar->setVisible(false);
    ui.progressBar->setValue(0);

    thumbnailModel = new SampleThumbnailModel(this);
    ui.sampleThumbnailListView->setModel(thumbnailModel);

    loadDataset(
        t10kImagesPath,
        t10kLabelsPath
    );

    m_model = ModelSerializer::loadFromFile("../models/model01.json");

    connect(ui.sampleIndexSpinBox, &QSpinBox::valueChanged,
        this, QOverload<int>::of(&DigitClassifier::showSample));

    connect(ui.sampleThumbnailListView, &QListView::clicked,
        this, QOverload<const QModelIndex&>::of(&DigitClassifier::showSample));

    connect(ui.openDatasetAction, &QAction::triggered,
        this, &DigitClassifier::openDataset);

    connect(ui.predictPushButton, &QPushButton::clicked,
        this, &DigitClassifier::onPredictSelectedSampleClicked);
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

    loadDataset(imagesPath, labelsPath);
}

void DigitClassifier::loadDataset(
    const QString& imagesPath, 
    const QString& labelsPath)
{
    ui.progressBar->setVisible(true);
    ui.progressBar->setValue(0);
     
    QThread* thread = new QThread(this);

    DatasetLoadWorker* worker = new DatasetLoadWorker(imagesPath, labelsPath);

    worker->moveToThread(thread);
    
    connect(thread, &QThread::started,
            worker, &DatasetLoadWorker::loadDataset);

    connect(worker, &DatasetLoadWorker::progressChanged,
        this, [this](int current, int total)
        {
            ui.progressBar->setMaximum(total);
            ui.progressBar->setValue(total);
        });

    connect(worker, &DatasetLoadWorker::datasetLoaded,
        this, [this](std::shared_ptr<Dataset> loadedDataset)
        {
            m_dataset = loadedDataset;
            m_thumbnailModel->setDataset(m_dataset);
        });

    connect(worker, &DatasetLoadWorker::errorOccurred,
        this, [this](const QString& message)
        {
            QMessageBox::critical(
                this,
                "Dataset loading error",
                message
            );
        });

    connect(worker, &DatasetLoadWorker::finished,
        this, [this]()
        {
            ui.progressBar->setVisible(false);
            ui.sampleIndexSpinBox->setMaximum(m_dataset->size());
            showSample(0);
        });

    connect(worker, &DatasetLoadWorker::finished,
        thread, &QThread::quit);

    connect(worker, &DatasetLoadWorker::finished,
        thread, &QThread::deleteLater);

    connect(worker, &DatasetLoadWorker::finished,
        worker, &QObject::deleteLater);

    thread->start();
}

void DigitClassifier::onPredictSelectedSampleClicked()
{
    if (!m_dataset || m_dataset->empty())
    {
        QMessageBox::warning(
            this,
            "Prediction error",
            "Dataset is not loaded."
        );
        return;
    }

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
        const Sample& sample = selectedSample();

        const std::vector<double> output = m_model->forward(sample.input);

        int predictedLabel = m_model->predict(output);

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

void DigitClassifier::showSample(int index)
{
    if (!m_dataset || m_dataset->empty())
    {
        return;
    }

    if (index < 0 || m_dataset->size() <= index)
    {
        return;
    }

    const Sample& sample = m_dataset->getSample(index);

    ui.sampleViewerWidget->setImage(
        ImageConverter::sampleToImage(sample)
    );

    ui.sampleLabel->setText(
        QString("Label: %1").arg(sample.label)
    );

    syncGalleryToSampleIndex(index);
}

void DigitClassifier::showSample(const QModelIndex& index)
{
    int sampleIndex = index.row();

    if (ui.sampleIndexSpinBox->value() != sampleIndex)
    {
        QSignalBlocker blocker(ui.sampleIndexSpinBox);
        ui.sampleIndexSpinBox->setValue(sampleIndex);
    }
    showSample(sampleIndex);
}

int DigitClassifier::selectedSampleIndex() const
{
    return ui.sampleIndexSpinBox->value();
}

const Sample& DigitClassifier::selectedSample() const
{
    if (!m_dataset || m_dataset->empty())
    {
        throw std::runtime_error(
            "Dataset is not loaded."
        );
    }

    const int index = selectedSampleIndex();

    if (index < 0 && index < m_dataset->size())
    {
        throw std::out_of_range(
            "Selected sample index is out of range."
        );
    }

    return m_dataset->getSample(index);
}

void DigitClassifier::syncGalleryToSampleIndex(int index)
{
    if (!m_thumbnailModel)
    {
        return;
    }

    QModelIndex modelIndex = m_thumbnailModel->index(index, 0);

    if (!modelIndex.isValid())
    {
        return;
    }

    ui.sampleThumbnailListView->selectionModel()->setCurrentIndex(
        modelIndex,
        QItemSelectionModel::ClearAndSelect
    );

    ui.sampleThumbnailListView->scrollTo(
        modelIndex,
        QAbstractItemView::PositionAtCenter
    );
}