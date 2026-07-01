#include "ui_DatasetViewerPage.h"
#include "DatasetViewerPage.h"
#include "ImageConverter.h"
#include "DatasetLoadWorker.h"
#include "Model.h"
#include "PredictionBarChartWidget.h"

#include <QThread>
#include <QMessageBox>
#include <QSpinBox>
#include <QListView>
#include <QPushButton>

const QString t10kImagesPath =
    QStringLiteral("../datasets/t10k-dataset/t10k-images.idx3-ubyte");

const QString t10kLabelsPath =
    QStringLiteral("../datasets/t10k-dataset/t10k-labels.idx1-ubyte");

DatasetViewerPage::DatasetViewerPage(
	QWidget* parent
) 
	: QWidget(parent),
    ui(new Ui::DatasetViewerPage)
{
    ui->setupUi(this);

    ui->progressBar->setVisible(false);
    ui->progressBar->setValue(0);

    m_thumbnailModel = new SampleThumbnailModel(this);
    ui->sampleThumbnailListView->setModel(m_thumbnailModel);

    connect(ui->sampleIndexSpinBox, &QSpinBox::valueChanged,
        this, QOverload<int>::of(&DatasetViewerPage::showSample));

    connect(ui->sampleThumbnailListView, &QListView::clicked,
        this, QOverload<const QModelIndex&>::of(&DatasetViewerPage::showSample));

    connect(ui->predictPushButton, &QPushButton::clicked,
        this, &DatasetViewerPage::onPredictSelectedSampleClicked);



    loadDataset(
        t10kImagesPath,
        t10kLabelsPath
    );
}

DatasetViewerPage::~DatasetViewerPage()
{
}

void DatasetViewerPage::loadDataset(
    const QString& imagesFilePath,
    const QString& labelsFilePath
)
{
    ui->progressBar->setVisible(true);
    ui->progressBar->setValue(0);

    QThread* thread = new QThread(this);

    DatasetLoadWorker* worker = new DatasetLoadWorker(imagesFilePath, labelsFilePath);

    worker->moveToThread(thread);

    connect(thread, &QThread::started,
        worker, &DatasetLoadWorker::loadDataset);

    connect(worker, &DatasetLoadWorker::progressChanged,
        this, [this](int current, int total)
        {
            ui->progressBar->setMaximum(total);
            ui->progressBar->setValue(total);
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
            ui->progressBar->setVisible(false);
            ui->sampleIndexSpinBox->setMaximum(m_dataset->size());
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

void DatasetViewerPage::setDataset(std::shared_ptr<Dataset> dataset)
{
	m_dataset = dataset;
}

void DatasetViewerPage::showPrediction(const std::vector<double>& probabilities)
{
    ui->predictionBarChartWidget->setProbabilities(probabilities);

    const int predictedLabel = Model::predict(probabilities);

    const double confidence = probabilities.at(predictedLabel) * 100;
}

void DatasetViewerPage::clearPrediction()
{
    ui->predictionBarChartWidget->clear();
}

void DatasetViewerPage::showSample(int index)
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

    ui->sampleViewerWidget->setImage(
        ImageConverter::sampleToImage(sample)
    );

    ui->sampleLabel->setText(
        QString("Label: %1").arg(sample.label)
    );

    syncGalleryToSampleIndex(index);
}

void DatasetViewerPage::showSample(const QModelIndex& index)
{
    int sampleIndex = index.row();

    if (ui->sampleIndexSpinBox->value() != sampleIndex)
    {
        QSignalBlocker blocker(ui->sampleIndexSpinBox);
        ui->sampleIndexSpinBox->setValue(sampleIndex);
    }
    showSample(sampleIndex);
}

int DatasetViewerPage::selectedSampleIndex() const
{
    return ui->sampleIndexSpinBox->value();
}

const Sample& DatasetViewerPage::selectedSample() const
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

void DatasetViewerPage::syncGalleryToSampleIndex(int index)
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

    ui->sampleThumbnailListView->selectionModel()->setCurrentIndex(
        modelIndex,
        QItemSelectionModel::ClearAndSelect
    );

    ui->sampleThumbnailListView->scrollTo(
        modelIndex,
        QAbstractItemView::PositionAtCenter
    );
}

void DatasetViewerPage::onPredictSelectedSampleClicked()
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

    try
    {
        const Sample& sample = selectedSample();
        
        emit predictSelectedSampleRequest(sample);
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