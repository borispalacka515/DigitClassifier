#include "DigitClassifier.h"
#include "MNISTLoader.h"
#include "DatasetLoadWorker.h"
#include "ImageConverter.h"
#include "DatasetOpenDialog.h"

#include <QThread>
#include <QMessageBox>
#include <QSpinBox>


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

    connect(ui.sampleIndexSpinBox, &QSpinBox::valueChanged,
        this, QOverload<int>::of(&DigitClassifier::showSample));

    connect(ui.sampleThumbnailListView, &QListView::clicked,
        this, QOverload<const QModelIndex&>::of(&DigitClassifier::showSample));

    connect(ui.openDatasetAction, &QAction::triggered,
        this, &DigitClassifier::openDataset);
}

DigitClassifier::~DigitClassifier()
{}

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
            dataset = loadedDataset;
            thumbnailModel->setDataset(dataset);
        });

    connect(worker, &DatasetLoadWorker::errorOccurred,
        this, [this](const QString& message)
        {
            QMessageBox::critical(this, "Dataset loading error", message);
        });

    connect(worker, &DatasetLoadWorker::finished,
        this, [this]()
        {
            ui.progressBar->setVisible(false);
            ui.sampleIndexSpinBox->setMaximum(dataset->size());
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

void DigitClassifier::showSample(int index)
{
    if (!dataset || dataset->empty())
    {
        return;
    }

    if (index < 0 || dataset->size() <= index)
    {
        return;
    }

    const Sample& sample = dataset->getSample(index);

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

void DigitClassifier::syncGalleryToSampleIndex(int index)
{
    if (!thumbnailModel)
    {
        return;
    }

    QModelIndex modelIndex = thumbnailModel->index(index, 0);

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