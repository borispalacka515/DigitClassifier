#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_DigitClassifier.h"
#include "Dataset.h"
#include "SampleThumbnailModel.h"

#include <memory>

class DigitClassifier : public QMainWindow
{
    Q_OBJECT

public:
    DigitClassifier(QWidget *parent = nullptr);
    ~DigitClassifier();

public slots:
    void openDataset();

private:
    Ui::DigitClassifierClass ui;

    std::shared_ptr<Dataset> dataset;

    SampleThumbnailModel* thumbnailModel = nullptr;

    void loadDataset(
        const QString& imagesPath,
        const QString& labelsPath
    );
    
    void showSample(int index);
    void showSample(const QModelIndex& index);

    void syncGalleryToSampleIndex(int index);

};

