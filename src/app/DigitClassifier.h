#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_DigitClassifier.h"
#include "Dataset.h"
#include "SampleThumbnailModel.h"
#include "Model.h"

#include <memory>
#include <optional>

class DigitClassifier : public QMainWindow
{
    Q_OBJECT

private:
    Ui::DigitClassifierClass ui;

    std::shared_ptr<Dataset> m_dataset;

    std::optional<Model> m_model;

    SampleThumbnailModel* m_thumbnailModel = nullptr;

public:
    DigitClassifier(QWidget *parent = nullptr);
    ~DigitClassifier();

public slots:
    void openDataset();

    void onPredictSelectedSampleClicked();

private:

    void loadDataset(
        const QString& imagesPath,
        const QString& labelsPath
    );
    
    void showSample(int index);
    void showSample(const QModelIndex& index);

    int selectedSampleIndex() const;
    const Sample& selectedSample() const;

    void syncGalleryToSampleIndex(int index);

};

