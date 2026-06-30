#pragma once

#include "ui_DatasetViewerPage.h"

#include "Dataset.h"
#include "SampleThumbnailModel.h"

#include <QWidget>
#include <memory>


class DatasetViewerPage : public QWidget
{
	Q_OBJECT

private:
	Ui::DatasetViewerPage* ui;

	std::shared_ptr<Dataset> m_dataset;
	SampleThumbnailModel* m_thumbnailModel;

public:
	explicit DatasetViewerPage(QWidget* parent = nullptr);
	~DatasetViewerPage();

	void setDataset(std::shared_ptr<Dataset> dataset);
	void loadDataset(
		const QString& imagesFilePath,
		const QString& labelsFilePath
	);

private slots:
	void showSample(int index);
	void showSample(const QModelIndex& index);
	void onPredictSelectedSampleClicked();

signals:
	void predictSelectedSampleRequest(const Sample& sample);

private:
	int selectedSampleIndex() const;
	const Sample& selectedSample() const;

	void syncGalleryToSampleIndex(int index);
};