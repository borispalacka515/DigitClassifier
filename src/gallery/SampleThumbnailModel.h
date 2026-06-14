#pragma once

#include "Dataset.h"

#include <QAbstractListModel>

class SampleThumbnailModel : public QAbstractListModel
{
	Q_OBJECT

public:
	explicit SampleThumbnailModel(QObject* parent = nullptr);

	void setDataset(std::shared_ptr<Dataset> dataset);

	int rowCount(const QModelIndex& parent = QModelIndex()) const override;
	QVariant data(const QModelIndex& index, int role) const override;
private:
	std::shared_ptr<Dataset> dataset;

};