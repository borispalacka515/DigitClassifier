#pragma once

#include "Dataset.h"

#include <QObject>
#include <QString>
#include <memory>

class DatasetLoadWorker : public QObject 
{
	Q_OBJECT

private:
	QString m_imagesPath;
	QString m_labelsPath;

public:
	DatasetLoadWorker(
		const QString& imagesPath,
		const QString& labelsPath,
		QObject* parent = nullptr);

public slots:
	void loadDataset();

signals:
	void progressChanged(int current, int total);
	void errorOccurred(const QString& message);
	void datasetLoaded(std::shared_ptr<Dataset> dataset);
	void finished();

};