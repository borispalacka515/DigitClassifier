#include "DatasetLoadWorker.h"
#include "MNISTLoader.h"

DatasetLoadWorker::DatasetLoadWorker(
	const QString& imagesPath,
	const QString& labelsPath,
	QObject* parent
)
	: QObject(parent),
	  imagesPath(imagesPath),
	  labelsPath(labelsPath)
{
}

void DatasetLoadWorker::loadDataset()
{
	try 
	{
		auto progressCallback = [this](int current, int total)
			{
				emit progressChanged(current, total);
			};

		auto dataset = std::make_shared<Dataset>(
				MNISTLoader::load(
					imagesPath.toStdString(),
					labelsPath.toStdString(),
					progressCallback
				));

		emit datasetLoaded(dataset);
	}
	catch(const std::exception& e)
	{
		emit errorOccurred(e.what());
	}

	emit finished();
}

