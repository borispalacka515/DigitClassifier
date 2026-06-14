#include "SampleThumbnailModel.h"
#include "ImageConverter.h"
#include "MNIST.h"

#include <QPixmap>
#include <QIcon>

SampleThumbnailModel::SampleThumbnailModel(
	QObject* parent
)
	: QAbstractListModel(parent) 
{
}

void SampleThumbnailModel::setDataset(
	std::shared_ptr<Dataset> dataset
)
{
	beginResetModel();

	this->dataset = dataset;

	endResetModel();
}

int SampleThumbnailModel::rowCount(
	const QModelIndex& parent
) const
{
	if (parent.isValid() || !dataset)
	{
		return 0;
	}

	return dataset->size();
}

QVariant SampleThumbnailModel::data(
	const QModelIndex& index, int role
) const
{
	if (!index.isValid() || !dataset)
	{
		return {};
	}

	int sampleIndex = index.row();
	const Sample& sample = dataset->getSample(sampleIndex);

	switch (role)
	{
	case Qt::UserRole:
	{
		return sampleIndex;
	}
	case Qt::DecorationRole:
	{
		const QImage image = ImageConverter::sampleToImage(sample);
		const QImage scaledImage = image.scaled(
			MNIST::ImageWidth, MNIST::ImageHeight,
			Qt::KeepAspectRatio, Qt::FastTransformation
		);

		const QPixmap pixmap = QPixmap::fromImage(scaledImage);

		return QIcon(pixmap);
	}
	case Qt::ToolTipRole:
	{
		const QString sampleInfo = 
			QString("Label: %1\nID: %2").
			arg(sample.label).arg(sampleIndex);

		return sampleInfo;
	}
	default:
		return {};
	}
}
