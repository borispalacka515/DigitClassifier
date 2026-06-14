#include "SampleViewerWidget.h"

#include <QPainter>

SampleViewerWidget::SampleViewerWidget(QWidget* parent)
	: QWidget(parent)
{
	setMinimumSize(280, 280);
	setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

void SampleViewerWidget::setImage(const QImage& image)
{
	this->image = image;
	update();
}

void SampleViewerWidget::clear()
{
	image = QImage();
	update();
}

bool SampleViewerWidget::isEmpty() const
{
	return image.isNull();
}

void SampleViewerWidget::paintEvent(QPaintEvent* event)
{
	QPainter painter(this);
	painter.fillRect(rect(), Qt::black);

	if (image.isNull())
	{
		return;
	}

	QImage scaledImage = image.scaled(
		size(),
		Qt::KeepAspectRatio,
		Qt::FastTransformation
	);

	QPoint point;
	point.setX((width() - scaledImage.width()) / 2);
	point.setY((height() - scaledImage.height()) / 2);

	painter.drawImage(point, scaledImage);
}
