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
	m_image = image;
	update();
}

void SampleViewerWidget::clear()
{
	m_image = QImage();
	update();
}

bool SampleViewerWidget::isEmpty() const
{
	return m_image.isNull();
}

void SampleViewerWidget::paintEvent(QPaintEvent* event)
{
	QPainter painter(this);
	painter.fillRect(rect(), Qt::black);

	if (m_image.isNull())
	{
		return;
	}

	QImage scaledImage = m_image.scaled(
		size(),
		Qt::KeepAspectRatio,
		Qt::FastTransformation
	);

	QPoint point;
	point.setX((width() - scaledImage.width()) / 2);
	point.setY((height() - scaledImage.height()) / 2);

	painter.drawImage(point, scaledImage);
}
