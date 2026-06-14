#pragma once

#include <QWidget>
#include <QImage>

class SampleViewerWidget : public QWidget
{
	Q_OBJECT

private:
	QImage image;

public:
	explicit SampleViewerWidget(QWidget* parent = nullptr);

	void setImage(const QImage& image);
	void clear();
	bool isEmpty() const;

private:
	void paintEvent(QPaintEvent* event);
};