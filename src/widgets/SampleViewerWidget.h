#pragma once

#include <QWidget>
#include <QImage>

class SampleViewerWidget : public QWidget
{
	Q_OBJECT

private:
	QImage m_image;

protected:
	void paintEvent(QPaintEvent* event) override;

public:
	explicit SampleViewerWidget(QWidget* parent = nullptr);

	void setImage(const QImage& image);
	void clear();
	bool isEmpty() const;
};