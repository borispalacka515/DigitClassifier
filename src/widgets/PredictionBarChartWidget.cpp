#include "PredictionBarChartWidget.h"
#include "Model.h"

#include <QPainter>

PredictionBarChartWidget::PredictionBarChartWidget(
	QWidget* parent
) :
	QWidget(parent)
{
	setMinimumHeight(180);
}

void PredictionBarChartWidget::setProbabilities(
	const std::vector<double>& probabilities
)
{
	m_probabilities = probabilities;
	update();
}

void PredictionBarChartWidget::clear()
{
	m_probabilities.clear();
	update();
}

void PredictionBarChartWidget::paintEvent(QPaintEvent* event)
{
	if (m_probabilities.empty())
	{
		return;
	}

	QWidget::paintEvent(event);

	QPainter painter(this);

	int predictedIndex = Model::predict(m_probabilities);

	QRect chartRect = rect().adjusted(10, 10, -10, -10);

    int spacing = 4;
    int barCount = static_cast<int>(m_probabilities.size());

    int barWidth =
        (chartRect.width() - spacing * (barCount - 1)) / barCount;

    double scaleFactor = static_cast<double>(chartRect.height());

    for (int i = 0; i < barCount; ++i)
    {
        int barHeight = static_cast<int>(
            m_probabilities.at(i) * scaleFactor
            );

        int x = chartRect.left() + i * (barWidth + spacing);
        int y = chartRect.bottom() - barHeight;

        QRect barRect(
            x,
            y,
            barWidth,
            barHeight
        );

        QColor fillColor =
            (i == predictedIndex) ? Qt::yellow : Qt::lightGray;

        painter.fillRect(barRect, fillColor);

        painter.setPen(Qt::black);
        painter.drawText(
            QRect(x, chartRect.bottom() + 2, barWidth, 20),
            Qt::AlignCenter,
            QString::number(i)
        );
    }

}