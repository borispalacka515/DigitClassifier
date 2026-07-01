#include "PredictionBarChartWidget.h"
#include "Model.h"

#include <QPainter>

PredictionBarChartWidget::PredictionBarChartWidget(
	QWidget* parent
) :
	QWidget(parent)
{
	setMinimumHeight(100);
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
    QWidget::paintEvent(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    const QColor widgetBackground(31, 31, 31);
    const QColor plotBackground(24, 24, 24);
    const QColor gridColor(55, 55, 55);
    const QColor axisColor(135, 135, 135);
    const QColor textColor(220, 220, 220);
    const QColor mutedTextColor(165, 165, 165);
    const QColor normalBarColor(86, 156, 214);
    const QColor predictedBarColor(245, 197, 66);

    painter.fillRect(rect(), widgetBackground);

    if (m_probabilities.empty())
    {
        painter.setPen(mutedTextColor);
        painter.drawText(
            rect(),
            Qt::AlignCenter,
            "No prediction yet"
        );
        return;
    }

    const int barCount = static_cast<int>(m_probabilities.size());

    if (barCount == 0)
    {
        return;
    }

    const auto maxIterator = std::max_element(
        m_probabilities.begin(),
        m_probabilities.end()
    );

    const int predictedIndex = static_cast<int>(
        std::distance(m_probabilities.begin(), maxIterator)
        );

    const QRect chartRect = rect().adjusted(
        48,   // left margin for y-axis labels
        18,   // top margin
        -18,  // right margin
        -38   // bottom margin for digit labels
    );

    if (chartRect.width() <= 0 || chartRect.height() <= 0)
    {
        return;
    }

    painter.setPen(Qt::NoPen);
    painter.setBrush(plotBackground);
    painter.drawRoundedRect(chartRect.adjusted(-8, -8, 8, 8), 8, 8);

    QFont labelFont = painter.font();
    labelFont.setPointSize(9);
    painter.setFont(labelFont);

    // Y-axis grid + percent labels
    const std::vector<int> yTicks = { 0, 25, 50, 75, 100 };

    for (int percent : yTicks)
    {
        const double ratio = static_cast<double>(percent) / 100.0;

        const int y = chartRect.bottom()
            - static_cast<int>(ratio * chartRect.height());

        painter.setPen(QPen(gridColor, 1));
        painter.drawLine(
            chartRect.left(),
            y,
            chartRect.right(),
            y
        );

        painter.setPen(mutedTextColor);

        QRect tickLabelRect(
            0,
            y - 10,
            chartRect.left() - 8,
            20
        );

        painter.drawText(
            tickLabelRect,
            Qt::AlignRight | Qt::AlignVCenter,
            QString("%1%").arg(percent)
        );
    }

    // Axes
    painter.setPen(QPen(axisColor, 1));

    painter.drawLine(
        chartRect.left(),
        chartRect.top(),
        chartRect.left(),
        chartRect.bottom()
    );

    painter.drawLine(
        chartRect.left(),
        chartRect.bottom(),
        chartRect.right(),
        chartRect.bottom()
    );

    const int spacing = 10;

    int barWidth = (
        chartRect.width() - spacing * (barCount - 1)
        ) / barCount;

    barWidth = std::max(6, barWidth);
    barWidth = std::min(42, barWidth);

    const int totalBarsWidth =
        barWidth * barCount + spacing * (barCount - 1);

    const int startX =
        chartRect.left() + (chartRect.width() - totalBarsWidth) / 2;

    const double scaleFactor =
        static_cast<double>(chartRect.height());

    // Bars + digit labels
    for (int i = 0; i < barCount; ++i)
    {
        const double probability = std::clamp(
            m_probabilities.at(i),
            0.0,
            1.0
        );

        const int barHeight = static_cast<int>(
            probability * scaleFactor
            );

        const int x = startX + i * (barWidth + spacing);
        const int y = chartRect.bottom() - barHeight;

        QRect barRect(
            x,
            y,
            barWidth,
            barHeight
        );

        const QColor barColor =
            (i == predictedIndex)
            ? predictedBarColor
            : normalBarColor;

        painter.setPen(Qt::NoPen);
        painter.setBrush(barColor);

        if (barHeight > 0)
        {
            painter.drawRoundedRect(barRect, 4, 4);
        }

        QRect digitLabelRect(
            x,
            chartRect.bottom() + 8,
            barWidth,
            20
        );

        painter.setPen(
            (i == predictedIndex)
            ? predictedBarColor
            : textColor
        );

        QFont digitFont = painter.font();
        digitFont.setBold(i == predictedIndex);
        painter.setFont(digitFont);

        painter.drawText(
            digitLabelRect,
            Qt::AlignCenter,
            QString::number(i)
        );
    }
}