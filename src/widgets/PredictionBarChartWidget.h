#pragma once

#include <QWidget>

class PredictionBarChartWidget : public QWidget
{
	Q_OBJECT

private:
	std::vector<double> m_probabilities;

protected:
	void paintEvent(QPaintEvent* event) override;

public:
	explicit PredictionBarChartWidget(QWidget* parent = nullptr);

	void setProbabilities(const std::vector<double>& probabilities);
	void clear();
};