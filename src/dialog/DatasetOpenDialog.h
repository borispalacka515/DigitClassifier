#pragma once

#include "ui_DatasetOpenDialog.h"

#include <QDialog>
#include <QString>

class DatasetOpenDialog : public QDialog
{
	
	Q_OBJECT
public:
	explicit DatasetOpenDialog(QWidget* parent = nullptr);

	QString imagesPath() const;
	QString labelsPath() const;

private:
	Ui::DatasetOpenDialog ui;

	void browseImagesFile();
	void browseLabelsFile();

	void tryFindImagesPath(const QString& labelsPath);
	void tryFindLabelsPath(const QString& imagesPath);

};