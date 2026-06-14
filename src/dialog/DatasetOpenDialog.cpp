#include "DatasetOpenDialog.h"

#include <QFileDialog>
#include <QFileInfo>
#include <QDir>

DatasetOpenDialog::DatasetOpenDialog(QWidget* parent)
	: QDialog(parent)
{
	ui.setupUi(this);

	connect(ui.browseImagesButton, &QPushButton::clicked,
		this, &DatasetOpenDialog::browseImagesFile);

	connect(ui.browseLabelsButton, &QPushButton::clicked,
		this, &DatasetOpenDialog::browseLabelsFile);

	connect(ui.buttonBox, &QDialogButtonBox::accepted,
		this, &QDialog::accept);

	connect(ui.buttonBox, &QDialogButtonBox::rejected,
		this, &QDialog::reject);
}

QString DatasetOpenDialog::imagesPath() const
{
	return ui.imagesPathLineEdit->text();
}

QString DatasetOpenDialog::labelsPath() const
{
	return ui.labelsPathLineEdit->text();
}

void DatasetOpenDialog::browseImagesFile()
{
	QString imagesPath = QFileDialog::getOpenFileName(
		this,
		tr("Open Images File"),
		"../data",
		tr("MNIST Images (*.ubyte *idx3-ubyte);;All Files (*)")
	);

	if (!imagesPath.isEmpty())
	{
		tryFindLabelsPath(imagesPath);
		ui.imagesPathLineEdit->setText(imagesPath);
	}
}

void DatasetOpenDialog::browseLabelsFile()
{
	QString labelsPath = QFileDialog::getOpenFileName(
		this,
		tr("Open Labels File"),
		"../data",
		tr("MNIST Labels (*.ubyte *idx1-ubyte);;All Files (*)")
	);

	if (!labelsPath.isEmpty())
	{
		tryFindImagesPath(labelsPath);
		ui.labelsPathLineEdit->setText(labelsPath);
	}
}

void DatasetOpenDialog::tryFindImagesPath(
	const QString& labelsPath)
{
	QFileInfo fileInfo(labelsPath);
	QString fileName = fileInfo.fileName();
	QString folderPath = fileInfo.absolutePath();
	QDir directory(folderPath);

	if (!fileName.contains("labels") ||
		!fileName.contains("idx1"))
	{
		return;
	}

	fileName.replace("labels", "images");
	fileName.replace("idx1", "idx3");

	if (directory.exists(fileName)) 
	{
		ui.imagesPathLineEdit->setText(
			directory.filePath(fileName)
		);
	}
}

void DatasetOpenDialog::tryFindLabelsPath(
	const QString& imagesPath)
{
	QFileInfo fileInfo(imagesPath);
	QString fileName = fileInfo.fileName();
	QString folderPath = fileInfo.absolutePath();
	QDir directory(folderPath);

	if (!fileName.contains("images") ||
		!fileName.contains("idx3"))
	{
		return;
	}

	fileName.replace("images", "labels");
	fileName.replace("idx3", "idx1");

	if (directory.exists(fileName))
	{
		ui.labelsPathLineEdit->setText(
			directory.filePath(fileName)
		);
	}
}