#include "DigitClassifier.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    DigitClassifier window;
    window.show();
    return app.exec();
}
