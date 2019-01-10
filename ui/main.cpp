#include <QApplication>
#include <QQmlApplicationEngine>
#include "backend.h"
#include "textfielddoublevalidator.cpp"
#include<QFile>
#include <string>
#include <QQmlContext>
#include <QtCore/qdebug.h>
#include "fileio.h"

int main(int argc, char *argv[])
{
    BackEnd b;
    QApplication app(argc, argv);
    QQmlApplicationEngine engine;
#if defined(Q_OS_WIN)
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
#endif
    qmlRegisterType<FileIO, 1>("FileIO", 1, 0, "FileIO");
    qmlRegisterType<TextFieldDoubleValidator>("TextFieldDoubleValidator", 1,0,"TextFieldDoubleValidator");
    engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    engine.rootContext()->setContextProperty("b", &b);
    if (engine.rootObjects().isEmpty())
        return -1;

    return app.exec();
}
