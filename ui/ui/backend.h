#ifndef BACKEND_H
#define BACKEND_H

#include <QObject>
#include <QString>

class BackEnd : public QObject
{
    Q_OBJECT

public:
    explicit BackEnd(QObject *parent = nullptr);

    QString userName();

public slots:
    void onbuttonClicked(const QString &command);

private:
    QString byte_array_command;
};

#endif // BACKEND_H
