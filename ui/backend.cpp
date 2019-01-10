#include "backend.h"
#include <bits/stdc++.h>
#include <iostream>
#include <QDebug>
using namespace std;
BackEnd::BackEnd(QObject *parent) :
    QObject(parent)
{
}

void BackEnd::onbuttonClicked(const QString &command)
{
    byte_array_command = command;
    QByteArray ba = byte_array_command.toLatin1();
    const char *c_str_command = ba.data();
    qDebug()<<c_str_command;
    system(c_str_command);
}

