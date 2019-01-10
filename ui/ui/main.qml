import QtQuick 2.9
import QtQuick.Window 2.3
import QtQuick.Controls 2.0
import QtCharts 2.2
import TextFieldDoubleValidator 1.0
import FileIO 1.0

//Main Window
Window {
    id: root
    visibility: "Maximized"
    visible: true
    minimumWidth: 1800
    minimumHeight: 1000

    Rectangle{
        id:main_rectangle
        anchors.fill: parent
        color: "#EFEFEF"

        //Pages
        TabBar {
            id: tabBar
            anchors.top:text1.bottom
            anchors.topMargin: parent.height/20
            anchors.left:parent.left
            anchors.leftMargin: parent.width/40
            height: parent.height/20
            width: parent.width/3
            background: Rectangle {
                color: "transparent"
            }

            //Parameters Tab
            TabButton {
                id:tabbutton1
                implicitWidth: parent.width/2
                implicitHeight: parent.height
                background: Rectangle{
                    color: "white"
                }
                Text{
                    width:parent.width
                    height:parent.height
                    font.pixelSize: 25
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    text: qsTr("PARAMETERS")
                    fontSizeMode: Text.Fit
                    color:"#aeadad"
                    Rectangle{
                        id:tabbutton1_rectangle
                        anchors.bottom: parent.bottom
                        anchors.bottomMargin: 0
                        width:parent.width
                        height:3
                        color:"#aeadad"
                    }
                }
                MouseArea{
                    anchors.fill: parent
                    onClicked: tabBar.state="tabbutton1_clicked"
                }
            }
            //Parameters Tab End

            //Benchmarking Results Tab
            TabButton {
                id:tabbutton2
                anchors.left: tabbutton1.right
                anchors.leftMargin: parent.width/40
                implicitWidth: parent.width/3
                implicitHeight: parent.height
                background: Rectangle{
                    color: "white"
                }
                Text{
                    width:parent.width
                    height:parent.height
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    font.pixelSize: 25
                    text: qsTr("BENCHMARKING")
                    fontSizeMode: Text.Fit
                    color:"#aeadad"
                    Rectangle{
                        id:tabbutton2_rectangle
                        anchors.bottom: parent.bottom
                        anchors.bottomMargin: 0
                        width:parent.width
                        height:3
                        color:"white"
                    }
                }
                MouseArea{
                    id: tabbutton2_mousearea
                    anchors.fill: parent
                    onClicked: tabBar.state="tabbutton2_clicked"
                }
            }
            TabButton {
                id:tabbutton3
                anchors.left: tabbutton2.right
                anchors.leftMargin: parent.width/40
                implicitWidth: parent.width/4
                implicitHeight: parent.height
                background: Rectangle{
                    color: "white"
                }
                Text{
                    width:parent.width
                    height:parent.height
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    font.pixelSize: 25
                    text: qsTr("IRIS RESULTS")
                    fontSizeMode: Text.Fit
                    color:"#aeadad"
                    Rectangle{
                        id:tabbutton3_rectangle
                        anchors.bottom: parent.bottom
                        anchors.bottomMargin: 0
                        width:parent.width
                        height:3
                        color:"white"
                    }
                }
                MouseArea{
                    id: tabbutton3_mousearea
                    anchors.fill: parent
                    onClicked: tabBar.state="tabbutton3_clicked"
                }
            }
            //Benchmarking Results Tab End

            //Code for Page Transition
            states:[
                State{
                    name:"tabbutton1_clicked"
                    PropertyChanges {
                        target: tabbutton1_rectangle
                        color:"#aeadad"
                    }
                    PropertyChanges {
                        target: tabbutton2_rectangle
                        color:"white"
                    }
                    PropertyChanges {
                        target: parameters_rectangle
                        visible:true
                    }
                    PropertyChanges {
                        target: graph_rectangle
                        visible:false
                    }
                    PropertyChanges {
                        target: other_parameters
                        visible:true
                    }
                    PropertyChanges {
                        target: velocity_parameters_rectangle
                        visible:true
                    }
                    PropertyChanges {
                        target: graphics
                        visible:true
                    }
                    PropertyChanges {
                        target: iris_rectangle
                        visible:false
                    }
                },
                State{
                    name:"tabbutton2_clicked"
                    PropertyChanges {
                        target: tabbutton2_rectangle
                        color:"#aeadad"
                    }
                    PropertyChanges {
                        target: tabbutton1_rectangle
                        color:"white"
                    }
                    PropertyChanges {
                        target: parameters_rectangle
                        visible:false
                    }
                    PropertyChanges {
                        target: other_parameters
                        visible:false
                    }
                    PropertyChanges {
                        target: velocity_parameters_rectangle
                        visible:false
                    }
                    PropertyChanges {
                        target: graphics
                        visible:false
                    }
                    PropertyChanges {
                        target: graph_rectangle
                        visible:true
                    }
                    PropertyChanges {
                        target: other_parameters
                        visible:false
                    }
                    PropertyChanges {
                        target: velocity_parameters_rectangle
                        visible:false
                    }
                    PropertyChanges {
                        target: runButton
                        visible:false
                    }
                    PropertyChanges {
                        target: iris_rectangle
                        visible:false
                    }
                },State{
                    name:"tabbutton3_clicked"
                    PropertyChanges {
                        target: tabbutton3_rectangle
                        color:"#aeadad"
                    }
                    PropertyChanges {
                        target: tabbutton1_rectangle
                        color:"white"
                    }
                    PropertyChanges {
                        target: tabbutton2_rectangle
                        color:"white"
                    }
                    PropertyChanges {
                        target: parameters_rectangle
                        visible:false
                    }
                    PropertyChanges {
                        target: other_parameters
                        visible:false
                    }
                    PropertyChanges {
                        target: velocity_parameters_rectangle
                        visible:false
                    }
                    PropertyChanges {
                        target: graphics
                        visible:false
                    }
                    PropertyChanges {
                        target: graph_rectangle
                        visible:false
                    }
                    PropertyChanges {
                        target: other_parameters
                        visible:false
                    }
                    PropertyChanges {
                        target: velocity_parameters_rectangle
                        visible:false
                    }
                    PropertyChanges {
                        target: runButton
                        visible:false
                    }
                    PropertyChanges {
                        target: iris_rectangle
                        visible:true
                    }
                }
            ]
            //Code for Page Transition End
        }
        //Pages End

        Text {
            id: text1
            horizontalAlignment: Text.AlignHCenter
            width: parent.width
            height: 50
            font.family: "Roboto"
            font.pixelSize: 40
            font.bold: true
            fontSizeMode: Text.Fit
            text: qsTr("Neural Network Particle Swarm Optimization Simulator")
        }

        //Parameters Page
        //PSO Parameters Section
        Rectangle{
            id:parameters_rectangle
            width:parent.width/3
            height: parent.height/2.6
            anchors.left:parent.left
            anchors.leftMargin: parent.width/40
            anchors.top:tabBar.bottom
            anchors.topMargin: parent.height/30
            border.color: "#C6C6C6"
            Rectangle{
                anchors.left:parent.left
                anchors.leftMargin: parent.width/2-parameter_text.width/2
                anchors.right:parent.right
                anchors.rightMargin: parent.width/2-parameter_text.width/2-10
                height:parent.height/10
                anchors.top:parent.top
                anchors.topMargin: -17
                color: "#EFEFEF"
                Text {
                    id: parameter_text
                    anchors.leftMargin: 5
                    anchors.left:parent.left
                    horizontalAlignment: Text.AlignHCenter
                    text: qsTr("PSO PARAMETERS")
                    font.family: "Roboto"
                    font.pixelSize: 23
                }
            }
            color:"transparent"
            //N_IN
            Text {
                id: input_text
                width:parent.width/2.5
                height:parent.height/10
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.top:parent.top
                anchors.topMargin: 40
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                text: qsTr("N_IN:")
                font.family: "Roboto"
                font.pixelSize: 23
                fontSizeMode: Text.Fit
            }
            TextField {
                id:input_textfield
                width:parent.width/2.5
                height: parent.height/10
                anchors.left: input_text.right
                anchors.leftMargin: 10
                anchors.top:parent.top
                anchors.topMargin: 40
                text: qsTr("4")
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                validator: IntValidator{
                    bottom:1
                }
                ToolTip {
                    text: "Number of XOR inputs range 1 to infinity"
                    y:30
                    visible: parent.hovered
                }
                placeholderText: qsTr("Number of XOR inputs")
                font.pixelSize: 20
            }
            //N_IN End

            //N_BATCHSIZE
            Text {
                id: batchsize_text
                width:parent.width/2.5
                height:parent.height/10
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.top:input_text.bottom
                anchors.topMargin: 10
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                text: qsTr("N_BATCHSIZE:")
                fontSizeMode: Text.Fit
                font.family: "Roboto"
                font.pixelSize: 23
            }
            TextField {
                id:batchsize_textfield
                hoverEnabled: true
                width:parent.width/2.5
                height: parent.height/10
                ToolTip {
                    text: "Batchsize range 1 to 2^N_IN"
                    y:30
                    visible: parent.hovered
                }
                anchors.left: batchsize_text.right
                anchors.leftMargin: 10
                anchors.top:input_text.bottom
                anchors.topMargin: 10
                text: qsTr("16")
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                placeholderText: qsTr("Batchsize")
                font.pixelSize: 20
                validator: IntValidator{
                    bottom:1
                    top:Math.pow(2,input_textfield.text)
                }
            }
            //N_BATCHSIZE End

            //N_PARTICLES
            Text {
                id: particles_text
                width:parent.width/2.5
                height:parent.height/10
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.top:batchsize_text.bottom
                anchors.topMargin: 10
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                text: qsTr("N_PARTICLES:")
                font.family: "Roboto"
                font.pixelSize: 23
                fontSizeMode: Text.Fit
            }
            TextField {
                id:particles_textfield
                width:parent.width/2.5
                height: parent.height/10
                anchors.left:particles_text.right
                anchors.leftMargin: 10
                anchors.top:batchsize_text.bottom
                anchors.topMargin: 10
                text: qsTr("32")
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                placeholderText: qsTr("N_Particles")
                font.pixelSize: 20
                validator: IntValidator{
                    bottom:2
                }
                ToolTip {
                    text: "Number of particles range 2 to infinity"
                    y:30
                    visible: parent.hovered
                }
            }
            //N_PARTICLES End

            //N_ITERATIONS
            Text {
                id: iterations_text
                width:parent.width/2.5
                height:parent.height/10
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.top:particles_text.bottom
                anchors.topMargin: 10
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                text: qsTr("N_ITERATIONS")
                font.family: "Roboto"
                font.pixelSize: 23
                fontSizeMode: Text.Fit
            }
            TextField {
                id:iterations_textfield
                width:parent.width/2.5
                height: parent.height/10
                anchors.left:pbest_text.right
                anchors.leftMargin: 10
                anchors.top:particles_text.bottom
                anchors.topMargin: 10
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                placeholderText: qsTr("N_Iterations")
                text: qsTr("200")
                font.pixelSize: 20
                validator: IntValidator{
                    bottom:1
                }
                ToolTip {
                    text: "Number of iterations range 1 to infinity"
                    y:30
                    visible: parent.hovered
                }
            }
            //N_ITERATIONS End

            //PBEST
            Text {
                id: pbest_text
                width:parent.width/2.5
                height:parent.height/10
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.top:iterations_text.bottom
                anchors.topMargin: 10
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                text: qsTr("PBEST FACTOR:")
                font.family: "Roboto"
                font.pixelSize: 23
                fontSizeMode: Text.Fit
            }
            TextField {
                id:pbest_textfield
                width:parent.width/2.5
                height: parent.height/10
                anchors.left:pbest_text.right
                anchors.leftMargin: 10
                anchors.top:iterations_text.bottom
                anchors.topMargin: 10
                text: qsTr("0.6")
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                placeholderText: qsTr("Pbest Factor")
                font.pixelSize: 20
                validator: TextFieldDoubleValidator{
                    bottom:0.0
                    top: 1.0
                    notation: DoubleValidator.StandardNotation
                }
                ToolTip {
                    text: "Personal best for PSO range 0 to 1"
                    y:30
                    visible: parent.hovered
                }
            }
            //PBEST End

            //PSO Algorithm
            Text {
                id: pso_algorithm
                width:parent.width/2.5
                height:parent.height/10
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.top:pbest_textfield.bottom
                anchors.topMargin: 10
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                text: qsTr("PSO ALGORITHM:")
                font.family: "Roboto"
                font.pixelSize: 23
                fontSizeMode: Text.Fit
            }
            Text {
                id:pso_algorithm_value
                text:lbest_checked.checked?(" --lbpso "):" "
                visible: false
            }
            RadioButton{
                id:lbest_checked
                hoverEnabled: true
                anchors.left: pso_algorithm.right
                anchors.leftMargin: 20
                anchors.top:  pbest_textfield.bottom
                anchors.topMargin: 10
                text: "Lbest"
                font.pixelSize: 20
                ToolTip {
                    text: "Use lbest version"
                    y:30
                    visible: parent.hovered
                }
            }
            RadioButton{
                id:gbest_checked
                anchors.left: lbest_checked.right
                anchors.leftMargin: 20
                anchors.top:  pbest_textfield.bottom
                anchors.topMargin: 10
                text: "Gbest"
                checked: true
                font.pixelSize: 20
                ToolTip {
                    text: "Use gbest version"
                    y:30
                    visible: parent.hovered
                }
            }
            //PSO Algorithm End
            //GBEST
            Text {
                id: gbest_text
                width:parent.width/2.5
                height:parent.height/10
                visible: gbest_checked.checked == true? true:false
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.top:pso_algorithm.bottom
                anchors.topMargin: 10
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                text: qsTr("GBEST FACTOR:")
                font.family: "Roboto"
                font.pixelSize: 23
                fontSizeMode: Text.Fit
            }
            TextField {
                id:gbest_textfield
                width:parent.width/2.5
                height: parent.height/10
                visible: gbest_checked.checked == true? true:false
                anchors.left:gbest_text.right
                anchors.leftMargin: 10
                anchors.top:pso_algorithm.bottom
                anchors.topMargin: 10
                text: qsTr("0.8")
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                placeholderText: qsTr("Gbest Factor")
                font.pixelSize: 20
                validator: TextFieldDoubleValidator{
                    bottom:0.0
                    top:1.0
                    notation: DoubleValidator.StandardNotation
                }
                ToolTip {
                    text: "Global best for PSO range 0 to 1"
                    y:30
                    visible: parent.hovered
                }
            }
            Text {
                id:gbest_value
                text:gbest_checked.checked?(" --gbest "+gbest_textfield.text):" "
                visible: false
            }
            //GBEST End

            //LBEST
            Text {
                id: lbest_text
                width:parent.width/2.5
                visible: lbest_checked.checked == true? true:false
                height:parent.height/10
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.top:pso_algorithm.bottom
                anchors.topMargin: 10
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                text: qsTr("LBEST:")
                font.family: "Roboto"
                font.pixelSize: 23
                fontSizeMode: Text.Fit
            }
            TextField {
                id:lbest_textfield
                width:parent.width/2.5
                visible: lbest_checked.checked == true? true:false
                height: parent.height/10
                anchors.left:lbest_text.right
                anchors.leftMargin: 10
                anchors.top:pso_algorithm.bottom
                anchors.topMargin: 10
                text: qsTr("0.8")
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                placeholderText: qsTr("Lbest factor")
                font.pixelSize: 20
                validator: TextFieldDoubleValidator{
                    bottom:0.0
                    top: 1.0
                    notation: DoubleValidator.StandardNotation
                }
                ToolTip {
                    text: "Local best for PSO range 0 to 1"
                    y:30
                    visible: parent.hovered
                }
            }
            Text {
                id:lbest_value
                text:lbest_checked.checked?(" --lbest "+lbest_textfield.text):" "
                visible: false
            }
        }
        //LBEST End
        //PSO Parameters Section End

        //Velocity Parameters section
        Rectangle{
            id:velocity_parameters_rectangle
            width:parent.width/3
            anchors.left:parent.left
            anchors.leftMargin: parent.width/40
            anchors.top:parameters_rectangle.bottom
            anchors.topMargin: parent.height/20
            anchors.bottom: parent.bottom
            anchors.bottomMargin: parent.height/12
            border.color: "#C6C6C6"
            color:"transparent"
            Rectangle{
                anchors.left:parent.left
                anchors.leftMargin: parent.width/2-velocity_parameters_text.width/2
                anchors.right:parent.right
                anchors.rightMargin: parent.width/2-velocity_parameters_text.width/2-10
                height:parent.height/10
                anchors.top:parent.top
                anchors.topMargin: -17
                color: "#EFEFEF"
                Text {
                    id: velocity_parameters_text
                    anchors.left: parent.left
                    anchors.leftMargin: 2
                    horizontalAlignment: Text.AlignHCenter
                    text: qsTr(" VELOCITY PARAMETERS")
                    font.family: "Roboto"
                    font.pixelSize: 25
                    fontSizeMode: Text.Fit
                }
            }
            //Velocity Decay
            Text {
                id: velocity_decay_text
                width:parent.width/2.5
                height:parent.height/10
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.top:velocity_parameters_rectangle.top
                anchors.topMargin: 50
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                text: qsTr("VELOCITY DECAY:")
                font.family: "Roboto"
                font.pixelSize: 23
                fontSizeMode: Text.Fit
            }
            TextField {
                id:velocity_decay_textfield
                width:parent.width/2.5
                height: parent.height/10
                anchors.left:velocity_decay_text.right
                anchors.leftMargin: 10
                anchors.top:velocity_parameters_rectangle.top
                anchors.topMargin: 50
                text: qsTr("1")
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                placeholderText: qsTr("Velocity Decay")
                font.pixelSize: 20
                validator: TextFieldDoubleValidator{
                    bottom:0
                    top:1
                    decimals: 10;
                    notation: DoubleValidator.StandardNotation
                }
                ToolTip {
                    text: "Decay in velocity after each position update range 0 to 1"
                    y:30
                    visible: parent.hovered
                }
            }
            //Velocity Decay End

            //Max Velocity Decay
            Text {
                id: max_velocity_decay_text
                width:parent.width/2.5
                height:parent.height/10
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.top:velocity_decay_text.bottom
                anchors.topMargin: 30
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                text: qsTr("MAX VELOCITY DECAY:")
                font.family: "Roboto"
                font.pixelSize: 23
                fontSizeMode: Text.Fit
            }
            TextField {
                id:max_velocity_decay_textfield
                width:parent.width/2.5
                height: parent.height/10
                anchors.left:max_velocity_decay_text.right
                anchors.leftMargin: 10
                anchors.top:velocity_decay_text.bottom
                anchors.topMargin: 30
                text: qsTr("0.005")
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                // text:backend.userName
                placeholderText: qsTr("Max Velocity Decay")
                font.pixelSize: 20
                validator: TextFieldDoubleValidator{
                    bottom:0.0
                    top:1.0
                    notation: DoubleValidator.StandardNotation
                }
                ToolTip {
                    text: "Multiplier for max velocity with each update range 0 to 1"
                    y:30
                    visible: parent.hovered
                }
            }
            //Max Velocity Decay End

            //Velocity Restrict
            Text {
                id: velocity_restrict_text
                width:parent.width/2.5
                height:parent.height/10
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.top:max_velocity_decay_text.bottom
                anchors.topMargin: 30
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                text: qsTr("VELOCITY RESTRICT:")
                font.family: "Roboto"
                font.pixelSize: 23
                fontSizeMode: Text.Fit
            }
            Text {
                id:velocity_restrict_value
                text:velocity_restrict_true_checked.checked?(" --vr --mv "+max_velocity_textfield.text):" "
                visible: false
            }
            RadioButton{
                id:velocity_restrict_true_checked
                hoverEnabled: true
                checked: true
                anchors.left: velocity_restrict_text.right
                anchors.leftMargin: 20
                anchors.top:  max_velocity_decay_text.bottom
                anchors.topMargin: 30
                text: "True"
                font.pixelSize: 20
                ToolTip {
                    text: "Restrict the particle velocity"
                    y:30
                    visible: parent.hovered
                }
            }
            RadioButton{
                id:velocity_restrict_false_checked
                anchors.left: velocity_restrict_true_checked.right
                anchors.leftMargin: 20
                anchors.top:  max_velocity_decay_text.bottom
                anchors.topMargin: 30
                text: "False"
                font.pixelSize: 20
            }
            //Velocity Restrict End

            //Max Velocity
            Text {
                id: max_velocity_text
                width:parent.width/2.5
                visible: velocity_restrict_true_checked.checked ? true:false
                height:parent.height/10
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.top:velocity_restrict_text.bottom
                anchors.topMargin: 30
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                text: qsTr("MAX_VELOCITY:")
                font.family: "Roboto"
                font.pixelSize: 23
                fontSizeMode: Text.Fit
            }
            TextField {
                id:max_velocity_textfield
                width:parent.width/2.5
                visible: velocity_restrict_true_checked.checked ? true:false
                height: parent.height/10
                anchors.left:max_velocity_text.right
                anchors.leftMargin: 10
                anchors.top:velocity_restrict_text.bottom
                anchors.topMargin: 30
                text: qsTr("1")
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                placeholderText: qsTr("Max Velocity")
                font.pixelSize: 20
                validator: TextFieldDoubleValidator{
                    bottom:0.0
                    notation: DoubleValidator.StandardNotation
                }
                ToolTip {
                    text: "Maximum velocity for a particle range 0 to infinity"
                    y:30
                    visible: parent.hovered
                }
            }
            //Max Velocity End
        }
        //Velocity Parameters Section End

        //Run Button
        Button{
            id:runButton
            width:parent.width/20
            height:40
            anchors.bottom: parent.bottom
            anchors.bottomMargin: parent.height/30
            anchors.horizontalCenter: parameters_rectangle.horizontalCenter
            hoverEnabled: true
            background:Rectangle{
                id:background
                height: parent.height
                radius: parent.width
                border.color: "#aeadad"
            }
            Text{
                id:runButton_text
                text: qsTr("RUN");
                font.pixelSize: 25
                anchors.centerIn: parent
                color: "#aeadad"
            }
            MouseArea{
                anchors.fill: parent
                onEntered: background.color = "#afadad", runButton_text.color="#000000"
                onExited: background.color = "#ffffff", runButton_text.color="#aeadad"
                onClicked: b.onbuttonClicked("python3 clinn.py --bs "+batchsize_textfield.text+" --xorn "+input_textfield.text+" --pno "+particles_textfield.text+pso_algorithm_value.text+gbest_value.text+lbest_value.text+" --pbest "+pbest_textfield.text+" --veldec "+velocity_decay_textfield.text+velocity_restrict_value.text+" --mvdec "+max_velocity_decay_textfield.text+hybrid_value.text+" --iter "+iterations_textfield.text+" --hl "+hidden_layers_textfield.text)
            }
        }
        //Run Button End

        //Graphics Section
        Rectangle{
            id:graphics
            visible: true
            width:parent.width/2
            height:parent.height/2.5
            anchors.left: parameters_rectangle.right
            anchors.leftMargin: parent.width/40
            anchors.right:parent.right
            anchors.rightMargin: parent.width/40
            anchors.top:tabBar.bottom
            anchors.topMargin: parent.height/35
            border.color: "#C6C6C6"
            color:"transparent"
            Rectangle{
                anchors.left:parent.left
                anchors.leftMargin: parent.width/2-graph_text.width/2
                anchors.right:parent.right
                anchors.rightMargin: parent.width/2-graph_text.width/2-10
                height:parent.height/10
                anchors.top:parent.top
                anchors.topMargin: -17
                color: "#EFEFEF"
                Text{
                    id: graph_text
                    anchors.left:parent.left
                    anchors.leftMargin: 5
                    horizontalAlignment: Text.AlignHCenter
                    text: qsTr("GRAPHICS")
                    font.family: "Roboto"
                    font.pixelSize: 25
                    fontSizeMode: Text.Fit
                }
            }
            AnimatedImage{
                id: animation;
                width:parent.width/2;
                height: parent.height/1.1;
                anchors.top:parent.top
                anchors.topMargin: parent.height/20
                anchors.left:parent.left
                anchors.leftMargin: parent.height/20
                source: "./images/ParticleSwarmArrowsAnimation.gif"
            }
            AnimatedImage{
                id: animation1;
                width:parent.width/2.5;
                height: parent.height/1.1;
                anchors.top:parent.top
                anchors.topMargin: parent.height/20
                anchors.left:animation.right
                anchors.leftMargin: parent.height/20
                source: "./images/pso.gif"
            }
        }
        //Graphics Section End

        //Other Parameters Section
        Rectangle{
            id:other_parameters
            visible: true
            width:parent.width/2
            height:parent.height/4
            anchors.top:graphics.bottom
            anchors.topMargin: parent.height/20
            anchors.left: parameters_rectangle.right
            anchors.leftMargin: parent.width/40
            anchors.right:parent.right
            anchors.rightMargin: parent.height/40
            anchors.bottom:parent.bottom
            anchors.bottomMargin: parent.height/20
            border.color: "#C6C6C6"
            color:"transparent"
            Rectangle{
                anchors.left:parent.left
                anchors.leftMargin: parent.width/2-other_parameters_text.width/2
                anchors.right:parent.right
                anchors.rightMargin: parent.width/2-other_parameters_text.width/2-10
                height:parent.height/10
                anchors.top:parent.top
                anchors.topMargin: -17
                color: "#EFEFEF"
                Text {
                    id: other_parameters_text
                    anchors.left:parent.left
                    anchors.leftMargin: 5
                    horizontalAlignment: Text.AlignHCenter
                    text: qsTr("OTHER PARAMETERS")
                    font.family: "Roboto"
                    font.pixelSize: 25
                    fontSizeMode: Text.Fit
                }
            }
            //Hidden Layers
            Text {
                id: hidden_layers_text
                width:parent.width/4
                height:parent.height/10
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.top:other_parameters.top
                anchors.topMargin: 40
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                text: qsTr("HIDDEN LAYERS:")
                font.family: "Roboto"
                font.pixelSize: 23
                fontSizeMode: Text.Fit
            }
            TextField {
                id:hidden_layers_textfield
                width:parent.width/2.5
                height: parent.height/8
                anchors.left:hidden_layers_text.right
                anchors.leftMargin: 10
                anchors.top:other_parameters.top
                anchors.topMargin: 40
                text: qsTr("20")
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                placeholderText: qsTr("Hidden Layers")
                font.pixelSize: 20
                validator: RegExpValidator{
                    regExp: /[0-9 ]+/
                }
                ToolTip {
                    text: "Enter hidden layers for the network"
                    y:30
                    visible: parent.hovered
                }
            }
            //Hidden Layers End

            //Hybrid
            Text {
                id:hybrid_value
                text:hybrid_true_checked.checked?(" --hybrid --lr "+learning_rate_textfield.text):" "
                visible: false
            }
            Text {
                id: hybrid_text
                width:parent.width/4
                height:parent.height/10
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.top:dataset_generator_text.bottom
                anchors.topMargin: 30
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                text: qsTr("HYBRID:")
                font.family: "Roboto"
                font.pixelSize: 23
                fontSizeMode: Text.Fit
            }
            RadioButton{
                id:hybrid_true_checked
                hoverEnabled: true
                anchors.left: hybrid_text.right
                anchors.leftMargin: 20
                anchors.top:  dataset_generator_text.bottom
                anchors.topMargin: 25
                text: "True"
                checked: true
                font.pixelSize: 20
                ToolTip {
                    text: "Use Adam along with PSO"
                    y:30
                    visible: parent.hovered
                }
            }
            RadioButton{
                id:hybrid_false_checked
                anchors.left: hybrid_true_checked.right
                anchors.leftMargin: 20
                anchors.top: dataset_generator_text.bottom
                anchors.topMargin: 25
                text: "False"
                font.pixelSize: 20
            }
            //Hybrid End

            //Dataset Generator
            Text {
                id: dataset_generator_text
                width:parent.width/4
                height:parent.height/10
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.top:hidden_layers_text.bottom
                anchors.topMargin: 30
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                text: qsTr("DATASET GENERATOR:")
                font.family: "Roboto"
                font.pixelSize: 23
                fontSizeMode: Text.Fit
            }
            CheckBox{
                id:fixed_dataset
                anchors.left: dataset_generator_text.right
                anchors.leftMargin: 20
                anchors.top:  hidden_layers_text.bottom
                anchors.topMargin: 25
                text: "2^N fixed dataset"
                font.pixelSize: 20
                checked: random_generator.checked?false:true
            }
            CheckBox{
                id:random_generator
                anchors.left: fixed_dataset.right
                anchors.leftMargin: 20
                anchors.top:hidden_layers_text.bottom
                anchors.topMargin: 25
                text: "Random Generator"
                font.pixelSize: 20
            }
            //Dataset Generator End

            //Learning Rate
            Text {
                id: learning_rate_text
                width:parent.width/4
                visible: hybrid_true_checked.checked ? true:false
                height:parent.height/10
                anchors.left: parent.left
                anchors.leftMargin: 10
                anchors.top:hybrid_text.bottom
                anchors.topMargin: 30
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                text: qsTr("LEARNING RATE:")
                font.family: "Roboto"
                font.pixelSize: 23
                fontSizeMode: Text.Fit
            }
            TextField {
                id:learning_rate_textfield
                width:parent.width/4
                visible: hybrid_true_checked.checked ? true:false
                height: parent.height/8
                anchors.left: learning_rate_text.right
                anchors.leftMargin: 10
                anchors.top:hybrid_text.bottom
                anchors.topMargin: 30
                text: qsTr("0.01")
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                placeholderText: qsTr("Learning rate")
                font.pixelSize: 20
                validator: TextFieldDoubleValidator{
                    bottom:0
                }
                ToolTip {
                    text: "Learning Rate range 0 to infinity"
                    y:30
                    visible: parent.hovered
                }
            }
            //Learning Rate End
        }
        //Other Parameteres Section End
        //Parameters Page End

        //Benchmarking Results Page
        //Benchmarking Results Section
        Rectangle{
            id:graph_rectangle
            visible: false
            anchors.left: parent.left
            anchors.leftMargin: 40
            anchors.right:parent.right
            anchors.rightMargin: 40
            anchors.top:tabBar.bottom
            anchors.topMargin: 60
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 20
            border.color: "#C6C6C6"
            color:"transparent"

            //NN-PSO Benchmarking Results Graph
            ChartView{
                title: "NN-PSO Benchmarking Results"
                anchors.fill: parent
                antialiasing: true
                animationOptions: ChartView.SeriesAnimations
                legend.alignment:Qt.AlignBottom
                ValueAxis{
                    id:axisY
                    titleText: "time in seconds"
                    min:0
                    max:100
                }

                BarSeries {
                    id: barchart
                    axisY: axisY
                    axisX: BarCategoryAxis { categories: ["N_IN:5 HL:[3,2]", "N_IN:12 HL:[6,3,2]", "N_IN:16 HL:[8,4,2]", "N_IN:20 HL:[10,5,3,2]"]  }
                    BarSet { label: "CPU"; values: [21.53,5.11,90.16,16.87] }
                    BarSet { label: "TitanX(Pascal)"; values: [42.21, 3.87, 66.72, 7.34] }
                }
            }
            //NN-PSO Benchmarking Results Graph End
        }

        //Benchmarking Results Section End
        //Benchmarking Results Page End
        //Iris section
        Rectangle
        {
            id:iris_rectangle
            visible: false
            anchors.left: parent.left
            anchors.leftMargin: 40
            anchors.right:parent.right
            anchors.rightMargin: 40
            anchors.top:tabBar.bottom
            anchors.topMargin: 60
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 20
            border.color: "#C6C6C6"
            color:"transparent"
            Image
            {
                id: class_image
                width:parent.width-100
                height:parent.height/3
                anchors.top: parent.top
                anchors.topMargin: 10
                source: "./images/iris.png"
                anchors.horizontalCenter: parent.horizontalCenter
            }
            Text {
                id: iris_text
                width:parent.width/4
                visible: true
                height:parent.height/10
                anchors.left: parent.left
                anchors.leftMargin: 200
                anchors.top:class_image.bottom
                anchors.topMargin: 30
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                text: qsTr("ENTER TEST SAMPLE VALUES:")
                font.family: "Roboto"
                font.pixelSize: 23
                fontSizeMode: Text.Fit
            }
            TextField {
                id:iris_textfield
                width:parent.width/4
                visible: true
                height: parent.height/12
                anchors.left: iris_text.right
                anchors.leftMargin: 10
                anchors.top:class_image.bottom
                anchors.topMargin: 30
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                placeholderText: qsTr("4.2 1.3 3.7 0.2")
                font.pixelSize: 20
                ToolTip {
                    text: "Enter inputs Sepal Length, Sepal Width, Petal Length, Petal Width"
                    y:50
                    visible: parent.hovered
                }
            }

            //Test Button
            Button {
                id:test_button
                width: parent.width/5
                height:50
                anchors.left: iris_textfield.right
                anchors.leftMargin: 10
                anchors.top:class_image.bottom
                anchors.topMargin: 40
                background:Rectangle{
                    height: parent.height
                    radius: parent.width
                    border.color: "#aeadad"
                }
                Text{
                    text: qsTr("TEST");
                    font.pixelSize: 25
                    anchors.centerIn: parent
                    color: "#aeadad"
                }
                Text {
                        id: myText
                        visible: false
                        text: "Hello World"
                        anchors.centerIn: parent
                    }

                onClicked: {

                    b.onbuttonClicked("python3 test.py --pno 1 --iter 1 --hl 20  --test "+iris_textfield.text)
                    myText.text =  myFile.read();
                    output_image.source=myText.text=="Versicolor"?"./images/Iris_versicolor.jpg":myText.text=="Setosa"?"./images/Iris_setosa.jpg":"./images/Iris_virginica.jpg"
                }
                FileIO {
                       id: myFile
                       source: "output1.txt"
                       onError: console.log(msg)
                   }
            }
            Image
            {
                id: output_image
                anchors.top:iris_textfield.bottom
                anchors.topMargin: 5
                width:parent.width/2
                anchors.bottom: parent.bottom
                anchors.bottomMargin: 10
                //source: test_button.pressed?myText.text=="Versicolor"?"./images/Iris_versicolor.jpg":myText.text=="Setosa"?"./images/Iris_setosa.jpg":"./images/Iris_virginica.jpg":a
                anchors.horizontalCenter: parent.horizontalCenter
            }
        }
    }
}
//Main Window End
