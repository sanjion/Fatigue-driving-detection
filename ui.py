from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets

#主函数
import sys
import os
from glob import glob
import cv2
import myframe




def det_yolov7v6(info1):
    # 定义变量

    # 眼睛闭合判断
    EYE_AR_THRESH = 0.3  # 眼睛长宽比
    EYE_AR_CONSEC_FRAMES = 5  # 闪烁阈值

    # 嘴巴开合判断
    MAR_THRESH = 0.65  # 打哈欠长宽比
    MOUTH_AR_CONSEC_FRAMES = 3  # 闪烁阈值

    # 定义检测变量，并初始化
    COUNTER = 0  # 眨眼帧计数器
    TOTAL = 0  # 眨眼总数
    mCOUNTER = 0  # 打哈欠帧计数器
    mTOTAL = 0  # 打哈欠总数
    ActionCOUNTER = 0  # 分心行为计数器器

    # 疲劳判断变量
    # Perclos模型
    # perclos = (Rolleye/Roll) + (Rollmouth/Roll)*0.2
    Roll = 0  # 整个循环内的帧技术
    Rolleye = 0  # 循环内闭眼帧数
    Rollmouth = 0  # 循环内打哈欠数

    ui.printf("正在打开摄像头请稍后...")
    # 打开摄像头
    cap = cv2.VideoCapture(info1)
    if not cap:
        ui.printf("打开摄像头失败")
        return
    # 在前端UI输出提示信息
    ui.printf("载入成功，开始运行程序")
    ui.printf("开始执行疲劳检测...")
    ui.printf("正在使用摄像头...")
    while True:
        # 读取摄像头的一帧画面
        success, frame = cap.read()
        cv2.putText(frame, 'Blinks:{}'.format(TOTAL), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        cv2.putText(frame, 'Yawn:{}'.format(mTOTAL), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        if frame is None:
            break
        if success:
            # 检测
            # 将摄像头读到的frame传入检测函数myframe.frametest()
            ret, frame = myframe.frametest(frame)
            lab, eye, mouth = ret
            # ret和frame，为函数返回
            # frame为标注了识别结果的帧画面，画上了标识框
            # 分心行为判断
            # 分心行为检测以15帧为一个循环
            ActionCOUNTER += 1
            # 如果检测到分心行为
            for i in lab:
                if (i == "phone"):
                    ui.printf("正在用手机")
                    ui.printf("请不要分心")
                    if ActionCOUNTER > 0:
                        ActionCOUNTER -= 1
                elif (i == "smoke"):
                    ui.printf("正在抽烟")
                    ui.printf("请不要分心")
                    if ActionCOUNTER > 0:
                        ActionCOUNTER -= 1
                elif (i == "drink"):
                    ui.printf("正在用喝水")
                    ui.printf("请不要分心")
                    if ActionCOUNTER > 0:
                        ActionCOUNTER -= 1

            # 如果超过15帧未检测到分心行为，将label修改为平时状态
            if ActionCOUNTER == 15:
                ActionCOUNTER = 0

            # 疲劳判断
            # 眨眼判断
            if eye < EYE_AR_THRESH:
                COUNTER += 1
                Rolleye += 1
            else:
                # 如果连续2次都小于阈值，则表示进行了一次眨眼活动
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                    # 重置眼帧计数器
                    COUNTER = 0

            # 哈欠判断，同上
            if mouth > MAR_THRESH:
                mCOUNTER += 1
                Rollmouth += 1
            else:
                # 如果连续3次都小于阈值，则表示打了一次哈欠
                if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:
                    mTOTAL += 1
                    # 重置嘴帧计数器
                    mCOUNTER = 0

            # 将画面显示在前端UI上
            ui.showimg(frame)
            QApplication.processEvents()
            # 疲劳模型
            # 疲劳模型以150帧为一个循环
            # 每一帧Roll加1
            Roll += 1
            # 当检测满150帧时，计算模型得分
            if Roll == 150:
                # 计算Perclos模型得分
                perclos = (Rolleye / Roll) + (Rollmouth / Roll) * 0.2
                # 在前端UI输出perclos值
                ui.printf("过去150帧中，Perclos得分为" + str(round(perclos, 3)))
                # 当过去的150帧中，Perclos模型得分超过0.38时，判断为疲劳状态
                if perclos > 0.38:
                    ui.printf("当前处于疲劳状态")
                else:
                    ui.printf("当前处于清醒状态")

                # 归零
                # 将三个计数器归零
                # 重新开始新一轮的检测
                Roll = 0
                Rolleye = 0
                Rollmouth = 0
                ui.printf("重新开始执行疲劳检测...")



class Thread_1(QThread):  # 线程1
    def __init__(self,info1):
        super().__init__()
        self.info1=info1
        self.run2(self.info1)

    def run2(self, info1):
        result = []
        result = det_yolov7v6(info1)




class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 960)
        MainWindow.setStyleSheet("background-image: url(\"./template/carui.png\")")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(168, 60, 600, 71))
        self.label.setAutoFillBackground(False)
        self.label.setStyleSheet("")
        self.label.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label.setStyleSheet("font-size:50px;font-weight:bold;font-family:SimHei;background:rgba(255,255,255,0);")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 188, 800, 551))
        self.label_2.setStyleSheet("background:rgba(255,255,255,0.7);")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(23, 780, 921, 164))
        self.textBrowser.setStyleSheet("background:rgba(255,255,255,0.7);")
        self.textBrowser.setObjectName("textBrowser")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(1000, 780, 150, 40))
        self.pushButton.setStyleSheet("background:rgba(53,142,255,1);border-radius:10px;padding:2px 4px;")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(1000, 840, 150, 40))
        self.pushButton_2.setStyleSheet("background:rgba(53,142,255,1);border-radius:10px;padding:2px 4px;")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(1000, 900, 150, 40))
        self.pushButton_3.setStyleSheet("background:rgba(53,142,255,1);border-radius:10px;padding:2px 4px;")
        self.pushButton_3.setObjectName("pushButton_3")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "疲劳驾驶检测系统"))
        self.label.setText(_translate("MainWindow", "疲劳驾驶检测系统"))
        self.label_2.setText(_translate("MainWindow", "请点击以添加视频，不选择则为实时识别"))
        self.pushButton.setText(_translate("MainWindow", "选择文件"))
        self.pushButton_2.setText(_translate("MainWindow", "文件/实时识别"))
        self.pushButton_3.setText(_translate("MainWindow", "退出系统"))


        # 点击文本框绑定槽事件
        self.pushButton.clicked.connect(self.openfile)
        self.pushButton_2.clicked.connect(self.click_1)
        self.pushButton_3.clicked.connect(self.handleCalc3)


    def openfile(self):
        global sname,filepath
        fname = QFileDialog()
        fname.setAcceptMode(QFileDialog.AcceptOpen)
        fname, _ = fname.getOpenFileName()
        if fname == '':
            return
        filepath = os.path.normpath(fname)
        sname = filepath.split(os.sep)
        ui.printf("当前选择的文件路径是：%s" % filepath)


    def handleCalc3(self):
        os._exit(0)

    def printf(self,text):
        self.textBrowser.append(text)
        self.cursor = self.textBrowser.textCursor()
        self.textBrowser.moveCursor(self.cursor.End)
        QtWidgets.QApplication.processEvents()

    def showimg(self,img):
        global vid
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _image = QtGui.QImage(img2[:], img2.shape[1], img2.shape[0], img2.shape[1] * 3,
                              QtGui.QImage.Format_RGB888)
        n_width = _image.width()
        n_height = _image.height()
        if n_width / 500 >= n_height / 400:
            ratio = n_width / 500
        else:
            ratio = n_height / 500
        new_width = int(n_width / ratio)
        new_height = int(n_height / ratio)
        new_img = _image.scaled(new_width, new_height, Qt.KeepAspectRatio)
        print(img2.shape[1],img2.shape[0])
        self.label_2.setPixmap(QPixmap.fromImage(new_img))

    def click_1(self):
        try:
            self.thread_1.quit()
        except:
            pass
        try:
            self.thread_1 = Thread_1(filepath)  # 创建线程
        except:
            self.thread_1 = Thread_1(0)  # 创建线程
        self.thread_1.wait()
        self.thread_1.start()  # 开始线程



class LoginDialog(QDialog):
    def __init__(self, *args, **kwargs):
        '''
        构造函数，初始化登录对话框的内容
        :param args:
        :param kwargs:
        '''
        super().__init__(*args, **kwargs)
        self.setWindowTitle('欢迎登录')  # 设置标题
        self.resize(600, 500)  # 设置宽、高
        self.setFixedSize(self.width(), self.height())
        self.setWindowFlags(Qt.WindowCloseButtonHint)  # 设置隐藏关闭X的按钮
        self.setStyleSheet("background-image: url(\"./template/1.png\")")

        '''
        定义界面控件设置
        '''
        self.frame = QFrame(self)
        self.frame.setStyleSheet("background:rgba(255,255,255,0);")
        self.frame.move(185, 180)

        # self.verticalLayout = QVBoxLayout(self.frame)
        self.mainLayout = QVBoxLayout(self.frame)

        # self.nameLb1 = QLabel('&Name', self)
        # self.nameLb1.setFont(QFont('Times', 24))
        self.nameEd1 = QLineEdit(self)
        self.nameEd1.setFixedSize(150, 30)
        self.nameEd1.setPlaceholderText("账号")
        # 设置透明度
        op1 = QGraphicsOpacityEffect()
        op1.setOpacity(0.5)
        self.nameEd1.setGraphicsEffect(op1)
        # 设置文本框为圆角
        self.nameEd1.setStyleSheet('''QLineEdit{border-radius:5px;}''')
        # self.nameLb1.setBuddy(self.nameEd1)


        self.nameEd3 = QLineEdit(self)
        self.nameEd3.setPlaceholderText("密码")
        op5 = QGraphicsOpacityEffect()
        op5.setOpacity(0.5)
        self.nameEd3.setGraphicsEffect(op5)
        self.nameEd3.setStyleSheet('''QLineEdit{border-radius:5px;}''')

        self.btnOK = QPushButton('登录')
        op3 = QGraphicsOpacityEffect()
        op3.setOpacity(1)
        self.btnOK.setGraphicsEffect(op3)
        self.btnOK.setStyleSheet(
            '''QPushButton{background:#1E90FF;border-radius:5px;}QPushButton:hover{background:#4169E1;}\
            QPushButton{font-family:'Arial';color:#FFFFFF;}''')  # font-family中可以设置字体大小，如下font-size:24px;

        self.btnCancel = QPushButton('注册')
        op4 = QGraphicsOpacityEffect()
        op4.setOpacity(1)
        self.btnCancel.setGraphicsEffect(op4)
        self.btnCancel.setStyleSheet(
            '''QPushButton{background:#1E90FF;border-radius:5px;}QPushButton:hover{background:#4169E1;}\
            QPushButton{font-family:'Arial';color:#FFFFFF;}''')

        # self.btnOK.setFont(QFont('Microsoft YaHei', 24))
        # self.btnCancel.setFont(QFont('Microsoft YaHei', 24))

        # self.mainLayout.addWidget(self.nameLb1, 0, 0)
        self.mainLayout.addWidget(self.nameEd1)

        # self.mainLayout.addWidget(self.nameLb2, 1, 0)

        self.mainLayout.addWidget(self.nameEd3)

        self.mainLayout.addWidget(self.btnOK)
        self.mainLayout.addWidget(self.btnCancel)

        self.mainLayout.setSpacing(50)


        # 绑定按钮事件
        self.btnOK.clicked.connect(self.button_enter_verify)
        self.btnCancel.clicked.connect(self.button_register_verify)  # 返回按钮绑定到退出

    def button_register_verify(self):
        global path1
        path1 = './user'
        if not os.path.exists(path1):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path1)
        user = self.nameEd1.text()
        pas = self.nameEd3.text()
        with open(path1 + '/' + user + '.txt', "w") as f:
            f.write(pas)
        self.nameEd1.setText("注册成功")


    def button_enter_verify(self):
        # 校验账号是否正确
        global administrator, userstext, passtext,strlist,name
        userstext = []
        passtext = []
        administrator = 0
        pw = 0
        path1 = './user'
        if not os.path.exists(path1):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path1)
        users = os.listdir(path1)

        for i in users:
            with open(path1 + '/' + i, "r") as f:
                userstext.append(i[:-4])
                passtext.append(f.readline())

        for i in users:
            if i[:-4] == self.nameEd1.text():
                with open(path1 + '/' + i, "r") as f:
                    lines = f.readline()
                    strlist = lines.split(',')
                    print(strlist)
                    print(strlist[0])
                    if strlist[0] == self.nameEd3.text():
                        name = i
                        if i[:2] == 'GM':
                            administrator = 1
                            self.accept()
                        else:
                            passtext.append(f.readline())
                            self.accept()
                    else:
                        self.nameEd3.setText("密码错误")
                        pw = 1
        if pw == 0:
            self.nameEd1.setText("账号错误")



if __name__ == "__main__":
    # 创建应用
    window_application = QApplication(sys.argv)
    # 设置登录窗口
    login_ui = LoginDialog()
    # 校验是否验证通过
    if login_ui.exec_() == QDialog.Accepted:
        # 初始化主功能窗口
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        MainWindow.show()
        if administrator == 1:
            ui.printf('欢迎管理员')
        else:
            ui.printf('欢迎用户')
        # 设置应用退出
        sys.exit(window_application.exec_())

