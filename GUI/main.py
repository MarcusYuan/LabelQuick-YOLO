import sys, os
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QCoreApplication, QRect, pyqtSignal,QTimer
from PyQt5.QtCore import Qt, QLineF,QUrl
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import cv2
import numpy as np
import logging
import datetime
from util.QtFunc import *
from util.xmlfile import *
from util.yolofile import *

# 配置日志记录
def setup_logger():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"labelquick_{timestamp}.log")
    
    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 初始化日志记录器
logger = setup_logger()
logger.info("LabelQuick 应用程序启动")

from GUI.UI_Main import Ui_MainWindow
from GUI.message import LabelInputDialog

sys.path.append("smapro")
from sampro.LabelQuick_TW import Anything_TW
from sampro.LabelVideo_TW import AnythingVideo_TW

from PyQt5.QtCore import QThread, pyqtSignal, QTimer

class VideoProcessingThread(QThread):
    finished = pyqtSignal()  # 完成信号
    frame_ready = pyqtSignal(object)  # 添加新信号用于传递处理后的帧

    def __init__(self, avt, video_path, output_dir, clicked_x, clicked_y, method, text, save_path, format_type="XML", class_map=None):
        super().__init__()
        logger.info("初始化视频处理线程")
        self.AVT = avt
        self.video_path = video_path
        self.output_dir = output_dir
        self.clicked_x = clicked_x
        self.clicked_y = clicked_y
        self.method = method
        self.text = text
        self.save_path = save_path
        self.xml_messages = []
        self.format_type = format_type
        self.class_map = class_map if class_map else {}
        os.makedirs(self.output_dir, exist_ok=True)
        logger.debug(f"视频处理线程初始化完成: 视频路径={video_path}, 输出目录={output_dir}, 格式={format_type}")

    def run(self):
        logger.info(f"开始视频处理: 点击坐标=({self.clicked_x}, {self.clicked_y}), 方法={self.method}")
        try:
            # 创建输出目录和mask子目录
            os.makedirs(self.output_dir, exist_ok=True)
            mask_dir = os.path.join(self.output_dir, "mask")
            os.makedirs(mask_dir, exist_ok=True)
            logger.debug(f"创建输出目录: {self.output_dir} 和 mask 子目录: {mask_dir}")
            
            # 确保YOLO目录结构存在
            images_dir = os.path.join(self.output_dir, "images")
            labels_dir = os.path.join(self.output_dir, "labels")
            train_images_dir = os.path.join(images_dir, "train")
            train_labels_dir = os.path.join(labels_dir, "train")
            
            os.makedirs(train_images_dir, exist_ok=True)
            os.makedirs(train_labels_dir, exist_ok=True)
            logger.debug(f"确保YOLO目录结构存在: {train_images_dir}, {train_labels_dir}")
            
            # 从视频中提取所有帧
            logger.info(f"开始从视频提取帧: {self.video_path}")
            try:
                # 提取视频帧，使用合适的fps
                video_dir, frame_count = self.AVT.extract_frames_from_video(self.video_path, self.output_dir, fps=5)
                logger.info(f"视频帧提取完成，共 {frame_count} 帧")
                
                # 复制所有帧到训练图像目录
                logger.info("复制所有帧到训练图像目录")
                frame_files = [f for f in os.listdir(self.output_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                import shutil
                for frame_file in frame_files:
                    source_path = os.path.join(self.output_dir, frame_file)
                    target_path = os.path.join(train_images_dir, frame_file)
                    if not os.path.exists(target_path):
                        shutil.copy2(source_path, target_path)
                logger.info(f"已复制 {len(frame_files)} 个帧到训练图像目录")
                
                # 发送信号更新UI
                frame = cv2.imread(os.path.join(self.output_dir, "0.jpg"))
                if frame is not None:
                    self.frame_ready.emit(frame)
                    logger.debug(f"发送第一帧到UI")
            except Exception as e:
                logger.error(f"提取视频帧时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.finished.emit()
                return
            
            # 设置视频目录
            logger.debug(f"设置 AVT 的视频目录: {self.output_dir}")
            self.AVT.set_video(self.output_dir)
            
            # 处理每一帧
            if not hasattr(self.AVT, 'frame_paths') or not self.AVT.frame_paths:
                logger.error("没有找到视频帧，无法继续处理")
                self.finished.emit()
                return
                
            logger.info(f"开始处理视频帧，总帧数: {len(self.AVT.frame_paths)}")
            
            # 确保已经设置了点击坐标和方法
            if self.clicked_x is None or self.clicked_y is None or self.method is None:
                logger.error("未设置点击坐标或方法，无法继续处理")
                self.finished.emit()
                return
            
            # 设置点击坐标
            logger.debug(f"设置 AVT 的点击坐标: ({self.clicked_x}, {self.clicked_y}), 方法: {self.method}")
            self.AVT.Set_Clicked([self.clicked_x, self.clicked_y], self.method)
            
            # 确保推理状态已初始化
            if not hasattr(self.AVT, 'inference_state') or self.AVT.inference_state is None:
                logger.debug(f"初始化 AVT 的推理状态: {self.output_dir}")
                self.AVT.inference(self.output_dir)
            
            # 使用新的自动视频打标方法
            logger.info("开始自动视频打标")
            processed_frames, xml_messages = self.AVT.auto_video_labeling(
                save_image_path=mask_dir, 
                save_path=self.save_path, 
                text=self.text,
                format_type=self.format_type
            )
            
            # 保存 XML 消息以供后续处理
            self.xml_messages = xml_messages if xml_messages else []
            logger.debug(f"收到 {len(self.xml_messages)} 个标签消息")
            
            # 如果没有处理帧，记录警告
            if not self.xml_messages:
                logger.warning("没有生成标签消息，自动视频打标可能失败")
            
            # 根据格式类型处理标签
            if self.format_type.upper() == "YOLO" and self.xml_messages:
                logger.info(f"保存所有 YOLO 格式标签，共 {len(self.xml_messages)} 个帧")
                
                # 保存类别映射文件到标签目录
                classes_path = os.path.join(labels_dir, "classes.txt")
                sorted_classes = sorted(self.class_map.items(), key=lambda x: x[1])
                with open(classes_path, 'w') as f:
                    for class_name, _ in sorted_classes:
                        f.write(f"{class_name}\n")
                logger.info(f"类别映射已保存到: {classes_path}")
                
                # 处理每个标签消息
                for msg in self.xml_messages:
                    if len(msg) >= 3:  # 确保消息有足够的元素
                        result, file_path, size = msg
                        
                        # 获取帧名称
                        frame_name = os.path.splitext(os.path.basename(file_path))[0]
                        
                        # 创建标签文件路径
                        label_file_path = os.path.join(train_labels_dir, f"{frame_name}.txt")
                        logger.debug(f"为帧 {frame_name} 保存 YOLO 标签到 {label_file_path}")
                        
                        # 直接写入标签文件
                        with open(label_file_path, 'w') as f:
                            class_name = self.text
                            class_id = self.class_map.get(class_name, 0)
                            
                            # 获取边界框坐标
                            x, y, x2, y2 = result['bndbox']
                            
                            # 获取图像尺寸
                            # 修复 size 解包错误 - size 是一个列表 [width, height, channels]
                            try:
                                if isinstance(size, (list, tuple)) and len(size) >= 2:
                                    img_width = size[0]
                                    img_height = size[1]
                                    logger.debug(f"从 size 获取图像尺寸: {img_width}x{img_height}")
                                else:
                                    # 如果 size 不是预期格式，使用 result 中的值
                                    logger.warning(f"意外的 size 格式: {size}，使用 result 中的值")
                                    img_width = result.get('img_width', 1300)
                                    img_height = result.get('img_height', 736)
                                    logger.debug(f"从 result 获取图像尺寸: {img_width}x{img_height}")
                            except Exception as e:
                                logger.error(f"获取图像尺寸时出错: {e}")
                                # 使用默认值
                                img_width = 1300
                                img_height = 736
                                logger.debug(f"使用默认图像尺寸: {img_width}x{img_height}")
                            
                            # 转换为 YOLO 格式
                            x_center, y_center, width, height = convert_to_yolo_format(img_width, img_height, x, y, x2, y2)
                            
                            # 写入文件
                            line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                            f.write(line)
                            logger.debug(f"写入行: {line.strip()}")
                        
                        logger.info(f"YOLO 标签已保存到: {label_file_path}")
                
                # 如果标签数量少于帧数量，为所有帧创建相同的标签
                if len(self.xml_messages) < frame_count * 0.5:
                    logger.warning(f"只生成了 {len(self.xml_messages)} 个标签，总帧数为 {frame_count}，尝试为所有帧创建相同的标签")
                    
                    # 获取第一个标签消息
                    if self.xml_messages:
                        first_msg = self.xml_messages[0]
                        result, _, size = first_msg
                        
                        # 获取边界框坐标
                        x, y, x2, y2 = result['bndbox']
                        
                        # 获取图像尺寸
                        if isinstance(size, (list, tuple)) and len(size) >= 2:
                            img_width = size[0]
                            img_height = size[1]
                        else:
                            img_width = 1300
                            img_height = 736
                        
                        # 转换为 YOLO 格式
                        x_center, y_center, width, height = convert_to_yolo_format(img_width, img_height, x, y, x2, y2)
                        
                        # 为所有帧创建相同的标签
                        class_name = self.text
                        class_id = self.class_map.get(class_name, 0)
                        line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                        
                        for i in range(frame_count):
                            label_file_path = os.path.join(train_labels_dir, f"{i}.txt")
                            if not os.path.exists(label_file_path):
                                with open(label_file_path, 'w') as f:
                                    f.write(line)
                                logger.debug(f"为帧 {i} 创建相同的标签: {line.strip()}")
                        
                        logger.info(f"已为所有 {frame_count} 个帧创建标签")
            
            logger.info("视频处理完成，发送完成信号")
            self.finished.emit()
        except Exception as e:
            logger.error(f"视频处理线程出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
        self.finished.emit()


class MainFunc(QMainWindow):
    my_signal = pyqtSignal()

    def __init__(self):
        super(MainFunc, self).__init__()
        logger.info("初始化主界面")
        # 连接应用程序的 aboutToQuit 信号到自定义的槽函数
        QCoreApplication.instance().aboutToQuit.connect(self.clean_up)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        logger.debug("UI 设置完成")

        self.sld_video_pressed=False

        self.image_files = None
        self.img_path = None
        self.save_path = None
        self.clicked_event = False
        self.paint_event = False
        self.labels = []
        self.clicked_save = []
        self.paint_save = []
        self.flag = False
        self.save = True
        self.cap = None
        self.video_path = None
        logger.debug("初始化变量完成")

        self.AT = Anything_TW()
        self.AVT = AnythingVideo_TW()
        logger.debug("初始化 Anything_TW 和 AnythingVideo_TW 完成")

        self.timer_camera = QTimer()
        logger.debug("初始化定时器完成")

        # 连接信号和槽
        logger.debug("开始连接信号和槽")
        self.ui.actionOpen_Dir.triggered.connect(self.get_dir)
        self.ui.actionNext_Image.triggered.connect(self.next_img)
        self.ui.actionPrev_Image.triggered.connect(self.prev_img)
        self.ui.actionChange_Save_Dir.triggered.connect(self.set_save_path)
        self.ui.actionCreate_RectBox.triggered.connect(self.mousePaint)
        self.ui.actionOpen_Video.triggered.connect(self.get_video)
        self.ui.actionVideo_marking.triggered.connect(self.video_marking)

        self.ui.pushButton.clicked.connect(self.Btn_Start)
        self.ui.pushButton_2.clicked.connect(self.Btn_Stop)
        self.ui.pushButton_3.clicked.connect(self.Btn_Save)
        self.ui.pushButton_4.clicked.connect(self.Btn_Replay)
        self.ui.pushButton_5.clicked.connect(self.Btn_Auto)
        self.ui.pushButton_start_marking.clicked.connect(self.Btn_Start_Marking)

        self.ui.horizontalSlider.sliderReleased.connect(self.releaseSlider)
        self.ui.horizontalSlider.sliderPressed.connect(self.pressSlider)
        self.ui.horizontalSlider.sliderMoved.connect(self.moveSlider)
        logger.debug("信号和槽连接完成")

        # 获取视频总帧数和当前帧位置
        self.total_frames = 0
        self.current_frame = 0

        # 添加标签格式选择变量
        self.format_type = "xml"  # 默认使用 XML 格式
        self.class_map = {}  # 类别名称到索引的映射
        logger.debug(f"初始化标签格式: {self.format_type}")
        
        # 连接 comboBox 的信号
        self.ui.comboBox.currentTextChanged.connect(self.onFormatChanged)
        logger.info("主界面初始化完成")

    def Change_Enable(self,method="",state=False):
        if method=="ShowVideo":
            self.ui.pushButton.setEnabled(state)
            self.ui.pushButton_2.setEnabled(state)
            self.ui.pushButton_3.setEnabled(state)
            self.ui.pushButton_4.setEnabled(state)  # 初始时禁用重播按钮
            self.ui.pushButton_5.setEnabled(state)
            self.ui.horizontalSlider.setEnabled(state)
        if method=="MakeTag":
            self.ui.actionPrev_Image.setEnabled(state)
            self.ui.actionNext_Image.setEnabled(state)
            self.ui.actionCreate_RectBox.setEnabled(state)
            
    def get_dir(self):
        self.ui.listWidget.clear()
        if self.cap:
            self.timer_camera.stop()
            self.ui.listWidget.clear()  # 清空listWidget
        self.directory = QtWidgets.QFileDialog.getExistingDirectory()
        if self.directory:
            self.image_files = list_images_in_directory(self.directory)
            self.current_index = 0
            self.show_path_image()
            self.Change_Enable(method="MakeTag",state=True)
            self.Change_Enable(method="ShowVideo",state=False)
            # 禁用开始检测打标按钮
            self.ui.pushButton_start_marking.setEnabled(False)
            # 鼠标点击触发
            self.ui.label_4.mousePressEvent = self.mouse_press_event

    def show_path_image(self):
        if self.image_files:
            self.image_path = self.image_files[self.current_index]
            # print(self.image_path)
            self.img_path = self.image_path
            self.image_name = os.path.basename(self.image_path).split('.')[0]
            # print(self.image_name)

            self.img_path, self.img_width, self.img_height = Change_image_Size(self.img_path)
            self.image = cv2.imread(self.img_path)
            self.AT.Set_Image(self.image)
            self.show_qt(self.img_path)
            self.Exists_Labels_And_Boxs()

    # 展示已保存所有标签
    def Exists_Labels_And_Boxs(self):
        logger.info(f"检查图像 {self.image_name} 的已存在标签")
        self.list_labels = []
        self.ui.listWidget.clear()  # 清空标签列表
        self.clicked_save = []  # 清空已保存的点击标签
        self.paint_save = []  # 清空已保存的绘制标签
        
        # 检查 XML 格式标签文件
        xml_file = os.path.exists(f"{self.save_path}/{self.image_name}.xml")
        # 检查 YOLO 格式标签文件
        yolo_file = os.path.exists(f"{self.save_path}/{self.image_name}.txt")
        
        logger.debug(f"XML 文件存在: {xml_file}, YOLO 文件存在: {yolo_file}, 当前格式: {self.format_type}")
        
        # 根据当前选择的格式和文件存在情况加载标签
        if xml_file and (self.format_type.upper() == "XML" or not yolo_file):
            # 处理 XML 格式标签
            label_path = f"{self.save_path}/{self.image_name}.xml"
            logger.debug(f"加载 XML 标签文件: {label_path}")
            self.labels = get_labels(label_path)
            self.list_labels, list_box = list_label(label_path)
            self.paint_save = list_box
            logger.debug(f"从 XML 加载了 {len(self.list_labels)} 个标签和 {len(list_box)} 个边界框")
            
            # 更新 class_map
            if self.format_type.upper() == "YOLO":
                for i, label in enumerate(self.list_labels):
                    if label not in self.class_map:
                        self.class_map[label] = len(self.class_map)
                        logger.debug(f"将类别 '{label}' 添加到 class_map，索引为 {self.class_map[label]}")
            
            # 显示标签和边界框
            self.Show_Exists()
            
            # 更新 UI 列表
            for label in self.list_labels:
                self.ui.listWidget.addItem(label)
                logger.debug(f"添加标签到 UI: {label}")
                
        elif yolo_file and (self.format_type.upper() == "YOLO" or not xml_file):
            # 处理 YOLO 格式标签
            try:
                # 读取 YOLO 格式标签文件
                yolo_path = f"{self.save_path}/{self.image_name}.txt"
                logger.debug(f"加载 YOLO 标签文件: {yolo_path}")
                with open(yolo_path, 'r') as f:
                    lines = f.readlines()
                
                logger.debug(f"YOLO 文件包含 {len(lines)} 行")
                self.labels = []
                
                # 如果 class_map 为空，尝试从类别文件加载
                if not self.class_map:
                    class_file = os.path.join(self.save_path, "classes.txt")
                    if os.path.exists(class_file):
                        logger.debug(f"从 {class_file} 加载类别映射")
                        with open(class_file, 'r') as f:
                            classes = [line.strip() for line in f.readlines()]
                            self.class_map = {cls: i for i, cls in enumerate(classes)}
                            logger.debug(f"加载的类别映射: {self.class_map}")
                
                # 反向映射：从类别索引到类别名称
                reverse_class_map = {v: k for k, v in self.class_map.items()} if self.class_map else {}
                logger.debug(f"反向类别映射: {reverse_class_map}")
                
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    logger.debug(f"处理第 {i+1} 行: {line.strip()}")
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # 将归一化坐标转换回像素坐标
                        x = int((x_center - width/2) * self.img_width)
                        y = int((y_center - height/2) * self.img_height)
                        w = int(width * self.img_width)
                        h = int(height * self.img_height)
                        
                        logger.debug(f"YOLO 坐标 ({x_center}, {y_center}, {width}, {height}) -> 像素坐标 ({x}, {y}, {w}, {h})")
                        
                        # 获取类别名称
                        class_name = reverse_class_map.get(class_id, f"class_{class_id}")
                        logger.debug(f"类别 ID {class_id} 映射到名称: {class_name}")
                        
                        # 创建标签信息
                        result = {
                            'name': class_name,
                            'img_width': self.img_width,
                            'img_height': self.img_height,
                            'bndbox': [x, y, x+w, y+h]  # 修正为 x2, y2 格式
                        }
                        
                        self.labels.append(result)
                        self.paint_save.append([x, y, x + w, y + h])
                        self.list_labels.append(class_name)
                        self.ui.listWidget.addItem(class_name)
                        logger.debug(f"添加标签到 UI: {class_name}")
                
                self.Show_Exists()
                logger.info(f"已加载 YOLO 格式标签: {yolo_path}, 共 {len(self.labels)} 个标签")
            except Exception as e:
                logger.error(f"读取 YOLO 标签文件出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.debug(f"未找到标签文件或格式不匹配: XML={xml_file}, YOLO={yolo_file}, 格式={self.format_type}")
            self.Show_Exists()  # 显示原始图像

    def show_qt(self, img_path):
        if img_path != None:
            Qt_Gui = QtGui.QPixmap(img_path)
            self.ui.label_3.setFixedSize(self.img_width, self.img_height)
            self.ui.label_3.setPixmap(Qt_Gui)

    def next_img(self):
        if self.img_path and not self.clicked_event and not self.paint_event:
            if self.image_files and self.current_index < len(self.image_files) - 1:
                self.current_index += 1
                print(self.current_index)
                self.Other_Img()
            else:
                upWindowsh("这是最后一张")

    def prev_img(self):
        if self.img_path and not self.clicked_event and not self.paint_event:
            if self.image_files and self.current_index > 0:
                self.current_index -= 1
                self.Other_Img()
                
            else:
                upWindowsh("这是第一张")
                
    def Other_Img(self):
        self.labels = []
        self.paint_save = []
        self.clicked_save = []
        self.ui.listWidget.clear()
        self.show_path_image()

    def set_save_path(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory()
        if directory:
            self.save_path = directory
            if self.img_path:
                self.Exists_Labels_And_Boxs()

# ########################################################################################################################
    # seg
    def mouse_press_event(self, event):
        # 获取鼠标点击位置
        x = event.pos().x()
        y = event.pos().y()
        logger.info(f"鼠标点击事件: 坐标=({x}, {y})")
        
        # 设置点击坐标和方法
        self.clicked_x = x
        self.clicked_y = y
        self.method = 1  # 默认方法为1（正样本）
        logger.debug(f"设置点击坐标: ({self.clicked_x}, {self.clicked_y}), 方法: {self.method}")
        
        # 如果是视频模式，设置 AVT 的点击坐标
        if hasattr(self, 'AVT') and self.AVT:
            logger.debug(f"设置 AVT 的点击坐标: ({self.clicked_x}, {self.clicked_y}), 方法: {self.method}")
            self.AVT.Set_Clicked([self.clicked_x, self.clicked_y], self.method)
            
            # 只有在 inference_state 为 None 时才初始化推理状态
            if hasattr(self.AVT, 'inference_state') and self.AVT.inference_state is None:
                if hasattr(self, 'output_dir') and self.output_dir:
                    try:
                        logger.debug(f"初始化 AVT 的推理状态: {self.output_dir}")
                        self.AVT.inference(self.output_dir)
                        logger.debug("AVT 推理状态初始化成功")
                    except Exception as e:
                        logger.error(f"AVT 推理初始化出错: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
            
            # 使用 Create_Mask 方法创建掩码
            try:
                if hasattr(self.AVT, 'Create_Mask'):
                    logger.debug("调用 AVT.Create_Mask")
                    success = self.AVT.Create_Mask()
                    if success:
                        logger.debug("AVT.Create_Mask 调用成功")
                    else:
                        logger.warning("AVT.Create_Mask 调用失败")
                else:
                    logger.warning("AVT 没有 Create_Mask 方法")
            except Exception as e:
                logger.error(f"创建掩码出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # 设置 AT 的点击坐标
        if hasattr(self, 'AT') and self.AT:
            logger.debug(f"设置 AT 的点击坐标: ({self.clicked_x}, {self.clicked_y}), 方法: {self.method}")
            self.AT.Set_Clicked([self.clicked_x, self.clicked_y], self.method)
            # 使用 Create_Mask 方法创建掩码
            logger.debug("调用 AT.Create_Mask")
            self.AT.Create_Mask()
            
            # 获取边界框坐标
            logger.debug("调用 AT.Draw_Mask")
            image = self.AT.Draw_Mask(self.AT.mask, self.image.copy())
            
            # 显示边界框
            h, w, channels = image.shape
            bytes_per_line = channels * w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            Qt_Gui = QtGui.QPixmap(q_image)
            self.ui.label_3.setFixedSize(self.img_width, self.img_height)
            self.ui.label_3.setPixmap(Qt_Gui)

            # 设置事件状态
            self.clicked_event = True
            self.paint_event = False
            logger.debug("设置点击事件状态: clicked_event=True, paint_event=False")

# ########################################################################################################################
# 重写QWidget类的keyPressEvent方法
    def keyPressEvent(self, event):
        if self.img_path:
            if self.clicked_event and not self.paint_event:
                image = self.AT.Key_Event(event.key())

            if self.video_path:
                if self.clicked_event or self.paint_event:
                    if (event.key() == 83):  # S 键
                        self.save = True
                        self.dialog = LabelInputDialog(self)
                        self.dialog.show()
                        self.dialog.confirmed.connect(self.video_on_dialog_confirmed)
                        
                        # 禁用label4的鼠标事件
                        self.ui.label_4.mousePressEvent = None

                    if (event.key() == 81):  # Q 键
                        self.clicked_event = False
                        self.paint_event = False
                        self.save = True
                        self.Show_Exists()
                        self.ui.label_4.mousePressEvent = self.mouse_press_event
                        self.ui.label_4.setCursor(Qt.ArrowCursor)
            else:
                if self.clicked_event or self.paint_event:
                    if (event.key() == 83):  # S 键
                        self.save = True
                        self.dialog = LabelInputDialog(self)
                        self.dialog.show()
                        self.dialog.confirmed.connect(self.on_dialog_confirmed)

                    if (event.key() == 81):  # Q 键
                        self.clicked_event = False
                        self.paint_event = False
                        self.save = True
                        self.Show_Exists()
                        self.ui.label_4.mousePressEvent = self.mouse_press_event
                        self.ui.label_4.setCursor(Qt.ArrowCursor)

                # 添加 D 键调试功能
                if (event.key() == 68):  # D 键
                    self.debug_check_labels()
                
            if (event.key() == 16777219):  # 删除键
                    self.clicked_event = False
                    self.paint_event = False
                    self.save = True
                    self.ui.listWidget.clear()
                    self.list_labels = []
                    self.clicked_save = []
                    self.paint_save = []
                    self.show_qt(self.img_path)
                    self.ui.label_4.mousePressEvent = self.mouse_press_event
                    self.ui.label_4.setCursor(Qt.ArrowCursor)
                    
                    # 删除 XML 格式标签文件
                    xml_path = f"{self.save_path}/{self.image_name}.xml"
                    if os.path.exists(xml_path):
                        os.remove(xml_path)
                        self.labels = []
                        print(f"已删除 XML 标签文件: {xml_path}")
                    
                    # 删除 YOLO 格式标签文件
                    yolo_path = f"{self.save_path}/{self.image_name}.txt"
                    if os.path.exists(yolo_path):
                        os.remove(yolo_path)
                        self.labels = []
                        print(f"已删除 YOLO 标签文件: {yolo_path}")
                    
                    else:
                        super(QMainWindow, self).keyPressEvent(event)

            


    
    def on_dialog_confirmed(self, text):
        self.text = text
        logger.info(f"对话框确认，标签文本: {text}")
        if not self.save_path:
            logger.warning("未设置保存路径")
            upWindowsh("请选择保存路径")
        elif text and self.clicked_event:
            logger.debug(f"添加标签到列表: {text}")
            self.ui.listWidget.addItem(text)
            
            # 确保 list_labels 已初始化
            if not hasattr(self, 'list_labels'):
                logger.debug("初始化 list_labels 列表")
                self.list_labels = []
            
            # 确保 labels 已初始化
            if not hasattr(self, 'labels'):
                logger.debug("初始化 labels 列表")
                self.labels = []
            
            # 确保YOLO目录结构存在
            images_dir = os.path.join(self.save_path, "images")
            labels_dir = os.path.join(self.save_path, "labels")
            train_images_dir = os.path.join(images_dir, "train")
            train_labels_dir = os.path.join(labels_dir, "train")
            
            os.makedirs(train_images_dir, exist_ok=True)
            os.makedirs(train_labels_dir, exist_ok=True)
            logger.debug(f"确保YOLO目录结构存在: {train_images_dir}, {train_labels_dir}")
            
            # 复制当前图像到训练图像目录
            train_image_path = os.path.join(train_images_dir, f"{self.image_name}.jpg")
            if not os.path.exists(train_image_path) and self.image_path:
                import shutil
                shutil.copy2(self.image_path, train_image_path)
                logger.debug(f"复制图像到训练目录: {train_image_path}")
            
            # 根据选择的格式保存标签
            if self.format_type.upper() == "XML":
                # XML 格式保存
                logger.debug(f"使用 XML 格式保存标签: {text}")
                logger.debug(f"边界框坐标: x={self.AT.x}, y={self.AT.y}, w={self.AT.w}, h={self.AT.h}")
                result, file_path, size = xml_message(self.save_path, self.image_name, self.img_width, self.img_height,
                                                    text, self.AT.x, self.AT.y, self.AT.w, self.AT.h)
                self.labels.append(result)
                self.clicked_save.append([self.AT.x, self.AT.y, (self.AT.w + self.AT.x), (self.AT.h + self.AT.y)])
                self.list_labels.append(text)  # 添加到列表标签中
                logger.debug(f"保存 XML 标签到: {file_path}, 大小: {size}")
                xml(self.image_path, file_path, size, self.labels)
                logger.info(f"XML 标签已保存: {file_path}")
            else:
                # YOLO 格式保存
                logger.debug(f"使用 YOLO 格式保存标签: {text}")
                logger.debug(f"边界框坐标: x={self.AT.x}, y={self.AT.y}, w={self.AT.w}, h={self.AT.h}")
                result, file_path, size = yolo_message(self.save_path, self.image_name, self.img_width, self.img_height,
                                               text, self.AT.x, self.AT.y, self.AT.w, self.AT.h)
                
                # 更新类别映射
                if text not in self.class_map:
                    self.class_map[text] = len(self.class_map)
                    logger.debug(f"添加新类别到映射: {text} -> {self.class_map[text]}")
                
                # 添加到标签列表和显示列表
                self.labels.append(result)
                self.clicked_save.append([self.AT.x, self.AT.y, (self.AT.w + self.AT.x), (self.AT.h + self.AT.y)])
                self.list_labels.append(text)  # 确保添加到列表标签中
                
                # 保存类别映射文件到标签目录
                classes_path = os.path.join(labels_dir, "classes.txt")
                sorted_classes = sorted(self.class_map.items(), key=lambda x: x[1])
                with open(classes_path, 'w') as f:
                    for class_name, _ in sorted_classes:
                        f.write(f"{class_name}\n")
                logger.info(f"类别映射已保存到: {classes_path}")
                
                # 创建标签文件路径
                label_file_path = os.path.join(train_labels_dir, f"{self.image_name}.txt")
                logger.debug(f"保存 YOLO 标签到: {label_file_path}, 大小: {size}, 类别映射: {self.class_map}")
                
                # 直接写入标签文件
                with open(label_file_path, 'w') as f:
                    class_id = self.class_map[text]
                    
                    # 获取边界框坐标
                    x, y = self.AT.x, self.AT.y
                    x2, y2 = self.AT.x + self.AT.w, self.AT.y + self.AT.h
                    
                    # 转换为 YOLO 格式
                    x_center, y_center, width, height = convert_to_yolo_format(self.img_width, self.img_height, x, y, x2, y2)
                    
                    # 写入文件
                    line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                    f.write(line)
                    logger.debug(f"写入行: {line.strip()}")
                
                logger.info(f"YOLO 标签已保存到: {label_file_path}")
            
            self.ui.label_4.mousePressEvent = self.mouse_press_event
            self.ui.label_4.setCursor(Qt.ArrowCursor)
            
        self.clicked_event = False
        self.paint_event = False
        logger.debug("重置点击和绘制事件状态")

        # 强制更新显示
        self.Show_Exists()
        logger.debug("显示已存在的标签")

    def video_on_dialog_confirmed(self, text):
        self.text = text
        logger.info(f"视频对话框确认，标签文本: {text}")
        if not self.save_path:
            logger.warning("未设置保存路径")
            upWindowsh("请选择保存路径")
        elif text and self.clicked_event:
            logger.debug(f"添加标签到列表: {text}")
            self.ui.listWidget.addItem(text)
            
            # 确保 list_labels 已初始化
            if not hasattr(self, 'list_labels'):
                logger.debug("初始化 list_labels 列表")
                self.list_labels = []
            
            # 确保 labels 已初始化
            if not hasattr(self, 'labels'):
                logger.debug("初始化 labels 列表")
                self.labels = []
            
            # 确保YOLO目录结构存在
            images_dir = os.path.join(self.save_path, "images")
            labels_dir = os.path.join(self.save_path, "labels")
            train_images_dir = os.path.join(images_dir, "train")
            train_labels_dir = os.path.join(labels_dir, "train")
            
            os.makedirs(train_images_dir, exist_ok=True)
            os.makedirs(train_labels_dir, exist_ok=True)
            logger.debug(f"确保YOLO目录结构存在: {train_images_dir}, {train_labels_dir}")
            
            # 复制当前图像到训练图像目录
            train_image_path = os.path.join(train_images_dir, f"{self.image_name}.jpg")
            if not os.path.exists(train_image_path) and self.image_path:
                import shutil
                shutil.copy2(self.image_path, train_image_path)
                logger.debug(f"复制图像到训练目录: {train_image_path}")
            
            # 根据选择的格式保存标签
            if self.format_type.upper() == "XML":
                # XML 格式保存
                logger.debug(f"使用 XML 格式保存标签: {text}")
                logger.debug(f"边界框坐标: x={self.AT.x}, y={self.AT.y}, w={self.AT.w}, h={self.AT.h}")
                result, file_path, size = xml_message(self.save_path, self.image_name, self.img_width, self.img_height,
                                                    text, self.AT.x, self.AT.y, self.AT.w, self.AT.h)
                self.labels.append(result)
                self.clicked_save.append([self.AT.x, self.AT.y, (self.AT.w + self.AT.x), (self.AT.h + self.AT.y)])
                self.list_labels.append(text)  # 添加到列表标签中
                logger.debug(f"保存 XML 标签到: {file_path}, 大小: {size}")
                xml(self.image_path, file_path, size, self.labels)
                logger.info(f"XML 标签已保存: {file_path}")
            else:
                # YOLO 格式保存
                logger.debug(f"使用 YOLO 格式保存标签: {text}")
                logger.debug(f"边界框坐标: x={self.AT.x}, y={self.AT.y}, w={self.AT.w}, h={self.AT.h}")
                result, file_path, size = yolo_message(self.save_path, self.image_name, self.img_width, self.img_height,
                                               text, self.AT.x, self.AT.y, self.AT.w, self.AT.h)
                
                # 更新类别映射
                if text not in self.class_map:
                    self.class_map[text] = len(self.class_map)
                    logger.debug(f"添加新类别到映射: {text} -> {self.class_map[text]}")
                
                # 添加到标签列表和显示列表
                self.labels.append(result)
                self.clicked_save.append([self.AT.x, self.AT.y, (self.AT.w + self.AT.x), (self.AT.h + self.AT.y)])
                self.list_labels.append(text)  # 确保添加到列表标签中
                
                # 保存类别映射文件到标签目录
                classes_path = os.path.join(labels_dir, "classes.txt")
                sorted_classes = sorted(self.class_map.items(), key=lambda x: x[1])
                with open(classes_path, 'w') as f:
                    for class_name, _ in sorted_classes:
                        f.write(f"{class_name}\n")
                logger.info(f"类别映射已保存到: {classes_path}")
                
                # 创建标签文件路径
                label_file_path = os.path.join(train_labels_dir, f"{self.image_name}.txt")
                logger.debug(f"保存 YOLO 标签到: {label_file_path}, 大小: {size}, 类别映射: {self.class_map}")
                
                # 直接写入标签文件
                with open(label_file_path, 'w') as f:
                    class_id = self.class_map[text]
                    
                    # 获取边界框坐标
                    x, y = self.AT.x, self.AT.y
                    x2, y2 = self.AT.x + self.AT.w, self.AT.y + self.AT.h
                    
                    # 转换为 YOLO 格式
                    x_center, y_center, width, height = convert_to_yolo_format(self.img_width, self.img_height, x, y, x2, y2)
                    
                    # 写入文件
                    line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                    f.write(line)
                    logger.debug(f"写入行: {line.strip()}")
                
                logger.info(f"YOLO 标签已保存到: {label_file_path}")
            
            # 启用"开始检测打标"按钮
            self.ui.pushButton_start_marking.setEnabled(True)
            logger.debug("启用开始检测打标按钮")
            
        self.clicked_event = False
        self.paint_event = False
        logger.debug("重置点击和绘制事件状态")

        # 强制更新显示
        self.Show_Exists()
        logger.debug("显示已存在的标签")

    # 显示已存在框
    def Show_Exists(self):
        logger.debug("开始显示已存在的标签和边界框")
        try:
            image = cv2.imread(self.img_path)
            if image is None:
                logger.error(f"无法读取图像: {self.img_path}")
                return
                
            if self.clicked_save == [] and self.paint_save == []:
                logger.debug("没有标签需要显示，显示原始图像")
                self.show_qt(self.img_path)
            else:
                logger.debug(f"显示 {len(self.clicked_save)} 个点击标签和 {len(self.paint_save)} 个绘制标签")
                
                # 显示点击保存的标签
                if self.clicked_save:
                    for i, box in enumerate(self.clicked_save):
                        logger.debug(f"绘制点击标签 {i+1}: ({box[0]}, {box[1]}) - ({box[2]}, {box[3]})")
                        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
                # 显示手动绘制的标签
                if self.paint_save:
                    for i, box in enumerate(self.paint_save):
                        logger.debug(f"绘制手动标签 {i+1}: ({box[0]}, {box[1]}) - ({box[2]}, {box[3]})")
                        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

                # 转换图像并显示
                h, w, channels = image.shape
                bytes_per_line = channels * w
                q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

                Qt_Gui = QtGui.QPixmap(q_image)
                self.ui.label_3.setFixedSize(self.img_width, self.img_height)
                self.ui.label_3.setPixmap(Qt_Gui)
                logger.debug("标签显示完成")
        except Exception as e:
            logger.error(f"显示标签时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())

# ##################################################################################################
    # 手动打标
    def mousePaint(self):
        if self.img_path != None:
            self.paint_event = True
            self.clicked_event = False
            if self.save:
                self.ui.label_4.mousePressEvent = self.mousePressEvent
                self.ui.label_4.mouseMoveEvent = self.mouseMoveEvent
                self.ui.label_4.mouseReleaseEvent = self.mouseReleaseEvent
                self.ui.label_4.paintEvent = self.paintEvent
                self.ui.label_4.setCursor(Qt.CrossCursor)
                self.save = False
            else:
                upWindowsh("请先输入标签")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.flag = True
            self.show_qt(self.img_path)
            self.x0, self.y0 = event.pos().x(), event.pos().y()
            self.x1, self.y1 = self.x0, self.y0
            self.ui.label_4.update()

    def mouseReleaseEvent(self, event):
        if self.flag:
            self.saveAndUpdate()
        self.flag = False
        self.save = False

    def mouseMoveEvent(self, event):
        if self.flag:
            self.x1, self.y1 = event.pos().x(), event.pos().y()
            self.ui.label_4.update()

    def paintEvent(self, event):
        super(MainFunc, self).paintEvent(event)
        if self.flag and self.x0 != 0 and self.y0 != 0 and self.x1 != 0 and self.y1 != 0:
            painter = QPainter(self.ui.label_4)
            painter.setPen(QPen(Qt.red, 4, Qt.SolidLine))
            painter.drawRect(QRect(self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0)))

    def saveAndUpdate(self):
        try:

            # 获取当前label上的QPixmap对象
            if self.ui.label_3.pixmap():
                pixmap = self.ui.label_3.pixmap()
                image = QImage(pixmap.size(), QImage.Format_ARGB32)
                painter = QPainter(image)

                # 绘制原始图像
                painter.drawPixmap(0, 0, pixmap)

                # 绘制矩形框
                if self.x0 != 0 and self.y0 != 0 and self.x1 != 0 and self.y1 != 0:
                    painter.setPen(QPen(Qt.red, 4, Qt.SolidLine))
                    painter.drawRect(QRect(self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0)))

                painter.end()
        except Exception as e:
            print(f"Error saving and updating image: {e}")

# ##################################################################################################
    # 获取视频
    def get_video(self):
        self.ui.listWidget.clear()  # 清空listWidget
        self.image_files = None
        self.img_path = None
        self.num = 0
        video_save_path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择图片保存文件夹")
        if video_save_path:
            self.video_save_path = video_save_path
        
        video_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 
            "选择视频", 
            "", 
            "Video Files (*.mp4 *.mpg)"
        )
        
        if video_path and video_save_path:
            self.video_path = video_path  # 保存视频路径以供重播使用
            self.cap = cv2.VideoCapture(video_path)
            # 获取视频总帧数
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # 设置滑块范围
            self.ui.horizontalSlider.setRange(0, self.total_frames)
            self.timer_camera.start(33)
            self.timer_camera.timeout.connect(self.OpenFrame)
            # 初始禁用重播按钮
            self.ui.pushButton_4.setEnabled(False)
        
            self.Change_Enable(method="ShowVideo", state=True)
            self.Change_Enable(method="MakeTag", state=False)
            # 禁用开始检测打标按钮
            self.ui.pushButton_start_marking.setEnabled(False)
        else:
            upWindowsh("请先选择视频和保存路径")


    def OpenFrame(self):
        if not self.sld_video_pressed:  # 只在未拖动时更新
            ret, image = self.cap.read()
            if ret:
                # 更新当前帧位置
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                # 更新滑块位置
                self.ui.horizontalSlider.setValue(self.current_frame)
                
                # 调整视频帧大小
                height, width = image.shape[:2]
                ratio = 1300 / width
                new_width = 1300
                new_height = int(height * ratio)
                
                if new_height > 850:
                    ratio = 850 / new_height
                    new_height = 850
                    new_width = int(new_width * ratio)
                
                # 调整图像大小
                image = cv2.resize(image, (new_width, new_height))
                
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    vedio_img = QImage(image.data, new_width, new_height, image.strides[0], QImage.Format_RGB888)
                elif len(image.shape) == 1:
                    vedio_img = QImage(image.data, new_width, new_height, QImage.Format_Indexed8)
                else:
                    vedio_img = QImage(image.data, new_width, new_height, image.strides[0], QImage.Format_RGB888)
                self.vedio_img = vedio_img
                
                # 调整label大小以适应新的图像尺寸
                self.ui.label_3.setFixedSize(new_width, new_height)
                self.ui.label_3.setPixmap(QPixmap(self.vedio_img))
                self.ui.label_3.setScaledContents(True)
            else:
                self.cap.release()
                self.timer_camera.stop()
                # 视频结束时启用重播按钮
                self.ui.pushButton_4.setEnabled(True)


    def Btn_Start(self):
        try:
            # 尝试断开之前的连接
            self.timer_camera.timeout.disconnect(self.OpenFrame)
        except TypeError:
            # 如果没有连接，直接忽略错误
            pass
        # 重新连接并启动定时器
        self.timer_camera.timeout.connect(self.OpenFrame)
        self.timer_camera.start(33)

    def Btn_Stop(self):
        self.timer_camera.stop()
        try:
            # 尝试断开定时器连接
            self.timer_camera.timeout.disconnect(self.OpenFrame)
        except TypeError:
            pass

    def Btn_Save(self):
        self.num += 1
        save_path = f'{self.video_save_path}/image{str(self.num)}.jpg'
        self.vedio_img.save(save_path)
        # 将保存信息添加到listWidget
        save_info = f'image{str(self.num)}.jpg保存成功！'
        self.ui.listWidget.addItem(save_info)
        print(f'{save_path}保存成功！')

    
    def moveSlider(self, position):
        """处理滑块移动"""
        if self.cap and self.total_frames > 0:
            # 设置视频帧位置
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            # 读取并显示新位置的帧
            ret, image = self.cap.read()
            if ret:
                # 调整视频帧大小
                height, width = image.shape[:2]
                ratio = 1300 / width
                new_width = 1300
                new_height = int(height * ratio)
                
                if new_height > 850:
                    ratio = 850 / new_height
                    new_height = 850
                    new_width = int(new_width * ratio)
                
                # 调整图像大小
                image = cv2.resize(image, (new_width, new_height))
                
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    vedio_img = QImage(image.data, new_width, new_height, image.strides[0], QImage.Format_RGB888)
                elif len(image.shape) == 1:
                    vedio_img = QImage(image.data, new_width, new_height, QImage.Format_Indexed8)
                else:
                    vedio_img = QImage(image.data, new_width, new_height, image.strides[0], QImage.Format_RGB888)
                self.vedio_img = vedio_img
                
                # 调整label大小以适应新的图像尺寸
                self.ui.label_3.setFixedSize(new_width, new_height)
                self.ui.label_3.setPixmap(QPixmap(self.vedio_img))
                self.ui.label_3.setScaledContents(True)

    def pressSlider(self):
        self.sld_video_pressed = True
        self.timer_camera.stop()  # 暂停视频播放

    def releaseSlider(self):
        self.sld_video_pressed = False
        try:
            # 尝试断开之前的连接
            self.timer_camera.timeout.disconnect(self.OpenFrame)
        except TypeError:
            pass
        # 重新连接并启动定时器
        self.timer_camera.timeout.connect(self.OpenFrame)
        self.timer_camera.start(33)

    def clean_up(self):
        file_path = 'GUI/history.txt'
        if os.path.exists(file_path):
            os.remove(file_path)

    def Btn_Replay(self):
        """重新播放视频"""
        if hasattr(self, 'video_path'):
            # 重新打开视频文件
            self.cap = cv2.VideoCapture(self.video_path)
            # 重置滑块位置
            self.ui.horizontalSlider.setValue(0)
            # 开始播放
            self.timer_camera.start(33)                 
            # 禁用重播按钮
            self.ui.pushButton_4.setEnabled(False)

    def Btn_Auto(self):
        if self.video_path and self.video_save_path:
            output_dir,saved_count = self.AVT.extract_frames_from_video(self.video_path, self.video_save_path, fps=2)
            content = f"已从视频中提取 {saved_count} 帧\n保存至 {output_dir}"
            print(content)
            self.ui.listWidget.addItem(content)
        else:
            upWindowsh("请先选择视频和保存路径")

    def video_marking(self):
        self.directory = None
        video_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.mpg)")
        self.video_path = video_path

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "选择图片保存文件夹")
        self.output_dir = output_dir
        self.ui.listWidget.clear()
        if self.video_path and self.output_dir:
            logger.info(f"选择视频: {self.video_path}, 输出目录: {self.output_dir}")
            self.Change_Enable(method="MakeTag",state=False)
            self.Change_Enable(method="ShowVideo",state=False)
            if self.cap:
                self.cap.release()
                self.timer_camera.stop()

            # 创建符合YOLO训练要求的目录结构
            self.images_dir = os.path.join(self.output_dir, "images")
            self.labels_dir = os.path.join(self.output_dir, "labels")
            self.train_images_dir = os.path.join(self.images_dir, "train")
            self.train_labels_dir = os.path.join(self.labels_dir, "train")
            
            # 创建目录
            os.makedirs(self.train_images_dir, exist_ok=True)
            os.makedirs(self.train_labels_dir, exist_ok=True)
            logger.info(f"创建YOLO训练目录结构: {self.images_dir}, {self.labels_dir}")
            
            # 检查输出目录中是否已有图像文件
            existing_images = [f for f in os.listdir(self.output_dir) if f.endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(self.output_dir) else []
            
            # 如果没有现有图像，只提取第一帧到输出目录根目录
            if not existing_images:
                logger.debug("输出目录中没有图像，提取视频第一帧")
                cap = cv2.VideoCapture(video_path)
                ret, frame = cap.read()
                if ret:
                    logger.debug("读取视频第一帧成功")
                    first_frame_path = os.path.join(self.output_dir, "0.jpg")
                    cv2.imwrite(first_frame_path, frame)
                    logger.info(f"保存第一帧到: {first_frame_path}")
                cap.release()
            else:
                logger.info(f"输出目录中已有 {len(existing_images)} 个图像文件，将使用现有图像")
            
            # 设置工作目录为输出目录根目录
            self.working_dir = self.output_dir
            
            # 重新初始化 AVT 对象，确保状态干净
            try:
                logger.info("初始化 AVT 对象")
                from sampro.LabelVideo_TW import AnythingVideo_TW
                self.AVT = AnythingVideo_TW()
                logger.debug("AVT 对象初始化完成")
                
                # 设置视频目录
                try:
                    logger.debug(f"设置 AVT 的视频目录: {self.working_dir}")
                    frame = self.AVT.set_video(self.working_dir)
                    logger.debug("AVT 视频目录设置成功")
                    
                    # 检查是否成功设置了视频目录
                    if not hasattr(self.AVT, 'frame_paths') or not self.AVT.frame_paths:
                        logger.error("设置视频目录失败，没有找到视频帧")
                        upWindowsh("设置视频目录失败，没有找到视频帧")
                        return
                    
                    logger.debug(f"找到 {len(self.AVT.frame_paths)} 个视频帧")
                except Exception as e:
                    logger.error(f"设置视频目录出错: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    upWindowsh("设置视频目录出错，请重试")
                    return
                
                # 初始化推理状态
                try:
                    logger.debug(f"初始化 AVT 的推理状态: {self.working_dir}")
                    self.AVT.inference(self.working_dir)
                    logger.debug("AVT 推理状态初始化成功")
                except Exception as e:
                    logger.error(f"初始化推理状态出错: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    upWindowsh("初始化推理状态出错，请重试")
                    return
            except Exception as e:
                logger.error(f"初始化 AVT 对象出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
                upWindowsh("初始化 AVT 对象出错，请重试")
                return
            
            if self.working_dir:
                self.image_files = list_images_in_directory(self.working_dir)
                if self.image_files:
                    self.image_path = self.image_files[0]
                    logger.debug(f"设置图像路径: {self.image_path}")
                    self.img_path = self.image_path
                    self.image_name = os.path.basename(self.image_path).split('.')[0]
                    logger.debug(f"图像名称: {self.image_name}")

                    self.img_path, self.img_width, self.img_height = Change_image_Size(self.img_path)
                    logger.debug(f"调整图像大小: {self.img_path}, {self.img_width}x{self.img_height}")
                    self.image = cv2.imread(self.img_path)
                    
                    # 设置 AT 的图像
                    self.AT.Set_Image(self.image)
                    logger.debug("设置 AT 的图像")
                    
                    # 转换为QPixmap并显示
                    Qt_Gui = QtGui.QPixmap(self.img_path)
                    # 设置label大小为图片原始大小
                    self.ui.label_3.setFixedSize(self.img_width, self.img_height)
                    self.ui.label_3.setPixmap(Qt_Gui)
                    logger.debug("显示图像完成")
                    
                    # 启用开始检测打标按钮
                    self.ui.pushButton_start_marking.setEnabled(True)
                    logger.debug("启用开始检测打标按钮")
                else:
                    logger.error("没有找到图像文件")
                    upWindowsh("没有找到图像文件，请重试")
                    return
                
            # 鼠标点击触发
            self.ui.label_4.mousePressEvent = self.mouse_press_event
            logger.debug("设置鼠标点击事件")
            
            # 显示提示信息
            self.ui.statusbar.showMessage("请点击目标对象，然后按 S 键保存标签")
            self.ui.listWidget.addItem("请点击目标对象，然后按 S 键保存标签")
        else:
            upWindowsh("请先选择视频和保存路径")
            self.ui.pushButton_start_marking.setEnabled(True)

    def on_video_processing_complete(self):
        logger.info("视频处理完成")
        try:
            self.worker_thread.deleteLater()
            self.xml_messages = self.worker_thread.xml_messages
            logger.debug(f"收到 {len(self.xml_messages)} 个标签消息")
            
            # 显示处理结果
            if self.xml_messages:
                self.ui.listWidget.addItem(f"视频处理完成，生成了 {len(self.xml_messages)} 个标签")
                
                # 显示标签保存路径
                if self.format_type.upper() == "YOLO":
                    train_labels_dir = os.path.join(self.output_dir, "labels", "train")
                    self.ui.listWidget.addItem(f"YOLO 标签已保存到: {train_labels_dir}")
                    
                    # 显示类别映射
                    classes_path = os.path.join(self.output_dir, "labels", "classes.txt")
                    if os.path.exists(classes_path):
                        self.ui.listWidget.addItem(f"类别映射已保存到: {classes_path}")
                        try:
                            with open(classes_path, 'r') as f:
                                classes = f.read().splitlines()
                            self.ui.listWidget.addItem(f"类别映射: {', '.join(classes)}")
                        except Exception as e:
                            logger.error(f"读取类别映射文件出错: {e}")
                else:
                    self.ui.listWidget.addItem(f"XML 标签已保存到: {self.save_path}")
                
                # 显示处理后的图像路径
                mask_dir = os.path.join(self.output_dir, "mask")
                if os.path.exists(mask_dir):
                    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                    if mask_files:
                        self.ui.listWidget.addItem(f"处理后的图像已保存到: {mask_dir}")
                        self.ui.listWidget.addItem(f"共 {len(mask_files)} 个处理后的图像")
                
                # 显示训练图像路径
                train_images_dir = os.path.join(self.output_dir, "images", "train")
                if os.path.exists(train_images_dir):
                    train_files = [f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                    if train_files:
                        self.ui.listWidget.addItem(f"训练图像已保存到: {train_images_dir}")
                        self.ui.listWidget.addItem(f"共 {len(train_files)} 个训练图像")
                
                # 显示视频帧总数
                frame_files = [f for f in os.listdir(self.output_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                if frame_files:
                    self.ui.listWidget.addItem(f"视频总帧数: {len(frame_files)}")
                    self.ui.listWidget.addItem(f"标签覆盖率: {len(self.xml_messages) / len(frame_files) * 100:.2f}%")
            else:
                self.ui.listWidget.addItem("视频处理完成，但未生成标签")
                self.ui.listWidget.addItem("请检查日志文件以获取更多信息")
                
                # 检查可能的错误原因
                if not hasattr(self.AVT, 'video_segments') or not self.AVT.video_segments:
                    self.ui.listWidget.addItem("错误: 未能收集视频分割结果")
                    self.ui.listWidget.addItem("可能原因: 目标跟踪失败或视频帧提取失败")
                    self.ui.listWidget.addItem("建议: 尝试重新点击目标，或选择更清晰的视频")
                
                # 检查日志文件中的错误
                log_dir = "logs"
                if os.path.exists(log_dir):
                    log_files = sorted([f for f in os.listdir(log_dir) if f.startswith("labelquick_") and f.endswith(".log")], 
                                      key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), 
                                      reverse=True)
                    if log_files:
                        self.ui.listWidget.addItem(f"最新日志文件: {os.path.join(log_dir, log_files[0])}")
            
            # 如果是 XML 格式，需要单独处理每个帧的标签
            if self.format_type.upper() == "XML":
                # 遍历输出目录中的图片
                img_files = [f for f in os.listdir(self.output_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                logger.debug(f"输出目录中有 {len(img_files)} 个图像文件")
                
                for img_file in img_files:
                    # 获取不带扩展名的文件名
                    img_name = os.path.splitext(img_file)[0]
                    img_file_path = os.path.join(self.output_dir, img_file)
                    logger.debug(f"处理图像文件: {img_file_path}")
                    
                    # 在xml_messages中查找对应的消息
                    matched_messages = []
                    for msg in self.xml_messages:
                        try:
                            if len(msg) > 1:  # 确保msg有足够的元素
                                label_path = msg[1]  # 获取索引值为1的路径
                                label_filename = os.path.splitext(os.path.basename(label_path))[0]
                                
                                # 如果文件名匹配，则保存标签文件
                                if label_filename == img_name and self.save_path:
                                    matched_messages.append(msg)
                        except Exception as e:
                            logger.error(f"处理标签消息时出错: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                    
                    logger.debug(f"找到 {len(matched_messages)} 个匹配的标签消息")
                    
                    for msg in matched_messages:
                        try:
                            self.labels = []
                            result = msg[0]
                            file_path = msg[1]
                            size = msg[2]
                            self.labels.append(result)
                            
                            # 保存 XML 格式标签
                            logger.debug(f"保存 XML 标签到: {file_path}, 大小: {size}")
                            xml(img_file_path, file_path, size, self.labels)
                            logger.info(f"已保存 XML 标签: {file_path}")
                        except Exception as e:
                            logger.error(f"保存 XML 标签时出错: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
            
            # YOLO 格式标签已在 VideoProcessingThread 中处理完成
            
            self.ui.listWidget.addItem("检测打标完成！")
            self.ui.listWidget.addItem("您可以使用生成的标签进行模型训练")
            logger.info("检测打标过程全部完成")
        except Exception as e:
            logger.error(f"视频处理完成回调出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.ui.listWidget.addItem(f"处理出错: {e}")
        finally:
            # 重新启用按钮
            self.ui.actionOpen_Video.setEnabled(True)
            self.ui.actionChange_Save_Dir.setEnabled(True)
            self.ui.pushButton_5.setEnabled(True)
            self.ui.pushButton_start_marking.setEnabled(True)
            
            # 更新状态栏
            self.ui.statusbar.showMessage("视频处理完成")
            
            # 滚动到列表底部
            self.ui.listWidget.scrollToBottom()

    def onFormatChanged(self, text):
        """处理标签格式变更"""
        old_format = self.format_type
        self.format_type = text.lower()  # 转换为小写
        logger.info(f"标签格式已更改: {old_format} -> {self.format_type}")
        
        # 如果切换到 YOLO 格式，确保 class_map 已初始化
        if self.format_type.lower() == "yolo":
            logger.debug("检查 YOLO 格式的类别映射")
            if not self.class_map:
                logger.debug("类别映射为空，尝试从现有标签初始化")
                # 如果已有标签，将它们添加到 class_map
                for label in self.labels:
                    class_name = label['name']
                    if class_name not in self.class_map:
                        self.class_map[class_name] = len(self.class_map)
                        logger.debug(f"添加类别到映射: {class_name} -> {self.class_map[class_name]}")
                
                # 如果还是空的，尝试从 listWidget 中获取标签
                if not self.class_map:
                    for i in range(self.ui.listWidget.count()):
                        class_name = self.ui.listWidget.item(i).text()
                        if class_name and class_name not in self.class_map:
                            self.class_map[class_name] = len(self.class_map)
                            logger.debug(f"从列表中添加类别到映射: {class_name} -> {self.class_map[class_name]}")
            
            logger.debug(f"当前类别映射: {self.class_map}")
            
            # 如果有保存路径，尝试保存类别映射文件
            if hasattr(self, 'save_path') and self.save_path:
                try:
                    classes_path = os.path.join(self.save_path, "classes.txt")
                    # 按照索引排序类别
                    sorted_classes = sorted(self.class_map.items(), key=lambda x: x[1])
                    with open(classes_path, 'w') as f:
                        for class_name, _ in sorted_classes:
                            f.write(f"{class_name}\n")
                    logger.info(f"类别映射已保存到: {classes_path}")
                except Exception as e:
                    logger.error(f"保存类别映射出错: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
        
        # 如果有当前图像，尝试重新加载标签
        if hasattr(self, 'image_name') and self.image_name and self.save_path:
            logger.debug(f"尝试重新加载当前图像的标签: {self.image_name}")
            self.Exists_Labels_And_Boxs()
            
        # 更新界面提示
        if self.format_type.lower() == "yolo":
            self.ui.statusbar.showMessage("当前标签格式: YOLO (.txt)")
        else:
            self.ui.statusbar.showMessage("当前标签格式: XML (.xml)")

    def debug_check_labels(self):
        """调试函数，检查标签文件是否正确保存"""
        logger.info("开始调试检查标签文件")
        
        if not hasattr(self, 'save_path') or not self.save_path:
            logger.warning("未设置保存路径")
            self.ui.listWidget.addItem("错误: 未设置保存路径")
            return
            
        if not hasattr(self, 'image_name') or not self.image_name:
            logger.warning("未加载图像")
            self.ui.listWidget.addItem("错误: 未加载图像")
            return
            
        # 检查 XML 格式标签文件
        xml_path = os.path.join(self.save_path, f"{self.image_name}.xml")
        xml_exists = os.path.exists(xml_path)
        
        # 检查 YOLO 格式标签文件
        yolo_path = os.path.join(self.save_path, f"{self.image_name}.txt")
        yolo_exists = os.path.exists(yolo_path)
        
        # 检查类别映射文件
        classes_path = os.path.join(self.save_path, "classes.txt")
        classes_exists = os.path.exists(classes_path)
        
        # 显示调试信息
        self.ui.listWidget.addItem(f"当前格式: {self.format_type}")
        self.ui.listWidget.addItem(f"保存路径: {self.save_path}")
        self.ui.listWidget.addItem(f"图像名称: {self.image_name}")
        self.ui.listWidget.addItem(f"XML 标签文件存在: {xml_exists} ({xml_path})")
        self.ui.listWidget.addItem(f"YOLO 标签文件存在: {yolo_exists} ({yolo_path})")
        self.ui.listWidget.addItem(f"类别映射文件存在: {classes_exists} ({classes_path})")
        self.ui.listWidget.addItem(f"类别映射: {self.class_map}")
        self.ui.listWidget.addItem(f"标签列表: {self.list_labels}")
        self.ui.listWidget.addItem(f"点击保存: {len(self.clicked_save)} 个")
        self.ui.listWidget.addItem(f"绘制保存: {len(self.paint_save)} 个")
        
        # 如果 YOLO 标签文件存在，显示其内容
        if yolo_exists:
            try:
                with open(yolo_path, 'r') as f:
                    lines = f.readlines()
                self.ui.listWidget.addItem(f"YOLO 标签文件内容 ({len(lines)} 行):")
                for i, line in enumerate(lines):
                    self.ui.listWidget.addItem(f"  行 {i+1}: {line.strip()}")
            except Exception as e:
                self.ui.listWidget.addItem(f"读取 YOLO 标签文件出错: {e}")
                
        # 如果类别映射文件存在，显示其内容
        if classes_exists:
            try:
                with open(classes_path, 'r') as f:
                    lines = f.readlines()
                self.ui.listWidget.addItem(f"类别映射文件内容 ({len(lines)} 行):")
                for i, line in enumerate(lines):
                    self.ui.listWidget.addItem(f"  类别 {i}: {line.strip()}")
            except Exception as e:
                self.ui.listWidget.addItem(f"读取类别映射文件出错: {e}")
                
        logger.info("调试检查标签文件完成")

    def Btn_Start_Marking(self):
        logger.info("开始视频检测打标")
        # 禁用开始检测打标按钮
        self.ui.pushButton_start_marking.setEnabled(False)
        logger.debug("禁用开始检测打标按钮")
        
        if self.video_path and self.output_dir:
            logger.debug(f"视频路径: {self.video_path}, 输出目录: {self.output_dir}")
            
            # 清空列表窗口
            self.ui.listWidget.clear()
            self.ui.listWidget.addItem("开始视频目标跟踪标注...")
            
            # 确保 AVT 对象已初始化
            if not hasattr(self, 'AVT') or self.AVT is None:
                logger.error("AVT 对象未初始化，无法继续")
                upWindowsh("视频处理对象未初始化，请重新选择视频")
                self.ui.pushButton_start_marking.setEnabled(True)
                return
            
            # 确保已设置点击坐标
            if not hasattr(self, 'clicked_x') or not hasattr(self, 'clicked_y') or self.clicked_x is None or self.clicked_y is None:
                logger.warning("未设置点击坐标，请先点击目标对象")
                upWindowsh("请先点击目标对象，然后再开始检测打标")
                self.ui.pushButton_start_marking.setEnabled(True)
                return
            
            # 确保已设置文本
            if not hasattr(self, 'text') or not self.text:
                logger.warning("未设置标签文本，请先点击目标并按S键输入标签")
                upWindowsh("请先点击目标并按S键输入标签，然后再开始检测打标")
                self.ui.pushButton_start_marking.setEnabled(True)
                return
            
            # 确保推理状态已初始化
            if not hasattr(self.AVT, 'inference_state') or self.AVT.inference_state is None:
                try:
                    logger.debug(f"初始化 AVT 的推理状态: {self.output_dir}")
                    self.ui.listWidget.addItem("正在初始化推理状态...")
                    self.AVT.inference(self.output_dir)
                    logger.debug("AVT 推理状态初始化成功")
                    self.ui.listWidget.addItem("推理状态初始化成功")
                except Exception as e:
                    logger.error(f"初始化 AVT 推理状态出错: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    upWindowsh("初始化视频处理出错，请重试")
                    self.ui.pushButton_start_marking.setEnabled(True)
                    return
            
            # 确保已添加点击点
            try:
                logger.debug(f"设置 AVT 的点击坐标: ({self.clicked_x}, {self.clicked_y}), 方法: {self.method}")
                self.ui.listWidget.addItem(f"设置点击坐标: ({self.clicked_x}, {self.clicked_y})")
                self.AVT.Set_Clicked([self.clicked_x, self.clicked_y], self.method)
                logger.debug("AVT 点击坐标设置成功")
                
                # 添加点击点
                logger.debug("添加点击点到模型")
                self.ui.listWidget.addItem("添加点击点到模型...")
                if not self.AVT.add_new_points_or_box():
                    logger.error("添加点击点失败")
                    upWindowsh("添加点击点失败，请重新点击目标对象")
                    self.ui.pushButton_start_marking.setEnabled(True)
                    return
                logger.debug("点击点添加成功")
                self.ui.listWidget.addItem("点击点添加成功")
            except Exception as e:
                logger.error(f"设置 AVT 点击坐标出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
                upWindowsh("设置视频处理出错，请重试")
                self.ui.pushButton_start_marking.setEnabled(True)
                return
            
            # 创建并启动工作线程
            logger.debug(f"创建视频处理线程，点击坐标: ({self.clicked_x}, {self.clicked_y}), 方法: {self.method}, 文本: {self.text}")
            logger.debug(f"保存路径: {self.save_path}, 格式: {self.format_type}, 类别映射: {self.class_map}")
            
            # 确保类别映射中包含当前标签
            if self.text not in self.class_map:
                logger.debug(f"将标签 '{self.text}' 添加到类别映射")
                self.class_map[self.text] = len(self.class_map)
                logger.debug(f"更新后的类别映射: {self.class_map}")
            
            self.ui.listWidget.addItem(f"开始处理视频，标签: {self.text}")
            self.ui.listWidget.addItem("这可能需要一些时间，请耐心等待...")
            self.ui.listWidget.addItem("系统将自动跟踪目标并生成标签...")
            self.ui.listWidget.addItem("正在从视频提取所有帧...")
            self.ui.listWidget.addItem("提取完成后，系统将对每一帧进行处理...")
            
            self.worker_thread = VideoProcessingThread(
                self.AVT, 
                self.video_path, 
                self.output_dir,
                self.clicked_x, 
                self.clicked_y, 
                self.method,
                self.text,
                self.save_path,
                self.format_type,
                self.class_map
            )
            logger.debug("连接视频处理线程信号")
            self.worker_thread.finished.connect(self.on_video_processing_complete)
            self.worker_thread.frame_ready.connect(self.update_frame)
            logger.info("启动视频处理线程")
            self.worker_thread.start()
            
            # 禁用按钮，防止重复处理
            logger.debug("禁用其他按钮，防止重复处理")
            self.ui.actionOpen_Video.setEnabled(False)
            self.ui.actionChange_Save_Dir.setEnabled(False)
            self.ui.pushButton_5.setEnabled(False)  # 自动提取按钮
            
            # 显示提示信息
            self.ui.statusbar.showMessage("正在处理视频，请稍候...")
            self.ui.listWidget.addItem("正在处理视频，请稍候...")
        else:
            upWindowsh("请先选择视频和保存路径")
            self.ui.pushButton_start_marking.setEnabled(True)

    def update_frame(self, frame):
        """更新视频帧显示"""
        if frame is not None:
            logger.debug("接收到新的视频帧")
            # 转换为 QImage
            height, width = frame.shape[:2]
            logger.debug(f"帧尺寸: {width}x{height}")
            
            try:
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    q_image = QImage(frame.data, width, height, frame.strides[0], QImage.Format_RGB888)
                    logger.debug("转换 RGB 帧为 QImage")
                else:
                    q_image = QImage(frame.data, width, height, QImage.Format_Indexed8)
                    logger.debug("转换灰度帧为 QImage")
                    
                # 显示图像
                pixmap = QPixmap.fromImage(q_image)
                self.ui.label_3.setFixedSize(width, height)
                self.ui.label_3.setPixmap(pixmap)
                logger.debug("更新 UI 显示帧")
                
                # 更新进度信息
                self.ui.listWidget.addItem(f"正在处理帧...")
                # 滚动到最新项
                self.ui.listWidget.scrollToBottom()
                logger.debug("更新处理进度信息")
            except Exception as e:
                logger.error(f"更新帧显示出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning("接收到空帧")

