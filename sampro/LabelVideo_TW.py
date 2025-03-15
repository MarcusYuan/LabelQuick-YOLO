import sys
import os
from pathlib import Path
from util.xmlfile import xml_message
from util.yolofile import yolo_message
from PIL import Image
import numpy as np
import torch
import cv2
import logging
from sampro.sam2.build_sam import build_sam2_video_predictor

# 获取日志记录器
logger = logging.getLogger(__name__)

# 设置当前文件夹
sys.path.append(r'sampro')

class AnythingVideo_TW():
    def __init__(self):
        # SAM2 模型配置
        self.sam2_checkpoint = "sampro/checkpoints/sam2.1_hiera_large.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.video_path = ""
        self.output_path = ""
        self.predictor = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint, self.device)

        # 全局变量
        self.coords = []
        self.methods = []
        self.frame = None
        self.frame_paths = []  # 初始化 frame_paths 属性

        self.option = False
        self.clicked_x = None
        self.clicked_y = None
        self.method = None

        self.inference_state = None
        self.out_obj_ids = None
        self.out_mask_logits = None
        self.video_segments = {}

        # 矩形框
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0

    def set_video(self, video_dir):
        self.video_path = video_dir
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        
        # 修改排序逻辑，提取文件名中的数字部分进行排序
        def extract_number(filename):
            import re
            # 从文件名中提取数字部分
            numbers = re.findall(r'\d+', os.path.splitext(filename)[0])
            if numbers:
                return int(numbers[-1])  # 使用最后一个数字作为排序依据
            return 0  # 如果没有数字，返回0
        
        logger.debug(f"找到 {len(frame_names)} 个帧文件")
        try:
            # 尝试使用新的排序逻辑
            frame_names.sort(key=extract_number)
            logger.debug(f"使用数字提取排序成功")
        except Exception as e:
            # 如果新排序失败，使用普通字符串排序
            logger.error(f"数字提取排序失败: {e}")
            frame_names.sort()
            logger.debug(f"使用普通字符串排序")
        
        logger.debug(f"排序后的前5个文件: {frame_names[:5] if len(frame_names) >= 5 else frame_names}")
        
        # 设置 frame_paths 属性
        self.frame_paths = [os.path.join(video_dir, name) for name in frame_names]

        # 加载第一帧
        frame_idx = 5 if len(frame_names) > 5 else 0
        frame_name = frame_names[frame_idx]
        frame_path = os.path.join(video_dir, frame_name)
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame = frame
        return frame

    def inference(self, video_dir):
        self.inference_state = self.predictor.init_state(video_path=video_dir)
        self.predictor.reset_state(self.inference_state)

    def extract_frames_from_video(self, video_path, output_dir, fps=24):
        """
        从视频中提取帧并保存为图片
        Args:
            video_path: 输入视频的路径
            output_dir: 输出图片的文件夹路径
            fps: 每秒提取的帧数，默认为2
        Returns:
            output_dir: 保存帧的文件夹路径
        Raises:
            ValueError: 当fps超过24或视频文件无法打开时
        """
        # 检查fps是否超过限制
        if fps > 24:
            raise ValueError(f"fps不能超过24帧，当前设置为{fps}帧")
        
        # 确保输出目录存在
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频的基本信息
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps)  # 计算需要跳过的帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 打印视频信息
        # print(f"视频信息:")
        # print(f"- 原始帧率: {video_fps:.2f} fps")
        # print(f"- 目标帧率: {fps} fps")
        # print(f"- 总帧数: {total_frames}")
        # print(f"- 预计提取帧数: {total_frames // frame_interval}")
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 按指定间隔保存帧
            if frame_count % frame_interval == 0:
                # 转换为PIL Image以便调整大小
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # 获取并调整尺寸
                width, height = frame_pil.size
                ratio = 1300 / width
                width = 1300
                height = int(height * ratio)
                reduced_image = frame_pil.resize((width, height))
                
                # 如果高度超过850，进一步调整
                if height > 850:
                    ratio = 850 / height
                    height = 850
                    width = int(width * ratio)
                    reduced_image = reduced_image.resize((width, height))
                
                # 保存调整后的帧
                frame_path = output_dir / f"{saved_count}.jpg"
                reduced_image.save(str(frame_path))
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        # content = f"已从视频中提取 {saved_count} 帧，保存至 {output_dir}"
        return str(output_dir),saved_count
    
    # 设置点击位置
    def Set_Clicked(self, clicked, method):
        self.clicked_x, self.clicked_y = clicked
        self.method = method

    # 显示点击点
    def Draw_Point(self, image, label):
        if label == 1:
            cv2.circle(image, (self.clicked_x, self.clicked_y), 5, (255, 0, 0), -1)  # 蓝色点
        elif label == 0:
            cv2.circle(image, (self.clicked_x, self.clicked_y), 5, (0, 0, 255), -1)  # 红色点

    def add_new_points_or_box(self):
        try:
            ann_frame_idx = 0  # 当前帧的索引
            ann_obj_id = 1  # 默认目标 ID

            # 检查点击坐标是否有效
            if self.clicked_x is None or self.clicked_y is None:
                logger.error("点击坐标无效")
                return False

            logger.debug(f"添加新点: 坐标=({self.clicked_x}, {self.clicked_y}), 方法={self.method}")
            self.coords.append([self.clicked_x, self.clicked_y])
            self.methods.append(self.method)

            points = np.array(self.coords)
            labels = np.array(self.methods)

            logger.debug(f"点集: {points}, 标签: {labels}")
            
            # 调用预测器添加新点或框
            try:
                _, self.out_obj_ids, self.out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                )
                
                # 记录返回的掩码信息
                if self.out_mask_logits is not None and hasattr(self.out_mask_logits, 'shape'):
                    logger.debug(f"生成的掩码形状: {self.out_mask_logits.shape}")
                else:
                    logger.warning("生成的掩码无效")
                    
                self.option = True
                return True
            except Exception as e:
                logger.error(f"添加新点或框时出错: {e}")
                return False
        except Exception as e:
            logger.error(f"add_new_points_or_box 方法出错: {e}")
            return False
            
    def Create_Mask(self):
        """
        创建掩码并绘制边界框
        Returns:
            bool: 成功返回True，失败返回False
        """
        try:
            logger.info("创建掩码")
            
            # 检查是否已经设置了点击坐标
            if self.clicked_x is None or self.clicked_y is None:
                logger.error("点击坐标未设置")
                return False
                
            # 添加新点并生成掩码
            if not self.add_new_points_or_box():
                logger.error("添加新点失败")
                return False
                
            # 检查掩码是否生成
            if self.out_mask_logits is None:
                logger.error("掩码生成失败")
                return False
                
            # 绘制掩码
            try:
                mask = (self.out_mask_logits[0] > 0.0).cpu().numpy()
                logger.debug(f"掩码形状: {mask.shape}")
                
                # 确保掩码和帧尺寸匹配
                frame_height, frame_width = self.frame.shape[:2]
                logger.debug(f"帧尺寸: {frame_width}x{frame_height}")
                
                # 如果掩码是一维的，尝试重塑
                if len(mask.shape) == 1:
                    logger.warning(f"掩码是一维数组，形状为 {mask.shape}，尝试重塑")
                    try:
                        if mask.shape[0] == frame_width * frame_height:
                            mask = mask.reshape(frame_height, frame_width)
                        else:
                            logger.warning(f"掩码大小 ({mask.shape[0]}) 与帧像素数 ({frame_width * frame_height}) 不匹配，创建空掩码")
                            mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                    except Exception as e:
                        logger.error(f"重塑掩码时出错: {e}")
                        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                
                # 确保掩码是 uint8 类型
                if mask.dtype != np.uint8:
                    mask = (mask > 0).astype(np.uint8)
                
                # 调整掩码尺寸
                if mask.shape[0] != frame_height or mask.shape[1] != frame_width:
                    logger.warning(f"掩码尺寸 ({mask.shape[1]}x{mask.shape[0]}) 与帧尺寸 ({frame_width}x{frame_height}) 不匹配，进行调整")
                    try:
                        # 确保掩码是二维的
                        if len(mask.shape) > 2:
                            mask = mask.squeeze()
                            if len(mask.shape) > 2:  # 如果仍然不是二维
                                mask = mask[:, :, 0]  # 取第一个通道
                        
                        # 使用OpenCV调整大小
                        mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
                    except Exception as e:
                        logger.error(f"调整掩码尺寸时出错: {e}")
                        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                
                # 绘制掩码和边界框
                result = self.Draw_Mask(mask, self.frame.copy())
                if result is None:
                    logger.error("绘制掩码失败")
                    return False
                    
                logger.info("掩码创建成功")
                return True
            except Exception as e:
                logger.error(f"绘制掩码时出错: {e}")
                return False
        except Exception as e:
            logger.error(f"Create_Mask 方法出错: {e}")
            return False

    def Draw_Mask(self, mask, frame, obj_id=None):
        logger.info(f"绘制掩码，对象 ID: {obj_id}")
        # 转换 mask 为 NumPy 数组
        mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        
        # 获取帧的尺寸
        frame_height, frame_width = frame.shape[:2]
        logger.debug(f"帧尺寸: {frame_width}x{frame_height}")
        
        # 获取掩码的尺寸
        h, w = mask.shape[-2:] if len(mask.shape) > 2 else mask.shape
        logger.debug(f"原始掩码尺寸: {w}x{h}")
        
        # 确保掩码尺寸与帧尺寸匹配
        if h != frame_height or w != frame_width:
            logger.warning(f"掩码尺寸 ({w}x{h}) 与帧尺寸 ({frame_width}x{frame_height}) 不匹配，进行调整")
            # 调整掩码尺寸以匹配帧尺寸
            mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
            h, w = mask.shape[:2]
            logger.debug(f"调整后的掩码尺寸: {w}x{h}")
        
        # 确保掩码是 2D 的
        if len(mask.shape) > 2:
            mask = mask.reshape(h, w)
        
        # 创建一个彩色的 mask
        color_mask = np.zeros_like(frame, dtype=np.uint8)
        color_mask[:, :, 0] = 0    # 蓝色分量
        color_mask[:, :, 1] = 255  # 绿色分量
        color_mask[:, :, 2] = 0    # 红色分量

        # 将 mask 应用到彩色 mask 上
        mask = (mask > 0).astype(np.uint8)  # 二值化
        
        # 确保掩码是单通道的
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        
        # 确保掩码尺寸与帧尺寸匹配
        if mask.shape[0] != frame_height or mask.shape[1] != frame_width:
            logger.warning(f"二值化后的掩码尺寸 ({mask.shape[1]}x{mask.shape[0]}) 与帧尺寸 ({frame_width}x{frame_height}) 不匹配，进行调整")
            mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
        
        try:
            color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask)
            logger.debug("创建彩色掩码完成")
        except cv2.error as e:
            logger.error(f"应用掩码时出错: {e}")
            logger.error(f"掩码尺寸: {mask.shape}, 颜色掩码尺寸: {color_mask.shape}")
            # 如果出错，尝试使用另一种方法
            mask_3d = np.stack([mask, mask, mask], axis=2)
            color_mask = color_mask * mask_3d
            logger.debug("使用替代方法创建彩色掩码完成")

        # 叠加到原始帧上 (半透明)
        alpha = 0.5  # 透明度
        frame_with_mask = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)
        logger.debug("叠加掩码到原始帧完成")

        # 显示轮廓
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame_with_mask, contours, -1, (0, 255, 0), 2)  # 绿色轮廓
            logger.debug(f"找到 {len(contours)} 个轮廓")
        except cv2.error as e:
            logger.error(f"查找轮廓时出错: {e}")
            logger.error(f"掩码尺寸: {mask.shape}, 掩码类型: {mask.dtype}")
            # 如果出错，返回原始帧
            return None

        # 初始化最大面积和对应的最大轮廓
        img = frame_with_mask.copy()

        max_area = 0
        max_contour = None

        # 遍历每个轮廓
        for i, contour in enumerate(contours):
            # 计算轮廓的面积
            area = cv2.contourArea(contour)
            logger.debug(f"轮廓 {i+1} 面积: {area}")
            
            # 如果当前面积大于最大面积，则更新最大面积和对应的最大轮廓
            if area > max_area:
                max_area = area
                max_contour = contour
                logger.debug(f"更新最大轮廓: 索引={i+1}, 面积={area}")
                
        # 使用矩形框绘制最大轮廓
        if max_contour is not None:
            self.x, self.y, self.w, self.h = cv2.boundingRect(max_contour)
            logger.debug(f"最大轮廓边界框: x={self.x}, y={self.y}, w={self.w}, h={self.h}")
            cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 2)
            # 在原图上绘制边缘线
            cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
            self.image_mask = img
            logger.debug(f"绘制边界框和轮廓完成")
        else:
            logger.warning("未找到有效轮廓")
            return None
            
        return img, self.x, self.y, self.w, self.h

    def Draw_Mask_Video(self, output_video_path="segmented_output.mp4"):
        # 收集所有帧的分割结果
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # 获取所有帧的名称
        frame_names = [
            p for p in os.listdir(self.video_path)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        # 初始化视频写入器
        first_frame_path = os.path.join(self.video_path, frame_names[0])
        first_frame = cv2.imread(first_frame_path)
        height, width, _ = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))

        # 遍历所有帧，处理分割结果并保存
        for out_frame_idx, frame_name in enumerate(frame_names):
            # 读取当前帧
            frame_path = os.path.join(self.video_path, frame_name)
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


            for out_obj_id, out_mask in self.video_segments[out_frame_idx].items():
                frame = self.Draw_Mask(out_mask, frame, out_obj_id)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    

                video_writer.write(frame)

        # 释放视频写入器
        video_writer.release()
        print(f"分割结果视频已保存到: {output_video_path}")


    def Draw_Mask_picture(self,frame_stride):
        # 1. 收集所有帧的分割结果
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # 2. 获取所有帧的名称
        frame_names = [
            p for p in os.listdir(self.video_path)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        # 3. 每几帧显示一次结果
        vis_frame_stride = frame_stride
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            # 读取当前帧
            frame_path = os.path.join(self.video_path, frame_names[out_frame_idx])
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 如果该帧有分割结果，则绘制
            if out_frame_idx in self.video_segments:
                for out_obj_id, out_mask in self.video_segments[out_frame_idx].items():
                    frame = self.Draw_Mask(out_mask, frame, out_obj_id)
            
            # 显示结果
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转换回BGR用于显示
            # cv2.imshow(f"Frame {out_frame_idx}", frame)
            # cv2.waitKey(1)  # 添加短暂延迟



    def Draw_Mask_at_frame(self, start_frame=0, return_frames=False, save_image_path=None, save_path=None, text=None, format_type="XML"):
        """
        遍历所有帧并绘制轮廓
        Args:
            start_frame (int): 起始帧序号
            return_frames (bool): 是否返回处理后的帧列表
            save_image_path (str): 保存图像的路径
            save_path (str): 保存标签的路径
            text (str): 标签文本说明
            format_type (str): 标签格式，可选 "XML" 或 "YOLO"
        Returns:
            tuple: (processed_frames, xml_messages) - processed_frames 在 return_frames=False 时为 None
        """
        logger.info(f"开始绘制轮廓，格式: {format_type}, 文本: {text}")
        logger.debug(f"起始帧: {start_frame}, 返回帧: {return_frames}, 保存图像路径: {save_image_path}, 保存标签路径: {save_path}")
        
        # 1. 收集所有帧的分割结果
        logger.debug("开始在视频中传播分割结果")
        try:
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
                try:
                    # 检查掩码数据的形状和类型
                    if hasattr(out_mask_logits, 'shape'):
                        logger.debug(f"帧 {out_frame_idx} 的掩码形状: {out_mask_logits.shape}")
                    else:
                        logger.warning(f"帧 {out_frame_idx} 的掩码没有shape属性")
                    
                    # 确保掩码是有效的数据
                    if out_mask_logits is None or len(out_obj_ids) == 0:
                        logger.warning(f"帧 {out_frame_idx} 没有有效的掩码数据")
                        continue
                    
                    # 处理掩码数据
                    self.video_segments[out_frame_idx] = {}
                    for i, out_obj_id in enumerate(out_obj_ids):
                        try:
                            # 获取当前对象的掩码
                            mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                            
                            # 记录掩码形状
                            logger.debug(f"帧 {out_frame_idx}, 对象 {out_obj_id} 的掩码形状: {mask.shape}")
                            
                            # 存储掩码
                            self.video_segments[out_frame_idx][out_obj_id] = mask
                        except Exception as e:
                            logger.error(f"处理帧 {out_frame_idx}, 对象 {out_obj_id} 的掩码时出错: {e}")
                    
                    logger.debug(f"帧 {out_frame_idx} 的分割结果: 对象 IDs {out_obj_ids}")
                except Exception as e:
                    logger.error(f"处理帧 {out_frame_idx} 的分割结果时出错: {e}")
                    continue
        except Exception as e:
            logger.error(f"在视频中传播分割结果时出错: {e}")
            # 确保 video_segments 至少有一个空条目
            if not self.video_segments:
                logger.warning("创建空的视频分割结果")
                self.video_segments[0] = {}

        # 2. 获取所有帧的名称
        frame_names = [
            p for p in os.listdir(self.video_path)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        logger.debug(f"找到 {len(frame_names)} 个帧")

        processed_frames = [] if return_frames else None
        result = None
        file_path = None
        size = None

        # 遍历所有帧
        xml_messages = []
        logger.info(f"开始处理 {len(frame_names) - start_frame} 个帧")
        for frame_idx in range(start_frame, len(frame_names)):
            frame_path = os.path.join(self.video_path, frame_names[frame_idx])
            logger.debug(f"处理帧 {frame_idx}: {frame_path}")
            frame = cv2.imread(frame_path)
            if frame is None:
                logger.error(f"无法读取帧 {frame_idx}: {frame_path}")
                continue
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_height, frame_width = frame.shape[:2]
            logger.debug(f"帧 {frame_idx} 尺寸: {frame_width}x{frame_height}")
            
            video_label = []
            
            if frame_idx in self.video_segments:
                logger.debug(f"帧 {frame_idx} 有 {len(self.video_segments[frame_idx])} 个分割对象")
                for out_obj_id, out_mask in self.video_segments[frame_idx].items():
                    logger.debug(f"处理帧 {frame_idx} 的对象 {out_obj_id}")
                    
                    # 检查掩码尺寸
                    mask_shape = out_mask.shape
                    logger.debug(f"原始掩码尺寸: {mask_shape}")
                    
                    # 确保掩码是有效的数据类型和形状
                    if len(mask_shape) == 1:
                        logger.warning(f"掩码是一维数组，形状为 {mask_shape}，尝试重塑")
                        try:
                            # 尝试将一维掩码重塑为二维
                            if mask_shape[0] == frame_width * frame_height:
                                # 如果掩码大小与帧像素数匹配，直接重塑
                                out_mask = out_mask.reshape(frame_height, frame_width)
                            else:
                                # 否则创建一个空掩码
                                logger.warning(f"掩码大小 ({mask_shape[0]}) 与帧像素数 ({frame_width * frame_height}) 不匹配，创建空掩码")
                                out_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                            logger.debug(f"重塑后的掩码尺寸: {out_mask.shape}")
                        except Exception as e:
                            logger.error(f"重塑掩码时出错: {e}")
                            # 创建一个空掩码
                            out_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                            logger.debug("创建了空掩码")
                    
                    # 确保掩码是 uint8 类型
                    if out_mask.dtype != np.uint8:
                        logger.debug(f"转换掩码类型从 {out_mask.dtype} 到 uint8")
                        out_mask = (out_mask > 0).astype(np.uint8)
                    
                    # 确保掩码尺寸与帧尺寸匹配
                    if out_mask.shape[0] != frame_height or out_mask.shape[1] != frame_width:
                        logger.warning(f"掩码尺寸 ({out_mask.shape[1]}x{out_mask.shape[0]}) 与帧尺寸 ({frame_width}x{frame_height}) 不匹配，进行调整")
                        try:
                            # 确保掩码是二维的
                            if len(out_mask.shape) > 2:
                                out_mask = out_mask.squeeze()
                                if len(out_mask.shape) > 2:  # 如果仍然不是二维
                                    out_mask = out_mask[:, :, 0]  # 取第一个通道
                            
                            # 使用OpenCV调整大小
                            out_mask = cv2.resize(out_mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
                            logger.debug(f"调整后的掩码尺寸: {out_mask.shape}")
                        except Exception as e:
                            logger.error(f"调整掩码尺寸时出错: {e}")
                            # 创建一个空掩码
                            out_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                            logger.debug("创建了空掩码")
                    
                    # 检查 Draw_Mask 的返回值
                    try:
                        result = self.Draw_Mask(out_mask, frame.copy(), out_obj_id)
                        if isinstance(result, tuple) and len(result) == 5:
                            frame, x, y, w, h = result
                            video_label.append([x, y, w, h])
                            logger.debug(f"绘制掩码成功，边界框: x={x}, y={y}, w={w}, h={h}")
                        else:
                            logger.warning(f"Warning: Draw_Mask returned unexpected format at frame {frame_idx}")
                            continue
                    except Exception as e:
                        logger.error(f"绘制掩码时出错: {e}")
                        continue
            
                    # 确保 frame 是有效的图像数组
                    if frame is not None and isinstance(frame, np.ndarray):
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        if return_frames:
                            processed_frames.append(frame)
                            logger.debug(f"添加处理后的帧到返回列表")
                            
                        if save_image_path:
                            logger.debug(f"保存处理后的帧图像")
                            try:
                                # 保存前调整图片大小
                                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                
                                # 获取照片大小
                                width, height = frame_pil.size
                                ratio = 1300 / width
                                width = 1300
                                height *= ratio
                                reduced_image = frame_pil.resize((int(width), int(height)))
                                
                                if height > 850:
                                    ratio = 850 / height
                                    height = 850
                                    width *= ratio
                                    reduced_image = frame_pil.resize((int(width), int(height)))
                                
                                # 保存调整后的图片
                                save_path_frame = f"{save_image_path}/{frame_idx}.jpg"
                                reduced_image.save(save_path_frame)
                                logger.debug(f"保存调整后的图像到: {save_path_frame}, 大小: {width}x{height}")
                                
                                # 根据格式类型创建标签信息
                                if format_type.upper() == "XML":
                                    # XML 格式
                                    logger.debug(f"创建 XML 格式标签，文本: {text}")
                                    result, file_path, size = xml_message(
                                        save_path, frame_idx, int(width), int(height),
                                        text, self.x, self.y, self.w, self.h
                                    )
                                    logger.debug(f"XML 标签信息: 路径={file_path}, 大小={size}")
                                else:
                                    # YOLO 格式
                                    logger.debug(f"创建 YOLO 格式标签，文本: {text}")
                                    result, file_path, size = yolo_message(
                                        save_path, frame_idx, int(width), int(height),
                                        text, self.x, self.y, self.w, self.h
                                    )
                                    logger.debug(f"YOLO 标签信息: 路径={file_path}, 大小={size}")
                                
                                xml_messages.append([result, file_path, size])
                                logger.debug(f"添加标签消息到列表，当前共 {len(xml_messages)} 个消息")
                            except Exception as e:
                                logger.error(f"保存图像或创建标签时出错: {e}")
                                continue
            else:
                logger.warning(f"Warning: Invalid frame at index {frame_idx}")

        # 始终返回元组，而不是在 return_frames=False 时返回 None
        logger.info(f"帧处理完成，返回 {len(xml_messages)} 个标签消息")
        return processed_frames, xml_messages

    def auto_video_labeling(self, save_image_path=None, save_path=None, text=None, format_type="XML"):
        """
        自动处理视频并生成标签
        Args:
            save_image_path (str): 保存图像的路径
            save_path (str): 保存标签的路径
            text (str): 标签文本说明
            format_type (str): 标签格式，可选 "XML" 或 "YOLO"
        Returns:
            tuple: (processed_frames, xml_messages) - 处理后的帧和标签消息
        """
        logger.info(f"开始自动视频打标，格式: {format_type}, 文本: {text}")
        
        # 确保已经设置了点击坐标
        if self.clicked_x is None or self.clicked_y is None:
            logger.error("点击坐标未设置，无法进行自动打标")
            return None, []
        
        # 确保已经初始化了推理状态
        if self.inference_state is None:
            logger.error("推理状态未初始化，无法进行自动打标")
            return None, []
        
        # 清空现有的视频分割结果
        self.video_segments = {}
        
        # 手动收集所有帧的分割结果
        logger.info("开始收集视频分割结果")
        try:
            # 使用 tqdm 显示进度
            from tqdm import tqdm
            for out_frame_idx, out_obj_ids, out_mask_logits in tqdm(self.predictor.propagate_in_video(self.inference_state), desc="视频分割进度"):
                try:
                    # 处理掩码数据
                    self.video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                    logger.debug(f"帧 {out_frame_idx} 的分割结果: 对象 IDs {out_obj_ids}")
                except Exception as e:
                    logger.error(f"处理帧 {out_frame_idx} 的分割结果时出错: {e}")
            
            logger.info(f"收集到 {len(self.video_segments)} 个帧的分割结果")
        except Exception as e:
            logger.error(f"收集视频分割结果时出错: {e}")
            logger.error(f"错误详情: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        # 如果没有收集到分割结果或只收集到少量帧的分割结果，使用第一帧的掩码为所有帧创建相同的掩码
        frame_names = [
            p for p in os.listdir(self.video_path)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        total_frames = len(frame_names)
        
        if len(self.video_segments) < total_frames * 0.5:  # 如果收集到的帧数少于总帧数的一半
            logger.warning(f"只收集到 {len(self.video_segments)} 个帧的分割结果，总帧数为 {total_frames}，尝试使用第一帧的掩码为所有帧创建相同的掩码")
            try:
                # 添加新点并生成掩码
                if not self.add_new_points_or_box():
                    logger.error("添加新点失败")
                    return None, []
                
                # 检查掩码是否生成
                if self.out_mask_logits is None:
                    logger.error("掩码生成失败")
                    return None, []
                
                # 获取第一帧的掩码
                mask = (self.out_mask_logits[0] > 0.0).cpu().numpy()
                logger.info(f"使用第一帧的掩码为所有帧创建相同的掩码，掩码形状: {mask.shape}")
                
                # 为每一帧设置相同的掩码
                for i in range(total_frames):
                    if i not in self.video_segments:
                        self.video_segments[i] = {1: mask}
                
                logger.info(f"为 {total_frames} 个帧创建了掩码，现在有 {len(self.video_segments)} 个帧的分割结果")
            except Exception as e:
                logger.error(f"创建掩码时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None, []
        
        # 处理所有帧并生成标签
        logger.info("开始处理所有帧并生成标签")
        return self.Draw_Mask_at_frame(
            save_image_path=save_image_path, 
            save_path=save_path, 
            text=text,
            format_type=format_type
        )



if __name__ == '__main__':
    
    AD = AnythingVideo_TW()
    # 提取视频帧
    video_path = r"sampro\notebooks\videos\bedroom.mp4"
    output_dir = r"sampro/notebooks/videos/extracted_frames"
    video_dir = AD.extract_frames_from_video(video_path, output_dir, fps=2)
    frame = AD.set_video(video_dir)
    AD.inference(video_dir)
    AD.Set_Clicked([300, 483], 1)
    AD.add_new_points_or_box()
    # AD.Draw_Mask_picture(frame_stride=1)
    # AD.Draw_Mask((AD.out_mask_logits[0] > 0.0).cpu().numpy(),frame) #暂时不用
    # AD.Draw_Mask_Video(output_video_path="segmented_output.mp4")
    AD.Draw_Mask_at_frame() # 在指定帧上绘制轮廓（例如第10帧）
    cv2.waitKey(0)