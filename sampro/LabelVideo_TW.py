from sampro.sam2.build_sam import build_sam2_video_predictor
import numpy as np
import torch
import cv2
import sys
import os
from pathlib import Path
from util.xmlfile import xml_message
from PIL import Image


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
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))  # 根据文件名排序

        # 加载第一帧
        frame_idx = 5
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
        ann_frame_idx = 0  # 当前帧的索引
        ann_obj_id = 1  # 默认目标 ID

        self.coords.append([self.clicked_x, self.clicked_y])
        self.methods.append(self.method)

        points = np.array(self.coords)
        labels = np.array(self.methods)

        _, self.out_obj_ids, self.out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        self.option = True
        

    # def Draw_Mask(self, mask, frame,obj_id=None):
    #     # 转换 mask 为 NumPy 数组
    #     mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    #     h, w = mask.shape[-2:]
    #     mask = mask.reshape(h, w, 1)
    #     white = np.zeros([h, w, 1], dtype="uint8")
    #     white[:, :, 0] = 255
    #     x = mask * white
    #     x = np.uint8(x)

    #     # 使用 Canny 算法提取轮廓
    #     canny = cv2.Canny(x, 50, 100)
        
    #     # 绘制点击点
    #     self.Draw_Point(frame, self.method)

    #     img = frame.copy()
    #     contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     # 获取最大轮廓
    #     max_area = 0
    #     max_contour = None
    #     for contour in contours:
    #         area = cv2.contourArea(contour)
    #         if area > max_area:
    #             max_area = area
    #             max_contour = contour

    #     if max_contour is not None:
    #         # 绘制最大轮廓的边界框
    #         self.x, self.y, self.w, self.h = cv2.boundingRect(max_contour)
    #         cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 2)
    #         cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    #     self.image_mask = img
    #     cv2.imshow("img", img)
    #     return img
    
    def Draw_Mask(self, mask, frame, obj_id=None):
        # 转换 mask 为 NumPy 数组
        mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        h, w = mask.shape[-2:]
        mask = mask.reshape(h, w, 1)
        
        # 创建一个彩色的 mask
        color_mask = np.zeros_like(frame, dtype=np.uint8)
        color_mask[:, :, 0] = 0    # 蓝色分量
        color_mask[:, :, 1] = 255  # 绿色分量
        color_mask[:, :, 2] = 0    # 红色分量

        # 将 mask 应用到彩色 mask 上
        mask = (mask > 0).astype(np.uint8)  # 二值化
        color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask.squeeze())

        # 叠加到原始帧上 (半透明)
        alpha = 0.5  # 透明度
        frame_with_mask = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)

        # 显示轮廓
        contours, _ = cv2.findContours(mask.squeeze(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame_with_mask, contours, -1, (0, 255, 0), 2)  # 绿色轮廓

        # 初始化最大面积和对应的最大轮廓

        img = frame_with_mask.copy()

        max_area = 0
        max_contour = None

        # 遍历每个轮廓
        for contour in contours:
        # 计算轮廓的面积
            area = cv2.contourArea(contour)
            
            # 如果当前面积大于最大面积，则更新最大面积和对应的最大轮廓
            if area > max_area:
                max_area = area
                max_contour = contour
                
        # 使用矩形框绘制最大轮廓
        self.x, self.y, self.w, self.h = cv2.boundingRect(max_contour)
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 2)
        # 在原图上绘制边缘线
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        self.image_mask = img
        print(self.x, self.y, self.w, self.h)
        return img,self.x, self.y, self.w, self.h

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



    def Draw_Mask_at_frame(self, start_frame=0, return_frames=False, save_image_path=None ,save_path=None, text=None):
        """
        遍历所有帧并绘制轮廓
        Args:
            start_frame (int): 起始帧序号
            return_frames (bool): 是否返回处理后的帧列表
            save_path (str): 保存路径
            text (str): XML文本说明
        Returns:
            tuple: (processed_frames, result, file_path, size) - processed_frames 在 return_frames=False 时为 None
        """
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

        processed_frames = [] if return_frames else None
        result = None
        file_path = None
        size = None

        # 遍历所有帧
        xml_messages = []
        for frame_idx in range(start_frame, len(frame_names)):
            frame_path = os.path.join(self.video_path, frame_names[frame_idx])
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            video_label = []
            
            if frame_idx in self.video_segments:
                for out_obj_id, out_mask in self.video_segments[frame_idx].items():
                    # 检查 Draw_Mask 的返回值
                    result = self.Draw_Mask(out_mask, frame.copy(), out_obj_id)
                    if isinstance(result, tuple) and len(result) == 5:
                        frame, x, y, w, h = result
                        video_label.append([x, y, w, h])
                    else:
                        print(f"Warning: Draw_Mask returned unexpected format at frame {frame_idx}")
                        continue
            
                    # 确保 frame 是有效的图像数组
                    if frame is not None and isinstance(frame, np.ndarray):
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        if return_frames:
                            processed_frames.append(frame)
                            
                        if save_image_path:
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
                            
                            result, file_path, size = xml_message(
                                save_path, frame_idx, int(width), int(height),
                                text, self.x, self.y, self.w, self.h
                            )
                            xml_messages.append([result, file_path, size])
            else:
                print(f"Warning: Invalid frame at index {frame_idx}")

        # 始终返回元组，而不是在 return_frames=False 时返回 None
        return processed_frames, xml_messages



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