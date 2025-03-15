import os
import numpy as np
import logging

# 获取日志记录器
logger = logging.getLogger(__name__)

def convert_to_yolo_format(img_width, img_height, x, y, x2, y2):
    """
    将边界框坐标转换为 YOLO 格式（归一化的中心点坐标和宽高）
    
    参数:
    - img_width, img_height: 图像宽高
    - x, y: 边界框左上角坐标
    - x2, y2: 边界框右下角坐标
    
    返回:
    - x_center, y_center, width, height: YOLO 格式的归一化坐标
    """
    logger.debug(f"转换坐标到 YOLO 格式: 图像尺寸=({img_width}, {img_height}), 边界框=({x}, {y}, {x2}, {y2})")
    
    # 计算宽高
    w = x2 - x
    h = y2 - y
    
    # 计算中心点坐标
    x_center = (x + x2) / 2 / img_width
    y_center = (y + y2) / 2 / img_height
    
    # 归一化宽高
    width = w / img_width
    height = h / img_height
    
    logger.debug(f"YOLO 格式坐标: 中心点=({x_center:.6f}, {y_center:.6f}), 尺寸=({width:.6f}, {height:.6f})")
    return x_center, y_center, width, height

def yolo(image_path, save_path, class_map, labels):
    """
    保存 YOLO 格式的标签文件
    
    参数:
    - image_path: 图像路径
    - save_path: 保存路径
    - class_map: 类别名称到索引的映射字典
    - labels: 标签列表，每个标签是一个字典，包含 name 和 bndbox
    
    返回:
    - 保存的文件路径
    """
    logger.info(f"保存 YOLO 格式标签: 图像={image_path}, 保存路径={save_path}")
    logger.debug(f"类别映射: {class_map}")
    logger.debug(f"标签数量: {len(labels)}")
    
    # 获取图像文件名（不含扩展名）
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    logger.debug(f"图像名称: {image_name}")
    
    # 构建保存路径
    txt_path = os.path.join(save_path, f"{image_name}.txt")
    logger.debug(f"标签文件路径: {txt_path}")
    
    # 获取图像尺寸
    img_width = labels[0].get('img_width', 0)
    img_height = labels[0].get('img_height', 0)
    logger.debug(f"图像尺寸: {img_width}x{img_height}")
    
    # 保存类别映射文件
    try:
        # 修复：确保类别映射文件保存在 save_path 目录下，而不是 txt_path 下
        classes_path = os.path.join(save_path, "classes.txt")
        # 按照索引排序类别
        sorted_classes = sorted(class_map.items(), key=lambda x: x[1])
        with open(classes_path, 'w') as f:
            for class_name, _ in sorted_classes:
                f.write(f"{class_name}\n")
        logger.info(f"类别映射已保存到: {classes_path}")
    except Exception as e:
        logger.error(f"保存类别映射出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # 打开文件准备写入
    try:
        # 修复：不需要为单个文件创建目录
        # os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        
        with open(txt_path, 'w') as f:
            for i, label in enumerate(labels):
                # 获取类别索引
                class_name = label['name']
                if class_name not in class_map:
                    # 如果类别不在映射中，添加新类别
                    class_map[class_name] = len(class_map)
                    logger.debug(f"添加新类别到映射: {class_name} -> {class_map[class_name]}")
                class_id = class_map[class_name]
                
                # 获取边界框坐标
                x, y, x2, y2 = label['bndbox']
                logger.debug(f"标签 {i+1}: 类别={class_name}(ID={class_id}), 边界框=({x}, {y}, {x2}, {y2})")
                
                # 转换为 YOLO 格式
                x_center, y_center, width, height = convert_to_yolo_format(img_width, img_height, x, y, x2, y2)
                
                # 写入文件
                line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                f.write(line)
                logger.debug(f"写入行: {line.strip()}")
        
        logger.info(f"YOLO 标签已保存到: {txt_path}")
        return txt_path
    except Exception as e:
        logger.error(f"保存 YOLO 标签出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def yolo_message(save_path, image_name, img_width, img_height, text, x, y, w, h):
    """
    创建 YOLO 格式的标签信息
    
    参数:
    - save_path: 保存路径
    - image_name: 图像名称
    - img_width, img_height: 图像宽高
    - text: 标签文本
    - x, y, w, h: 边界框坐标和尺寸
    
    返回:
    - result: 标签信息字典
    - file_path: 文件保存路径
    - size: 图像尺寸 [宽, 高, 通道数]
    """
    logger.debug(f"创建 YOLO 标签信息: 图像={image_name}, 尺寸={img_width}x{img_height}, 文本={text}")
    logger.debug(f"边界框: x={x}, y={y}, w={w}, h={h}")
    
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # 构建文件路径
    file_path = os.path.join(save_path, f"{image_name}.txt")
    logger.debug(f"YOLO 标签文件路径: {file_path}")
    
    size = [img_width, img_height, 3]  # 添加 size 参数，与 xml_message 保持一致
    
    # 确保边界框格式为 [x, y, x2, y2]，而不是 [x, y, w, h]
    result = {
        'name': text,
        'img_width': img_width,
        'img_height': img_height,
        'bndbox': [x, y, x + w, y + h]  # 修改为 [x, y, x2, y2] 格式
    }
    
    logger.debug(f"YOLO 标签信息已创建: 路径={file_path}, 大小={size}, 边界框={result['bndbox']}")
    return result, file_path, size

# 测试代码
if __name__ == "__main__":
    # 测试转换函数
    img_width, img_height = 640, 480
    x, y, x2, y2 = 100, 100, 200, 150
    x_center, y_center, width, height = convert_to_yolo_format(img_width, img_height, x, y, x2, y2)
    print(f"YOLO format: {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # 测试保存函数
    path = r'dog.jpg'
    save_path = "."
    class_map = {"dog": 0, "cat": 1}
    labels = [
        {
            'name': 'dog',
            'img_width': 640,
            'img_height': 480,
            'bndbox': [100, 100, 200, 150]
        }
    ]
    
    print(yolo(path, save_path, class_map, labels)) 