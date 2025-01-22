import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(r"G:\project\sam2-main\notebooks\images\cars.jpg")
    
    # 示例1：使用点提示
    point_coords = torch.tensor([[500, 375]])  # [x, y]坐标
    point_labels = torch.tensor([1])  # 1表示前景点
    masks, _, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels)
    
    # 示例2：使用框提示
    # box_prompt = torch.tensor([[100, 100, 400, 400]])  # [x1, y1, x2, y2]
    # masks, _, _ = predictor.predict(box=box_prompt)
    
    # 示例3：使用文本提示
    # text_prompt = "a red car"
    # masks, _, _ = predictor.predict(text=text_prompt)