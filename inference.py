import os
import argparse
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import decord

# Import các file mã nguồn của bạn
from model.corr_clip_spatial_transformer2_anchor_2heads_hnm import ClipMatcher
from config.config import config, update_config

decord.bridge.set_bridge("torch")

def load_config_and_model(cfg_path, checkpoint_path, device):
    """
    Tải config, khởi tạo mô hình, và nạp trọng số đã huấn luyện từ file .pkl hoặc .pth.tar.
    """
    # 1. Tải config
    update_config(cfg_path)

    # 2. Khởi tạo mô hình
    model = ClipMatcher(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    print("Đã nạp trọng số vào mô hình.")
    
    model.to(device)
    model.eval()
    return model, config

# ... (Các hàm còn lại giữ nguyên, không cần thay đổi)
def preprocess_query(query_image_path, query_size=224):
    query = Image.open(query_image_path).convert("RGB")
    transform_to_tensor = transforms.ToTensor()
    query = transform_to_tensor(query)
    _, h, w = query.shape
    max_size = max(h, w)
    pad_h = (max_size - h) // 2
    pad_w = (max_size - w) // 2
    padding = (pad_w, pad_h, pad_w, pad_h)
    transform_pad = transforms.Pad(padding, fill=0.5)
    query = transform_pad(query)
    query = F.interpolate(query.unsqueeze(0), size=(query_size, query_size), mode='bilinear', align_corners=False).squeeze(0)
    return query.float()

def preprocess_clip(clip_frames, target_size=320):
    processed_frames = []
    for frame_img in clip_frames:
        frame_tensor = torch.from_numpy(frame_img).permute(2, 0, 1).float() / 255.0
        processed_frames.append(frame_tensor)
    clip = torch.stack(processed_frames, dim=0)
    _, _, h, w = clip.shape
    max_size = max(h, w)
    pad_h = (max_size - h) // 2
    pad_w = (max_size - w) // 2
    padding = (pad_w, pad_h, pad_w, pad_h)
    transform_pad = transforms.Pad(padding, fill=0.5)
    clip = transform_pad(clip)
    clip = F.interpolate(clip, size=(target_size, target_size), mode='bilinear', align_corners=False)
    return clip.float()



def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    model, config = load_config_and_model(args.cfg, args.checkpoint, device)

    query_tensor = preprocess_query(args.query_image, config.dataset.query_size)
    query_tensor = query_tensor.unsqueeze(0).to(device)

    video_reader = decord.VideoReader(args.video, num_threads=1)
    video_frames = video_reader.get_batch(range(len(video_reader))).numpy()
    video_fps = video_reader.get_avg_fps()

    original_h, original_w = video_frames[0].shape[:2]
    print(f"Video có {len(video_frames)} frames, kích thước {original_w}x{original_h}, {video_fps} FPS.")

    clip_len = config.dataset.clip_num_frames
    stride = clip_len // 2
    all_predictions = {}

    with torch.no_grad():
        for start_idx in range(0, len(video_frames) - clip_len + 1, stride):
            end_idx = start_idx + clip_len
            print(f"Đang xử lý clip từ frame {start_idx} đến {end_idx-1}...")

            clip_to_process = video_frames[start_idx:end_idx]
            clip_tensor = preprocess_clip(clip_to_process, config.dataset.clip_size_fine)
            clip_tensor = clip_tensor.unsqueeze(0).to(device)

            results = model(clip=clip_tensor, query=query_tensor)
            pred_bboxes = results['bbox'].squeeze(0)
            pred_probs = torch.sigmoid(results['prob'].squeeze(0))

            for i in range(clip_len):
                frame_idx_global = start_idx + i
                frame_probs = pred_probs[i]
                high_prob_indices = torch.where(frame_probs > args.threshold)[0]

                if len(high_prob_indices) > 0:
                    frame_bboxes_normalized = pred_bboxes[i][high_prob_indices]
                    frame_scores = frame_probs[high_prob_indices]

                    box_xyxy_normalized = frame_bboxes_normalized
                    max_dim = max(original_w, original_h)
                    pad_w = (max_dim - original_w) / 2
                    pad_h = (max_dim - original_h) / 2
                    box_xyxy_pixel = box_xyxy_normalized * max_dim
                    box_xyxy_pixel[:, [0, 2]] -= pad_w
                    box_xyxy_pixel[:, [1, 3]] -= pad_h
                    box_xyxy_pixel[:, [0, 2]] = torch.clamp(box_xyxy_pixel[:, [0, 2]], 0, original_w)
                    box_xyxy_pixel[:, [1, 3]] = torch.clamp(box_xyxy_pixel[:, [1, 3]], 0, original_h)

                    if frame_idx_global not in all_predictions:
                        all_predictions[frame_idx_global] = []
                    for j in range(len(box_xyxy_pixel)):
                        all_predictions[frame_idx_global].append(
                            (box_xyxy_pixel[j].cpu().numpy().astype(int), frame_scores[j].cpu().item())
                        )

    print(f"Tìm thấy đối tượng trong {len(all_predictions)} frames.")

    # --- Thay vì VideoWriter, lưu từng frame ---
    os.makedirs(args.output, exist_ok=True)
    for idx, frame in enumerate(video_frames):
        if idx in all_predictions:
            for bbox, score in all_predictions[idx]:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{score:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            out_path = os.path.join(args.output, f"frame_{idx:05d}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print(f"✅ Đã lưu {len(all_predictions)} frames có đối tượng tại: {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chạy inference cho ClipMatcher.')
    parser.add_argument('--cfg', type=str, required=True, help='Đường dẫn đến file config .yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Đường dẫn đến file trọng số pretrain (.pkl, .pth, ...)')
    parser.add_argument('--video', type=str, required=True, help='Đường dẫn đến video đầu vào')
    parser.add_argument('--query_image', type=str, required=True, help='Đường dẫn đến ảnh query')
    parser.add_argument('--output', type=str, default='output', help='Đường dẫn để lưu frame kết quả')
    parser.add_argument('--threshold', type=float, default=0.5, help='Ngưỡng xác suất để hiển thị bounding box')
    args = parser.parse_args()
    run_inference(args)
