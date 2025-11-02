import os
import json
import random
import torch
import decord
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

# Đảm bảo decord sử dụng torch tensor
decord.bridge.set_bridge("torch")

class CustomVideoDataset(Dataset):
    def __init__(self,
                 data_root,          # Đường dẫn đến thư mục gốc chứa 'annotations' và 'samples'
                 split='train',        # 'train' hoặc 'val'
                 query_params=None,    # Dict chứa các tham số cho query
                 clip_params=None,     # Dict chứa các tham số cho clip
                 split_ratio=0.8):     # Tỷ lệ chia train/val
        
        self.data_root = data_root
        self.split = split
        self.query_params = query_params
        self.clip_params = clip_params
        
        # Giá trị để padding
        self.padding_value = 0.5 if self.clip_params.get('padding_value', 'mean') == 'mean' else 0.0

        self.samples = self._load_metadata(split_ratio)
        print(f"Đã tải xong dữ liệu. Split '{self.split}' có {len(self.samples)} mẫu.")

    def _load_metadata(self, split_ratio):
        """
        Tải file annotations.json và "làm phẳng" nó để mỗi cặp (ảnh query, video) là một mẫu riêng biệt.
        """
        annotations_path = os.path.join(self.data_root, 'annotations', 'annotations.json')
        with open(annotations_path, 'r') as f:
            all_video_annotations = json.load(f)

        # Tạo danh sách các video_id để chia train/val
        all_video_ids = [anno['video_id'] for anno in all_video_annotations]
        random.shuffle(all_video_ids)
        num_train = int(len(all_video_ids) * split_ratio)
        
        if self.split == 'train':
            split_video_ids = set(all_video_ids[:num_train])
        else: # 'val'
            split_video_ids = set(all_video_ids[num_train:])

        flat_samples = []
        for video_info in all_video_annotations:
            video_id = video_info['video_id']
            if video_id not in split_video_ids:
                continue

            video_path = os.path.join(self.data_root, 'samples', video_id, 'drone_video.mp4')
            object_images_dir = os.path.join(self.data_root, 'samples', video_id, 'object_images')

            if not os.path.exists(video_path) or not os.path.exists(object_images_dir):
                print(f"Cảnh báo: Bỏ qua {video_id} vì thiếu video hoặc thư mục object_images.")
                continue

            # Mỗi ảnh trong object_images sẽ tạo ra một mẫu dữ liệu
            query_images = [f for f in os.listdir(object_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for query_image_name in query_images:
                sample = {
                    "video_id": video_id,
                    "video_path": video_path,
                    "query_path": os.path.join(object_images_dir, query_image_name),
                    "bbox_annotations": video_info['annotations'] # bbox theo pixel xyxy
                }
                flat_samples.append(sample)
        
        return flat_samples

    def __len__(self):
        return len(self.samples)

    def _get_video_info(self, video_path):
        """Lấy thông tin cơ bản của video."""
        try:
            video_reader = decord.VideoReader(video_path, num_threads=1)
            vlen = len(video_reader)
            # Lấy kích thước từ frame đầu tiên
            first_frame = video_reader[0].numpy()
            h, w, _ = first_frame.shape
            return vlen, h, w
        except Exception as e:
            print(f"Lỗi khi đọc video {video_path}: {e}")
            return 0, 0, 0

    def _preprocess_image(self, image_tensor, target_size):
        """Tiền xử lý một ảnh: padding thành vuông và resize."""
        # image_tensor có dạng [C, H, W]
        _, h, w = image_tensor.shape
        max_dim = max(h, w)
        pad_h = (max_dim - h) // 2
        pad_w = (max_dim - w) // 2
        padding = (pad_w, pad_h, pad_w, pad_h)
        
        padded_tensor = F.pad(image_tensor, padding, "constant", self.padding_value)
        resized_tensor = F.interpolate(padded_tensor.unsqueeze(0), size=(target_size, target_size), mode='bilinear', align_corners=False).squeeze(0)
        return resized_tensor

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        video_path = sample_info['video_path']
        query_path = sample_info['query_path']
        bbox_annotations = sample_info['bbox_annotations']

        # Lấy thông tin video
        vlen, original_h, original_w = self_get_video_info(video_path)
        if vlen == 0:
            # Trả về một mẫu dummy nếu video bị lỗi
            return self.__getitem__((idx + 1) % len(self))

        # 1. Lấy mẫu clip
        num_frames = self.clip_params['clip_num_frames']
        # Lấy mẫu một đoạn clip ngẫu nhiên
        max_start_idx = max(0, vlen - num_frames)
        start_idx = random.randint(0, max_start_idx)
        clip_idxs = list(range(start_idx, start_idx + num_frames))

        video_reader = decord.VideoReader(video_path, num_threads=1)
        clip_tensors = video_reader.get_batch(clip_idxs).permute(0, 3, 1, 2).float() / 255.0 # [T, C, H, W]

        # 2. Chuẩn bị bbox cho clip
        bbox_lookup = {item['frame']: [item['x1'], item['y1'], item['x2'], item['y2']] for item in bbox_annotations}
        
        clip_bbox_list = []
        clip_with_bbox_list = []
        
        # Chuẩn hóa bbox về khoảng [0, 1]
        norm_tensor = torch.tensor([original_w, original_h, original_w, original_h], dtype=torch.float32)

        for frame_idx in clip_idxs:
            if frame_idx in bbox_lookup:
                bbox = torch.tensor(bbox_lookup[frame_idx], dtype=torch.float32)
                clip_bbox_list.append(bbox / norm_tensor)
                clip_with_bbox_list.append(True)
            else:
                clip_bbox_list.append(torch.zeros(4, dtype=torch.float32)) # Bbox dummy
                clip_with_bbox_list.append(False)

        clip_bbox = torch.stack(clip_bbox_list)
        clip_with_bbox = torch.tensor(clip_with_bbox_list, dtype=torch.float32)

        # 3. Tiền xử lý clip
        clip_processed = self._preprocess_image(clip_tensors[0], self.clip_params['fine_size']).unsqueeze(0)
        for i in range(1, num_frames):
             frame_processed = self._preprocess_image(clip_tensors[i], self.clip_params['fine_size']).unsqueeze(0)
             clip_processed = torch.cat((clip_processed, frame_processed), dim=0)
        
        # 4. Tải và tiền xử lý ảnh query
        query_img = Image.open(query_path).convert("RGB")
        query_tensor = transforms.ToTensor()(query_img)
        query_processed = self._preprocess_image(query_tensor, self.query_params['query_size'])

        # 5. Chuẩn bị query_frame và query_frame_bbox (để cung cấp ngữ cảnh)
        # Chọn một frame ngẫu nhiên có bbox từ video
        annotated_frames = [item for item in bbox_annotations if item['frame'] < vlen]
        if not annotated_frames:
            # Nếu không có frame nào có bbox, dùng frame đầu tiên của clip làm tham chiếu
            ref_frame_info = {'frame': clip_idxs[0], 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1}
        else:
            ref_frame_info = random.choice(annotated_frames)

        ref_frame_idx = ref_frame_info['frame']
        query_frame_tensor = video_reader.get_batch([ref_frame_idx]).squeeze(0).permute(2, 0, 1).float() / 255.0
        query_frame_processed = self._preprocess_image(query_frame_tensor, self.clip_params['fine_size'])
        
        query_frame_bbox_pixel = torch.tensor([ref_frame_info['x1'], ref_frame_info['y1'], ref_frame_info['x2'], ref_frame_info['y2']], dtype=torch.float32)
        query_frame_bbox_normalized = query_frame_bbox_pixel / norm_tensor

        # 6. Chuẩn bị `before_query`
        # Dựa trên frame tham chiếu đã chọn
        before_query = torch.tensor(clip_idxs, dtype=torch.long) < ref_frame_idx
        
        results = {
            'clip': clip_processed.float(),                           # [T,3,H,W]
            'clip_with_bbox': clip_with_bbox.float(),                  # [T]
            'before_query': before_query.bool(),                       # [T]
            'clip_bbox': clip_bbox.float().clamp(min=0.0, max=1.0),    # [T,4]
            'query': query_processed.float(),                          # [3,H2,W2]
            'clip_h': torch.tensor(original_h),
            'clip_w': torch.tensor(original_w),
            'query_frame': query_frame_processed.float(),              # [3,H,W]
            'query_frame_bbox': query_frame_bbox_normalized.float()    # [4]
        }
        return results