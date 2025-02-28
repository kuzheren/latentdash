import os
import pickle
import torch
import math
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

class BlockDataset(Dataset):
    def __init__(self, data_dir, grid_resolution=5):
        self.grid_resolution = grid_resolution
        self.blocks = []  # уникальные блоки
        self.id_to_idx = OrderedDict()
        unique_blocks = set()

        # Загрузка и обработка всех pickle файлов
        for file in os.listdir(data_dir):
            if file.endswith('.pkl'):
                with open(os.path.join(data_dir, file), 'rb') as f:
                    levels = pickle.load(f)
                    for level in tqdm(levels, desc=f"Processing levels in {file}"):
                        for chunk in level:
                            for block in chunk:
                                # Валидация блока: блок должен состоять ровно из 7 элементов
                                if len(block) != 7:
                                    continue
                                # Проверка координат на сетку
                                x, y = block[1], block[2]
                                if x % self.grid_resolution != 0 or y % self.grid_resolution != 0:
                                    continue
                                # Преобразуем блок в кортеж, чтобы использовать для уникальности.
                                block_tuple = tuple(block)
                                if block_tuple in unique_blocks:
                                    continue
                                unique_blocks.add(block_tuple)
                                self.blocks.append(block)
        
        unique_ids = set()
        max_x_idx = 0
        max_y_idx = 0
        max_seg_idx = 0

        for b in self.blocks:
            unique_ids.add(b[0])
            x_idx = b[1] // self.grid_resolution
            y_idx = b[2] // self.grid_resolution
            seg_idx = b[6]
            if x_idx > max_x_idx:
                max_x_idx = x_idx
            if y_idx > max_y_idx:
                max_y_idx = y_idx
            if seg_idx > max_seg_idx:
                max_seg_idx = seg_idx
        
        # Создание маппинга ID блоков
        self.unique_ids = sorted(list(unique_ids))
        self.id_to_idx = {uid: idx for idx, uid in enumerate(self.unique_ids)}
        self.idx_to_id = {v: k for k, v in self.id_to_idx.items()}

        self.x_idx_max = max_x_idx
        self.y_idx_max = max_y_idx
        self.max_seg_idx = max_seg_idx

        print(f"Blocks after filtering (unique blocks): {len(self.blocks)}")
        print(f"Unique ID count: {len(self.unique_ids)}")
        print(f"{self.unique_ids}")
        print(f"Max x_idx={self.x_idx_max}, y_idx={self.y_idx_max} ({self.x_idx_max * self.grid_resolution}, {self.y_idx_max * self.grid_resolution})")
        if self.blocks:
            print(f"Example block: {self.blocks[0]}")

    def __len__(self):
        return len(self.blocks)
    
    def __getitem__(self, idx):
        block = self.blocks[idx]
        # [id, x, y, flip_x, flip_y, rotation_deg, segment]
        block_id = torch.tensor(self.id_to_idx[block[0]], dtype=torch.long)
        
        # приводим к индексам на сетке (делим на grid_resolution)
        grid_x = block[1] // self.grid_resolution
        grid_y = block[2] // self.grid_resolution
        
        rotation_deg = float(block[5])
        rotation_rad = math.radians(rotation_deg)
        sin_rot = math.sin(rotation_rad)
        cos_rot = math.cos(rotation_rad)
        
        x_flip = 1.0 if float(block[3]) != 0 else 0.0
        y_flip = 1.0 if float(block[4]) != 0 else 0.0
        
        segment = torch.tensor(block[6], dtype=torch.long)
        
        numeric_features = torch.tensor([
            grid_x,   # x_idx
            grid_y,   # y_idx
            sin_rot,  # sin(rotation)
            cos_rot,  # cos(rotation)
            x_flip,   # flip_x
            y_flip    # flip_y
        ], dtype=torch.float32)
        
        return (block_id, numeric_features, segment)

    def get_num_ids(self):
        return len(self.unique_ids)

    def get_max_x_idx(self):
        return self.x_idx_max

    def get_max_y_idx(self):
        return self.y_idx_max

    def get_max_seg_idx(self):
        return self.max_seg_idx
