import torch
import math
from typing import List
from block_model import BlockAutoencoder

class BlockAutoencoderWrapper:
    def __init__(self, checkpoint_path, device=None):
        """
        Инициализация обёртки для инференса.
         - Загружает чекпоинт, включая состояние модели, конфигурацию и маппинги.
         - Передаёт модель на указанное устройство и переводит её в eval режим.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]
        self.config = config
        
        # Сохраняем маппинги и параметры датасета
        self.id_to_idx = checkpoint.get("id_to_idx", {})
        self.idx_to_id = checkpoint.get("idx_to_id", {})
        self.grid_resolution = checkpoint.get("grid_resolution", 5)
        
        # Инициализируем модель согласно конфигурации
        self.model = BlockAutoencoder(
            num_ids=config["num_ids"],
            grid_size=config["grid_size"],
            max_coord=config["max_coord"],
            num_segments=config["num_segments"],
            id_embed_dim=config["id_embed_dim"],
            pos_embed_dim=config["pos_embed_dim"],
            seg_embed_dim=config["seg_embed_dim"],
            hidden_dim=config["hidden_dim"],
            latent_dim=config["latent_dim"]
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
    
    def _preprocess(self, raw_blocks):
        """
        Принимает один блок или список блоков в формате:
            [id, x, y, flip_x, flip_y, rotation_deg, segment]
        и возвращает кортеж тензоров:
          (block_ids, pos_x, pos_y, sin_vals, cos_vals, flip_x, flip_y, segments)
        где:
          - pos_x, pos_y вычисляются как (grid_index * grid_resolution), grid_index = x // grid_resolution
          - sin и cos вычисляются из rotation_deg (из поля 5)
          - flip_x, flip_y приводятся: !=0 → 1, иначе 0
        """
        single = False
        if not isinstance(raw_blocks, list):
            raw_blocks = [raw_blocks]
            single = True
        
        block_ids, pos_x_list, pos_y_list = [], [], []
        sin_list, cos_list = [], []
        flip_x_list, flip_y_list = [], []
        segments = []
        
        for block in raw_blocks:
            # block = [id, x, y, flip_x, flip_y, rotation_deg, segment]
            # Преобразуем id с помощью маппинга; если нет – используем 0 по умолчанию.
            block_id = self.id_to_idx.get(block[0], 0)
            block_ids.append(block_id)
            # Координаты: сначала получаем grid_index, затем абсолютное значение grid_index * grid_resolution.
            grid_x = int(block[1]) // self.grid_resolution
            grid_y = int(block[2]) // self.grid_resolution
            pos_x_list.append(grid_x * self.grid_resolution)
            pos_y_list.append(grid_y * self.grid_resolution)
            # Вращение: берём из поля rotation_deg, переводим в радианы и вычисляем sin, cos.
            rot_deg = float(block[5])
            rot_rad = math.radians(rot_deg)
            sin_list.append(math.sin(rot_rad))
            cos_list.append(math.cos(rot_rad))
            # Flip: приводим к бинарному значению (если значение != 0, ставим 1).
            flip_x_list.append(1.0 if float(block[3]) != 0 else 0.0)
            flip_y_list.append(1.0 if float(block[4]) != 0 else 0.0)
            segments.append(int(block[6]))
        
        block_ids = torch.tensor(block_ids, dtype=torch.long, device=self.device)
        pos_x = torch.tensor(pos_x_list, dtype=torch.float32, device=self.device)
        pos_y = torch.tensor(pos_y_list, dtype=torch.float32, device=self.device)
        sin_vals = torch.tensor(sin_list, dtype=torch.float32, device=self.device)
        cos_vals = torch.tensor(cos_list, dtype=torch.float32, device=self.device)
        flip_x = torch.tensor(flip_x_list, dtype=torch.float32, device=self.device)
        flip_y = torch.tensor(flip_y_list, dtype=torch.float32, device=self.device)
        segments = torch.tensor(segments, dtype=torch.long, device=self.device)
        
        return (block_ids, pos_x, pos_y, sin_vals, cos_vals, flip_x, flip_y, segments), single

    def encode(self, raw_blocks):
        """
        Принимает сырые блоки (один экземпляр или список блоков) в формате:
           [id, x, y, flip_x, flip_y, rotation_deg, segment]
        возвращает латентные векторы.
        """
        inputs, single = self._preprocess(raw_blocks)
        # Для энкодера нам нужно сформировать входной тензор с размерностью 44, как ожидала модель.
        id_emb = self.model.id_embed(inputs[0])
        pos_x_idx = (inputs[1] / self.model.grid_size).long()
        pos_y_idx = (inputs[2] / self.model.grid_size).long()
        pos_x_emb = self.model.pos_x_embed(pos_x_idx)
        pos_y_emb = self.model.pos_y_embed(pos_y_idx)
        seg_emb = self.model.seg_embed(inputs[7])
        angle_feats = torch.cat([inputs[3].unsqueeze(-1), inputs[4].unsqueeze(-1)], dim=-1)
        flip_feats = torch.cat([inputs[5].unsqueeze(-1), inputs[6].unsqueeze(-1)], dim=-1)
        encoder_input = torch.cat([id_emb, pos_x_emb, pos_y_emb, angle_feats, flip_feats, seg_emb], dim=-1)
        latent = self.model.encoder(encoder_input)
        if single:
            return latent[0].unsqueeze(0)
        return latent

    def decode_from_latent(self, latent):
        """
        Принимает латентное представление (батч) и возвращает восстановленные признаки:
          { 'id':, 'pos_x':, 'pos_y':, 'angle':, 'flip_x':, 'flip_y':, 'segment': }
        """
        outputs = self.model.decode_to_features(latent)
        # Преобразуем индекс id обратно, если map idx_to_id сохранён
        if self.idx_to_id:
            outputs['id'] = [self.idx_to_id[int(idx)] for idx in outputs['id'].tolist()]
        else:
            outputs['id'] = outputs['id'].tolist()
        return outputs

    def forward(self, raw_blocks: List):
        """
        Полный прогон модели: принимает сырые блоки в формате:
           [id, x, y, flip_x, flip_y, rotation_deg, segment]
        и возвращает батч реконструированных блоков в том же формате.
        """
        if not isinstance(raw_blocks, List):
            raise TypeError(f"Blocks should be a List type. Got: {type(raw_blocks)}")

        # Препроцессинг: преобразуем сырые блоки в тензоры
        inputs, single = self._preprocess(raw_blocks)
        # Формируем вход для энкодера вручную:
        id_emb = self.model.id_embed(inputs[0])
        pos_x_idx = (inputs[1] / self.model.grid_size).long()
        pos_y_idx = (inputs[2] / self.model.grid_size).long()
        pos_x_emb = self.model.pos_x_embed(pos_x_idx)
        pos_y_emb = self.model.pos_y_embed(pos_y_idx)
        seg_emb = self.model.seg_embed(inputs[7])
        angle_feats = torch.cat([inputs[3].unsqueeze(-1), inputs[4].unsqueeze(-1)], dim=-1)
        flip_feats = torch.cat([inputs[5].unsqueeze(-1), inputs[6].unsqueeze(-1)], dim=-1)
        encoder_input = torch.cat([id_emb, pos_x_emb, pos_y_emb, angle_feats, flip_feats, seg_emb], dim=-1)
        latent = self.model.encoder(encoder_input)
        decoded = self.model.decode_to_features(latent)
        
        # Преобразование: decoded – это словарь с ключами: 
        # 'id', 'pos_x', 'pos_y', 'angle', 'flip_x', 'flip_y', 'segment'
        # Нам нужно:
        #  - id: преобразовать индекс в реальный id через self.idx_to_id,
        #  - pos_x, pos_y: округлить до целых,
        #  - angle: округлить до целого,
        #  - flip_x, flip_y: взять скаляр, 0 или 1.
        if isinstance(decoded['id'], torch.Tensor):
            pred_ids = decoded['id'].tolist()
            # Преобразуем индексы в оригинальные id, если маппинг есть
            if self.idx_to_id:
                actual_ids = [self.idx_to_id[int(idx)] for idx in pred_ids]
            else:
                actual_ids = pred_ids
            pos_xs = [int(round(x)) for x in decoded['pos_x'].tolist()]
            pos_ys = [int(round(y)) for y in decoded['pos_y'].tolist()]
            angles = [int(round(a)) for a in decoded['angle'].tolist()]
            flip_xs = [int(round(f[0])) for f in decoded['flip_x'].tolist()]
            flip_ys = [int(round(f[0])) for f in decoded['flip_y'].tolist()]
            segments = [int(s) for s in decoded['segment'].tolist()]
        else:
            actual_ids = decoded['id']
            pos_xs = decoded['pos_x']
            pos_ys = decoded['pos_y']
            angles = decoded['angle']
            flip_xs = decoded['flip_x']
            flip_ys = decoded['flip_y']
            segments = decoded['segment']
            
        reconstructed = []
        for i in range(len(actual_ids)):
            reconstructed.append([actual_ids[i],
                                  pos_xs[i],
                                  pos_ys[i],
                                  flip_xs[i],
                                  flip_ys[i],
                                  angles[i],
                                  segments[i]])
        
        if single and len(reconstructed) == 1:
            return reconstructed[0]
        return reconstructed


def test_inference():
    sample_blocks = [
        [5, 995, 0, 1, 0, 180, 19],
        [1, 0, 55, 0, 1, 45, 3]
    ]

    # Инициализация обёртки для инференса (путь к чекпоинту, адаптируйте по необходимости)
    checkpoint_path = r"models_16\final\model.pth"
    wrapper = BlockAutoencoderWrapper(checkpoint_path)

    # Выполняем полный проход через модель: вход -> латентное представление -> восстановленные признаки
    decoded = wrapper.forward(sample_blocks)

    print("Сырые блоки:")
    for block in sample_blocks:
        print(block)
    print("\nРеконструированные признаки:")
    for block in decoded:
        print(block)

if __name__ == "__main__":
    test_inference()
