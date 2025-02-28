import torch
import torch.nn as nn
import math

def generate_fixed_embeddings(num_embeddings, embed_dim):
    """
    Генерация фиксированных эмбеддингов, распределённых на гиперсфере
    (для id и сегмента)
    """
    embeddings = torch.randn(num_embeddings, embed_dim)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings

def generate_sinusoidal_embeddings(num_embeddings, embed_dim):
    """
    Генерация синусоидальных позиционных эмбеддингов для координат
    """
    pe = torch.zeros(num_embeddings, embed_dim)
    position = torch.arange(0, num_embeddings, dtype=torch.float).unsqueeze(1)  # (num_embeddings, 1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class BlockAutoencoder(nn.Module):
    def __init__(self,
                 num_ids: int,
                 grid_size: int = 5,
                 max_coord: int = 995,
                 num_segments: int = 20,
                 id_embed_dim: int = 16,
                 pos_embed_dim: int = 8,
                 seg_embed_dim: int = 8,
                 hidden_dim: int = 64,
                 latent_dim: int = 32):
        super().__init__()
        
        # Инициализация параметров сетки
        self.grid_size = grid_size
        self.max_coord = max_coord
        self.num_positions = (max_coord // grid_size) + 1
        
        train_embeddings = False  # Фиксированные эмбеддинги
        
        # Для эмбеддингов id и сегмента используем "как есть" фиксированные значения
        self.id_embed = self.create_fixed_embedding(num_ids, id_embed_dim)
        self.seg_embed = self.create_fixed_embedding(num_segments, seg_embed_dim)
        
        # Для эмбеддингов координат используем синусоидальное позиционное кодирование
        self.pos_x_embed = self.create_fixed_positional_embedding(self.num_positions, pos_embed_dim)
        self.pos_y_embed = self.create_fixed_positional_embedding(self.num_positions, pos_embed_dim)
        
        # Вычисление размерности входа энкодера
        encoder_input_dim = (id_embed_dim + 
                             2 * pos_embed_dim +
                             2 +  # sin + cos для угла
                             2 +  # flip_x + flip_y
                             seg_embed_dim)
        
        print("Encoder input dimension:", encoder_input_dim)
        print("Hidden dimension:", hidden_dim)
        print("Latent dimension:", latent_dim)
        print("Max coord:", max_coord)
        print("Grid size:", grid_size)
        
        # Энкодер
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Декодер
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Головы декодера
        self.id_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, id_embed_dim)
        )
        self.pos_x_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pos_embed_dim)
        )
        self.pos_y_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pos_embed_dim)
        )
        self.angle_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        self.flip_x_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.flip_y_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.segment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seg_embed_dim)
        )
        
        # Инициализация весов
        self._init_weights()

    def create_fixed_embedding(self, num_embeddings, embed_dim):
        """
        Создаёт слой эмбеддингов с фиксированными значениями (для id и сегмента)
        """
        embeddings = generate_fixed_embeddings(num_embeddings, embed_dim)
        layer = nn.Embedding(num_embeddings, embed_dim)
        layer.weight.data = embeddings
        layer.weight.requires_grad = False
        return layer

    def create_fixed_positional_embedding(self, num_embeddings, embed_dim):
        """
        Создаёт фиксированные позиционные эмбеддинги на основе синусоидального кодирования
        """
        embeddings = generate_sinusoidal_embeddings(num_embeddings, embed_dim)
        layer = nn.Embedding(num_embeddings, embed_dim)
        layer.weight.data = embeddings
        layer.weight.requires_grad = False
        return layer

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            # Инициализируем только те слои эмбеддингов, которые обучаемые
            elif isinstance(module, nn.Embedding) and module.weight.requires_grad:
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def _prepare_inputs(self, inputs):
        """Подготовка входных данных для энкодера"""
        try:
            id_val, pos_x, pos_y, angle, flip_x, flip_y, segment = inputs
        except ValueError:
            print(f"Len of inputs: {len(inputs)}. Input: {inputs}")
            raise
        
        # Преобразование координат в индексы сетки
        pos_x_idx = (pos_x / self.grid_size).long()
        pos_y_idx = (pos_y / self.grid_size).long()
        
        # Эмбеддинги
        id_emb = self.id_embed(id_val)
        pos_x_emb = self.pos_x_embed(pos_x_idx)
        pos_y_emb = self.pos_y_embed(pos_y_idx)
        seg_emb = self.seg_embed(segment)
        
        # Преобразование угла в sin и cos
        angle_rad = torch.deg2rad(angle)
        angle_sin = torch.sin(angle_rad).unsqueeze(-1)
        angle_cos = torch.cos(angle_rad).unsqueeze(-1)
        
        return torch.cat([
            id_emb,
            pos_x_emb,
            pos_y_emb,
            angle_sin,
            angle_cos,
            flip_x.float().unsqueeze(-1),
            flip_y.float().unsqueeze(-1),
            seg_emb
        ], dim=-1)

    def forward(self, inputs):
        # inputs: (block_id, pos_x, pos_y, sin_val, cos_val, flip_x, flip_y, segment)
        block_id, pos_x, pos_y, sin_val, cos_val, flip_x, flip_y, segment = inputs
    
        # Преобразуем координаты в индексы сетки
        pos_x_idx = (pos_x / self.grid_size).long()  
        pos_y_idx = (pos_y / self.grid_size).long()
    
        # Эмбеддинги
        id_emb = self.id_embed(block_id)
        pos_x_emb = self.pos_x_embed(pos_x_idx)
        pos_y_emb = self.pos_y_embed(pos_y_idx)
        seg_emb = self.seg_embed(segment)
    
        # Угол передаётся как sin и cos напрямую
        angle_features = torch.cat([sin_val.unsqueeze(-1), cos_val.unsqueeze(-1)], dim=-1)
    
        encoder_input = torch.cat([
            id_emb,
            pos_x_emb,
            pos_y_emb,
            angle_features,
            flip_x.float().unsqueeze(-1),
            flip_y.float().unsqueeze(-1),
            seg_emb
        ], dim=-1)
    
        latent = self.encoder(encoder_input)
        decoder_out = self.decoder(latent)
    
        return {
            "id_emb": self.id_head(decoder_out),
            "pos_x_emb": self.pos_x_head(decoder_out),
            "pos_y_emb": self.pos_y_head(decoder_out),
            "angle_sin_cos": self.angle_head(decoder_out),
            "flip_x": self.flip_x_head(decoder_out),
            "flip_y": self.flip_y_head(decoder_out),
            "segment_emb": self.segment_head(decoder_out)
        }

    def decode_to_features(self, latent):
        """
        Преобразование латентного представления в фичи.
        Особое внимание уделяем декодированию позиционных эмбеддингов обратно в координаты.
        """
        decoder_out = self.decoder(latent)
        
        # Восстановление параметров
        id_emb = self.id_head(decoder_out)
        pos_x_emb = self.pos_x_head(decoder_out)
        pos_y_emb = self.pos_y_head(decoder_out)
        angle_sin_cos = self.angle_head(decoder_out)
        flip_x = torch.sigmoid(self.flip_x_head(decoder_out))
        flip_y = torch.sigmoid(self.flip_y_head(decoder_out))
        segment_emb = self.segment_head(decoder_out)
        
        # Поиск ближайших фиксированных эмбеддингов (id и сегмента остаются старым способом)
        id_pred = self._nearest_embedding(id_emb, self.id_embed)
        segment_pred = self._nearest_embedding(segment_emb, self.seg_embed)
        
        # Для координат используется новый фиксированный позиционный эмбеддинг (синусоидальный)
        pos_x_idx = self._nearest_embedding(pos_x_emb, self.pos_x_embed)
        pos_y_idx = self._nearest_embedding(pos_y_emb, self.pos_y_embed)
        
        # Восстановление угла из sin и cos
        angle_pred = torch.atan2(angle_sin_cos[:, 0], angle_sin_cos[:, 1])
        angle_deg = torch.rad2deg(angle_pred)
        
        return {
            "id": id_pred,
            "pos_x": pos_x_idx * self.grid_size,  # преобразуем индекс обратно в координату
            "pos_y": pos_y_idx * self.grid_size,
            "angle": angle_deg,
            "flip_x": (flip_x > 0.5).float(),
            "flip_y": (flip_y > 0.5).float(),
            "segment": segment_pred
        }

    def _nearest_embedding(self, query, embedding_layer):
        """
        Поиск ближайшего эмбеддинга в фиксированной таблице методом косинусного сходства.
        Позволяет восстановить индекс позиции по декодированному эмбеддингу.
        """
        with torch.no_grad():
            weights = embedding_layer.weight.detach()  # (num_embeddings, embed_dim)
            query = query.unsqueeze(1)  # (batch_size, 1, embed_dim)
            weights = weights.unsqueeze(0)  # (1, num_embeddings, embed_dim)
            # Косинусное сходство по последней размерности
            similarities = torch.cosine_similarity(query, weights, dim=-1)
            return torch.argmax(similarities, dim=-1)
