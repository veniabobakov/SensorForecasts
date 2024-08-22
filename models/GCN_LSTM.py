import torch
import torch.nn as nn
from models.GCN_CONV import GCN_CONV
from models.AttentionLayer_GRU_LSTM import AttentionLayerGRULSTM

class GCN_LSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_gcn_layers, num_rnn_layers, num_features, horizon, dropout=0):
        super(GCN_LSTM, self).__init__()
        self.num_features = num_features
        self.horizon = horizon
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Добавляем первый слой GCN
        self.layers.append(GCN_CONV(in_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        self.dropouts.append(nn.Dropout(dropout))

        # Добавляем остальные GCN-слои
        for _ in range(num_gcn_layers - 2):
            self.layers.append(GCN_CONV(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            self.dropouts.append(nn.Dropout(dropout))

        # LSTM слой
        self.lstm = nn.LSTM(hidden_channels, hidden_channels, num_layers=num_rnn_layers, batch_first=True)

        # Опциональный слой внимания
        self.attention_layer = AttentionLayerGRULSTM(hidden_channels, hidden_channels)

        # Глобальный MaxPooling слой для агрегирования по временной оси
        self.global_max_pooling = nn.AdaptiveMaxPool1d(1)

        # Выходной линейный слой, предсказывающий horizon шагов по num_features на каждый шаг
        self.output_layer = nn.Linear(hidden_channels, num_features * horizon)

    def forward(self, x, edge_index, edge_weight):
        # Проходим через GCN-слои
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = self.dropouts[i](x)

        # Пропускаем через LSTM
        x, _ = self.lstm(x.unsqueeze(0))  # добавляем batch размерность

        # Применяем Global Max Pooling
        x = self.global_max_pooling(x.transpose(1, 2))  # Меняем оси для пула (batch_size, hidden_channels, seq_len)
        x = x.squeeze(-1)  # Убираем ось, созданную после pooling

        # Применяем внимание, если нужно (раскомментируйте следующую строку, если используется AttentionLayer)
        # x = self.attention_layer(x)

        # Предсказываем num_features * horizon значений (один вектор)
        x = self.output_layer(x)

        # Изменяем форму выхода на (num_features, horizon)
        x = x.view(self.num_features, self.horizon)

        return x
