import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class TemporalGraphNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout_prob=0.1):
        super(TemporalGraphNetwork, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.dropouts = torch.nn.ModuleList()
        self.dropouts.append(torch.nn.Dropout(dropout_prob))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.dropouts.append(torch.nn.Dropout(dropout_prob))

        self.additional_convs = torch.nn.ModuleList()
        self.additional_dropouts = torch.nn.ModuleList()

        for _ in range(2):  # Добавление двух дополнительных сверточных слоев
            self.additional_convs.append(GCNConv(hidden_dim, hidden_dim))
            self.additional_dropouts.append(torch.nn.Dropout(dropout_prob))

        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.output_dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, data_list):
        batch_size = len(data_list)
        x_seq = []

        for data in data_list:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            for conv, dropout in zip(self.convs, self.dropouts):
                x = conv(x, edge_index)
                x = F.relu(x)
                x = dropout(x)

            for conv, dropout in zip(self.additional_convs, self.additional_dropouts):
                x = conv(x, edge_index)
                x = F.relu(x)
                x = dropout(x)

            x = global_mean_pool(x, batch)
            x_seq.append(x)

        x_seq = torch.stack(x_seq, dim=1)  # Создаем последовательность из графовых представлений
        lstm_out, _ = self.lstm(x_seq)
        lstm_out = self.output_dropout(lstm_out)
        out = self.fc(lstm_out[:, -1, :])  # Используем только последний выход LSTM для предсказания

        return out