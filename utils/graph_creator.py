import numpy as np
import networkx as nx
import pandas as pd
from torch_geometric.utils import from_networkx
import torch
from tqdm.notebook import tqdm
from utils.utils import *


def split_features(node_features, split_size):
    return [node_features[i:i + split_size] for i in range(0, len(node_features), split_size)]


def corr_graph(
        name: str,
        threshold: float = 0.2,
        window_size: int = 12,
        temporal: bool = False,
        temporal_window: int = 2
):
    if window_size % 2 != 0:
        raise ValueError("window_size must be even")
    if window_size % temporal_window != 0 and temporal:
        raise ValueError("window_size must be completely divided into temporal_windows")
    time_series = pd.read_csv(get_project_root() / name)
    time_series.drop('Unnamed: 0', axis=1, inplace=True)
    time_series = time_series.to_numpy()
    num_dimensions = time_series.shape[1]
    graphs = []
    # Параметры окна
    for window_start in tqdm(range(time_series.shape[0] - window_size), desc="Processing"):
        window_end = window_start + window_size
        # Извлечение окна
        window = time_series[window_start:window_end, :]
        # Вычисление корреляционной матрицы для временного окна
        correlation_matrix = np.corrcoef(window, rowvar=False)

        # Построение графа
        G = nx.Graph()

        # Добавление узлов и рёбер на основе корреляций
        for i in range(num_dimensions):
            G.add_node(i, feature=window[:, i])

        for i in range(num_dimensions):
            for j in range(i + 1, num_dimensions):
                if abs(correlation_matrix[i, j]) > threshold:  # Порог корреляции
                    G.add_edge(i, j, weight=1)
        if temporal:
            temporal_graphs = [nx.Graph() for _ in range(num_dimensions // temporal_window)]
            for num in range(len(temporal_graphs)):
                temporal_graphs[num].add_edges_from(G.edges())
            for node in G.nodes():
                features = G.nodes[node]['feature']
                splitted_features = split_features(features, temporal_window)
                for num in range(len(temporal_graphs)):
                    temporal_graphs[num].add_node(node, feature=splitted_features[num])
            torch_graphs = []
            for num in range(len(temporal_graphs)):
                temporal_graphs[num] = temporal_graphs[num].to_undirected()
                data = from_networkx(temporal_graphs[num].to_undirected())
                data.x = torch.tensor([temporal_graphs[num].nodes[n]['feature'] for n in temporal_graphs[num].nodes()],
                                      dtype=torch.float)
                data.y = time_series[window_end]
                torch_graphs.append(data)
            graphs.append(torch_graphs)
        else:
            G = G.to_undirected()
            data = from_networkx(G.to_undirected())
            data.x = torch.tensor([G.nodes[n]['feature'] for n in G.nodes()], dtype=torch.float)
            data.y = time_series[window_end]
            graphs.append(data)
    return graphs


def fully_connected_graph_window(
        name: str,
        threshold: float = 0.2,
        window_size: int = 12,
        temporal: bool = False,
        temporal_window: int = 2
):
    pass
