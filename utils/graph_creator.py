import numpy as np
import networkx as nx
import pandas as pd
from utils import *


def corr_graph(path: str, threshold: float = 0.5, window_size: int = 12):
    time_series = pd.read_csv(get_project_root() / 'data' / path)
    time_series.drop('Unnamed: 0', axis=1, inplace=True)
    time_series = time_series.to_numpy()
    num_dimensions = time_series.shape[1]
    graphs = []
    # Параметры окна
    for window_start in range(time_series.shape[0] - window_size):
        window_end = window_start + window_size

        # Извлечение окна
        window = time_series[window_start:window_end, :]

        # Вычисление корреляционной матрицы для временного окна
        correlation_matrix = np.corrcoef(window, rowvar=False)

        # Построение графа
        G = nx.Graph()

        # Добавление узлов и рёбер на основе корреляций
        for i in range(num_dimensions):
            G.add_node(i, value=window[:, i])

        for i in range(num_dimensions):
            for j in range(i + 1, num_dimensions):
                if abs(correlation_matrix[i, j]) > threshold:  # Порог корреляции
                    G.add_edge(f'dim{i}', f'dim{j}', weight=correlation_matrix[i, j])
        graphs.append(G)
    return graphs


