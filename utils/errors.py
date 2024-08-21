import torch


def mape_loss(output, target):
    """
    Вычисляет среднюю абсолютную процентную ошибку (MAPE) для многомерного выхода.

    Аргументы:
    output -- прогнозируемые значения (тензор PyTorch)
    target -- истинные значения (тензор PyTorch)

    Возвращает:
    mape -- средняя абсолютная процентная ошибка (скаляр)
    """
    epsilon = 1e-8  # чтобы избежать деления на ноль
    return torch.mean(torch.abs((target - output) / (target + epsilon))) * 100
