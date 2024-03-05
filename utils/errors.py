import numpy as np


def masked_mape(v,
                v_,
                axis=None
                ):
    """
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    """
    mask = (v == 0)
    percentage = np.abs(v_ - v) / np.abs(v)
    if np.any(mask):
        masked_array = np.ma.masked_array(percentage, mask=mask)  # mask the dividing-zero as invalid
        result = masked_array.mean(axis=axis)
        if isinstance(result, np.ma.MaskedArray):
            return result.filled(np.nan)
        else:
            return result
    return np.mean(percentage, axis).astype(np.float64)


def mape(v,
         v_,
         axis=None
         ):
    """
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    """
    _mape = (np.abs(v_ - v) / np.abs(v) + 1e-5).astype(np.float64)
    _mape = np.where(_mape > 5, 5, _mape)
    return np.mean(_mape, axis)


def rmse(v,
         v_,
         axis=None
         ):
    """
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, RMSE averages on all elements of input.
    """
    return np.sqrt(np.mean((v_ - v) ** 2, axis)).astype(np.float64)


def mae(v,
        v_,
        axis=None
        ):
    """
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAE averages on all elements of input.
    """

    return np.mean(np.abs(v_ - v), axis).astype(np.float64)


def evaluate(y,
             y_hat,
             by_step=False,
             by_node=False
             ):
    """
    :param y: array in shape of [count, time_step, node].
    :param y_hat: in same shape with y.
    :param by_step: evaluate by time_step dim.
    :param by_node: evaluate by node dim.
    :return: array of mape, mae and rmse.
    """
    if not by_step and not by_node:
        return mape(y, y_hat), mae(y, y_hat), rmse(y, y_hat)
    if by_step and by_node:
        return mape(y, y_hat, axis=0), mae(y, y_hat, axis=0), rmse(y, y_hat, axis=0)
    if by_step:
        return mape(y, y_hat, axis=(0, 2)), mae(y, y_hat, axis=(0, 2)), rmse(y, y_hat, axis=(0, 2))
    if by_node:
        return mape(y, y_hat, axis=(0, 1)), mae(y, y_hat, axis=(0, 1)), rmse(y, y_hat, axis=(0, 1))
