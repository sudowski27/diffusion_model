"""version 0.1.0"""
import numpy as np


def get_dataset(number_of_samples: int, param: float, number_of_diffrent_sets: int) -> np.ndarray:
    """
    number_of_samples: int
    param: int

    Returns
    -------
    np.ndarray, list
    """
    assert number_of_samples > 0
    assert number_of_diffrent_sets != 0
    assert number_of_samples % number_of_diffrent_sets == 0

    number_of_samples_per_set = number_of_samples // number_of_diffrent_sets
    start = 0
    stop = np.pi / 2
    x = np.linspace(start, stop, number_of_samples_per_set)
    if number_of_diffrent_sets == 1:
        sets = [1.0]
    elif number_of_diffrent_sets == 2:
        sets = [0.5, 1.0]
        sets = np.array(sets)
    else:
        sets = np.linspace(0, 1, number_of_diffrent_sets + 1)[1:]
    data = []

    for i in range(sets.shape[0]):
        temp_data = np.sin(x) * sets[i] * param
        data.append(temp_data)

    data_debug = data.copy()
    data = np.array(data)
    data = data.flatten()

    return data, data_debug
