"""version 0.1.0"""
from src.dataset import get_dataset
import matplotlib.pyplot as plt


def main():
    """
    Main function
    """
    number_of_samples = 2000
    param = 5.0
    number_of_diffrent_sets = 5
    data, data_debug = get_dataset(
        number_of_samples,
        param,
        number_of_diffrent_sets
    )
    for db_data in data_debug:
        plt.plot(db_data, 'b', linewidth=3)
    plt.show()


if __name__ == "__main__":
    main()
