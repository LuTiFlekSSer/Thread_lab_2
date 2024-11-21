import numpy as np
from matplotlib import pyplot as plt

BASE_PATH = '../../cmake-build-release-wsl/'
FILE_NAME = 'mat.txt'


def main():
    with open(BASE_PATH + FILE_NAME) as file:
        mat = np.array([list(map(float, line.strip().split(' '))) for line in file])

    x, y = np.meshgrid(lin := np.linspace(0, 1, len(mat)), lin)

    _, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(x, y, mat, cmap='Spectral_r')
    ax.view_init(30, 35)
    plt.show()


if __name__ == '__main__':
    main()
