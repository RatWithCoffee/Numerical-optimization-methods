import sympy as sym
import numpy as np

from utils import Utils
from steepest_descent_method import SteepestDescentMethod


if __name__ == "__main__":
    x, y, z = sym.symbols('x y z')
    # вар 4
    f = 8 * x ** 2 + 6 * y ** 2 + 3 * z ** 2 + 10 * x * y + 2 * x * z - 3 * y * z - 35 * x - 2 * y - 40 * z
    fi1 = 2 * x - y + 7 * z - 11
    fi2 = -x + 3 * y - 9 * z - 4
    p0 = np.array([0, 0, 0])

    #
    # f = 4 * x ** 2 + 5 * y ** 2 + 6 * z ** 2 + 7 * x * y + 3 * x * z - 3 * y * z - 30 * x - 2 * y - 20 * z
    # fi1 = x - 5 * y + 10 * z - 30
    # fi2 = x + 3 * y - 9 * z - 40
    # p0 = np.array([0, 0, 0])

    # из методы
    # f = 4 * x ** 2 + 6 * y ** 2 + 7 * z ** 2 + 2 * x * y + 10 * x * z + 4 * y * z + 120 * x - 350 * y + 124 * z
    # fi1 = 851 * x + 41 * y + 11 * z - 3950
    # fi2 = -272 * x - 31 * y + 238 * z - 750
    # p0 = np.array([0, 0, 0])

    # f = 4 * x ** 2 + 5 * y ** 2 + 6 * z ** 2 + 7 * x * y + 3 * x * z - 3 * y * z - 30 * x - 2 * y - 20 * z
    # fi1 = 2 * x + y + 7 * z - 6;  # <=0
    # fi2 = -x + 2 * y - 5 * z - 5;
    # p0 = np.array([0, 0, 0])

    penalty = sym.exp(fi1) + sym.exp(fi2)
    v = 10 ** 4
    f1 = f + v * penalty
    sdm = SteepestDescentMethod(f1)
    pk = sdm.calculate_extremum(p0)
    print('Значение функции = ' + str(Utils.func_value(f, pk)))
    print('Значение штрафной функции в конечной точке = ' + str(Utils.func_value(penalty, pk)))
