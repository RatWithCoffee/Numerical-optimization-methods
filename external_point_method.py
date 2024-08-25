import sympy as sym
import numpy as np

from utils import Utils
from newton_raphson_method import NewtonRaphsonMethod
from steepest_descent_method import SteepestDescentMethod

import sympy as sym
import numpy as np

from utils import Utils


class ExternalPoint:

    def __init__(self, f):
        x, y, z = sym.symbols('x y z', real=True)
        self.f = f

        gradient_vector = [sym.diff(f, var) for var in (x, y, z)]
        self.gradient_vector = sym.lambdify((x, y, z), gradient_vector, 'numpy')

    def compute_next_point(self, t, point):
        gradient_values = np.array(self.gradient_vector(*point))

        return point - t * gradient_values

    def calculate_extremum(self, p0, t=0.003):
        p0 = np.array(p0)
        e1 = e2 = e3 = 1
        i = 0
        print(f"{'i':<5} {'x':<15} {'y':<15} {'z':<15} {'f':<15} {'e1':<15} {'e2':<15} {'e3':<15}")

        while e1 > 0.01 or e2 > 0.01 or e3 > 0.01:
            p1 = self.compute_next_point(t, p0)
            e1 = Utils.calculate_error1(p0, p1)
            e2 = Utils.calculate_error2(self.f, p0, p1)
            e3 = calculate_error3(self.f, p1)

            print(f"{i:<5} {p1[0]:<15.3f} {p1[1]:<15.3f} {p1[2]:<15.3f}"
                  f" {func_value(self.f, p1):<15.8f} {e1:<15.3f} {float(e2):<15.3f} {float(e3):<15.3f}")

            p0 = p1
            i += 1

        return p0


def func_value(f, p):
    x, y, z = sym.symbols('x y z', real=True)
    return f.subs({x: p[0], y: p[1], z: p[2]})


def calculate_error3(f, p1):
    max_error = -1
    x, y, z = sym.symbols('x y z', real=True)
    vars = [x, y, z]
    for i in range(p1.size):
        if func_value(f, p1) != 0:
            max_error = max(max_error,
                            abs(func_value(sym.diff(f, vars[i], real=True), p1) / func_value(f, p1)))
    return max_error if max_error != -1 else 1


if __name__ == "__main__":
    x, y, z = sym.symbols('x y z', real=True)

    f = 20 * x ** 2 + 5 * y ** 2 + 13 * z ** 2 - 10 * x * y - 5 * x * z - 12 * y * z + 25 * x - 20 * y - 30 * z
    fi1 = x + 10 * y + 10 * z - 1000
    fi2 = 3 * x + 15 * y + 2 * z - 700
    p0 = np.array([50, 50, 50])

    # f = 2 * x ** 2 + 5 * y ** 2 + 10 * z ** 2 + 6 * x * z - 9 * y * z - 10 * x - 13 * y + 10 * z
    # fi1 = 2 * x - 2 * y + 4 * z - 50
    # fi2 = 2 * x + 3 * y - 2 * z - 100
    # p0 = np.array([10, 10, 10])
    # вар 4
    # f = 8 * x ** 2 + 6 * y ** 2 + 3 * z ** 2 + 10 * x * y + 2 * x * z - 3 * y * z - 35 * x - 2 * y - 40 * z
    # fi1 = 2 * x - y + 7 * z - 11
    # fi2 = -x + 3 * y - 9 * z - 4
    # p0 = np.array([-1, 1, -1])

    # из методы
    # f = 4 * x ** 2 + 6 * y ** 2 + 7 * z ** 2 + 2 * x * y + 10 * x * z + 4 * y * z + 120 * x - 350 * y + 124 * z
    # fi1 = 851 * x + 41 * y + 11 * z - 3950
    # fi2 = -272 * x - 31 * y + 238 * z - 750
    # p0 = np.array([5, 0, 0])

    #
    # f = 4 * x ** 2 + 5 * y ** 2 + 6 * z ** 2 + 7 * x * y + 3 * x * z - 3 * y * z - 30 * x - 2 * y - 20 * z
    # fi1 = x - 5 * y + 10 * z - 30
    # fi2 = x + 3 * y - 9 * z - 40
    # p0 = np.array([4, 16, -1])

    # f = 4 * x ** 2 + 5 * y ** 2 + 6 * z ** 2 + 7 * x * y + 3 * x * z - 3 * y * z - 30 * x - 2 * y - 20 * z
    # fi1 = 2 * x + y + 7 * z - 6;  # <=0
    # fi2 = -x + 2 * y - 5 * z - 5;
    # p0 = np.array([0, 7, 0])

    penalty = 0.25 * ((fi1 + abs(fi1)) ** 2 + (fi2 + abs(fi2)) ** 2)
    c = 5

    tau = 10 ** (-4)
    F = f + tau * penalty

    sdm = ExternalPoint(F)
    pk = sdm.calculate_extremum(p0, 0.05)
    print('Значение функции = ' + str(func_value(f, pk)))
    print('Значение штрафной функции в конечной точке = ' + str(func_value(penalty, pk)))
