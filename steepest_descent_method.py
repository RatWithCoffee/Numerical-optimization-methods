import sympy as sym
import numpy as np

from utils import Utils


class SteepestDescentMethod:

    def __init__(self, f):
        x, y, z = sym.symbols('x y z')
        self.f = f

        gradient_vector = [sym.diff(f, var) for var in (x, y, z)]
        self.gradient_vector = sym.lambdify((x, y, z), gradient_vector, 'numpy')

        hessian_matrix = sym.hessian(f, (x, y, z))
        self.hessian_matrix = sym.lambdify((x, y, z), hessian_matrix, 'numpy')

    # зачение t получаем из решения уравнения dfi(t; X0) / dt = 0
    def compute_t(self, point):
        gradient_values = np.array(self.gradient_vector(*point))
        hessian_values = np.array(self.hessian_matrix(*point))

        numerator = np.matmul(gradient_values, gradient_values.T)
        denominator = np.matmul(np.matmul(gradient_values, hessian_values), gradient_values.T)

        return numerator / denominator

    # x1 = x0 + t * S
    def compute_next_point(self, t, point):
        gradient_values = np.array(self.gradient_vector(*point))
        next_point = point - t * gradient_values
        return next_point

    def calculate_extremum(self, p0):
        p0 = np.array(p0)
        e1 = e2 = 1
        i = 0
        print(f"{'i':<5} {'t':<10} {'x':<15} {'y':<15} {'z':<15} {'f':<15} {'e1':<15} {'e2':<15}")

        while e1 > 0.01 or e2 > 0.01:
            t = self.compute_t(p0)
            p1 = self.compute_next_point(t, p0)
            e1 = Utils.calculate_error2(self.f, p0, p1).evalf()
            e2 = Utils.calculate_error3(self.f, p1).evalf()

            print(f"{i:<5} {t:<10.3f} {p1[0]:<15.3f} {p1[1]:<15.3f} {p1[2]:<15.3f}"
                  f" {Utils.func_value(self.f, p1):<15.3f} {e1:<15.3f} {e2:<15.3f}")

            p0 = p1
            i += 1

        return p0


if __name__ == '__main__':
    x, y, z = sym.symbols('x y z')
    # f = 2 * x ** 2 + 3 * y ** 2 + 4 * z ** 2 + 4 * x * y + 4 * x * z + 2 * y * z + 3 * x - y + 5
    f = 3 * x ** 2 + 4 * y ** 2 + 5 * z ** 2 - x * y - 3 * y * z - 2 * x * z + 1
    sdm = SteepestDescentMethod(f)
    sdm.calculate_extremum([10, 10, 10])
