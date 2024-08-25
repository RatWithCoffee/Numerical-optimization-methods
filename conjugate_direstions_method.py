import sympy as sym
import numpy as np

from utils import Utils


class ConjugateDirectionsMethod:

    def __init__(self, f):
        x, y, z = sym.symbols('x y z')
        self.f = f

        gradient_vector = [sym.diff(f, var) for var in (x, y, z)]
        self.gradient_vector = sym.lambdify((x, y, z), gradient_vector, 'numpy')

        hessian_matrix = sym.hessian(f, (x, y, z))
        self.hessian_matrix = sym.lambdify((x, y, z), hessian_matrix, 'numpy')

    def compute_t(self, point, s0):
        gradient_values = np.array(self.gradient_vector(*point))
        hessian_values = np.array(self.hessian_matrix(*point))

        numerator = np.matmul(-gradient_values, s0.T)
        denominator = np.matmul(np.matmul(s0, hessian_values), s0.T)

        if denominator == 0:
            return 0.02

        return numerator / denominator

    def compute_next_point(self, t, point, s):
        next_point = point + t * s
        return next_point

    def compute_beta(self, p0, p1, s):
        num = np.matmul(np.matmul(self.hessian_matrix(*p1), s), np.array(self.gradient_vector(*p1)))
        denum = np.matmul(np.matmul(s, self.hessian_matrix(*p1)),  s.T)
        return num / denum

    def compute_s(self, p, b, s):
        return -np.array(self.gradient_vector(*p)) + b * s

    def calculate_extremum(self, p0):
        p0 = np.array(p0)
        e1 = e2 = 1
        i = 0
        print(f"{'i':<5} {'t':<10} {'x':<15} {'y':<15} {'z':<15} {'f':<15} {'e1':<15} {'e2':<15}")

        s0 = -np.array(self.gradient_vector(*p0))
        while e1 > 0.01 or e2 > 0.01:
            t = self.compute_t(p0, s0)
            p1 = self.compute_next_point(t, p0, s0)
            b1 = self.compute_beta(p0, p1, s0)
            s1 = -np.array(self.gradient_vector(*p1)) + b1 * s0

            e1 = Utils.calculate_error2(self.f, p0, p1)
            e2 = Utils.calculate_error3(self.f, p1)

            print(f"{i:<5} {t:<10.3f} {p1[0]:<15.3f} {p1[1]:<15.3f} {p1[2]:<15.3f}"
                  f" {Utils.func_value(self.f, p1):<15.3f} {float(e1):<15.3f} {float(e2):<15.3f}")

            s0 = s1
            p0 = p1
            i += 1


if __name__ == '__main__':
    x, y, z = sym.symbols('x y z')
    # f = 2 * x ** 2 + 3 * y ** 2 + 4 * z ** 2 + 4 * x * y + 4 * x * z + 2 * y * z + 3 * x - y + 5
    f = 3 * x ** 2 + 4 * y ** 2 + 5 * z ** 2 - x * y - 3 * y * z - 2 * x * z + 1
    sdm = ConjugateDirectionsMethod(f)
    sdm.calculate_extremum([10, 10, 10])
