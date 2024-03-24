import sympy as sym
import numpy as np

from utils import Utils


class NewtonRaphsonMethod:

    def __init__(self, f):
        x, y, z = sym.symbols('x y z')
        self.f = f

        gradient_vector = [sym.diff(f, var) for var in (x, y, z)]
        self.gradient_vector = sym.lambdify((x, y, z), gradient_vector, 'numpy')

        hessian_matrix = sym.hessian(f, (x, y, z))
        self.hessian_matrix = sym.lambdify((x, y, z), hessian_matrix, 'numpy')

    def compute_next_point(self, t, point,  newton_raphson=True):
        gradient_values = np.array(self.gradient_vector(*point))

        if newton_raphson:
            next_point = point - t * gradient_values
        else:
            next_point = point - np.matmul(t, gradient_values.T)
        return next_point

    def calculate_extremum(self, p0, t=0.1, newton_raphson=True):
        p0 = np.array(p0)
        e1 = e2 = e3 = 1
        i = 0
        print(f"{'i':<5} {'x':<15} {'y':<15} {'z':<15} {'f':<15} {'e1':<15} {'e2':<15} {'e3':<15}")

        if not newton_raphson:
            t = np.linalg.inv(np.array(self.hessian_matrix(*p0)))

        while e1 > 0.01 or e2 > 0.01 or e3 > 0.01:
            p1 = self.compute_next_point(t, p0, newton_raphson)
            e1 = Utils.calculate_error1(p0, p1)
            e2 = Utils.calculate_error2(self.f, p0, p1)
            e3 = Utils.calculate_error3(self.f, p1)

            print(f"{i:<5} {p1[0]:<15.3f} {p1[1]:<15.3f} {p1[2]:<15.3f}"
                  f" {Utils.func_value(self.f, p1):<15.8f} {e1:<15.3f} {float(e2):<15.3f} {float(e3):<15.3f}")

            p0 = p1
            i += 1

        return p0


if __name__ == '__main__':
    x, y, z = sym.symbols('x y z')
    f = 2 * x ** 2 + 3 * y ** 2 + 4 * z ** 2 + 4 * x * y + 4 * x * z + 2 * y * z + 3 * x - y + 5
    nrm = NewtonRaphsonMethod(f)

    nrm.calculate_extremum([10, 10, 10], t=0.1, newton_raphson=True)
    nrm.calculate_extremum([10, 10, 10], newton_raphson=False)

