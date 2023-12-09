import sympy as sym
import numpy as np

x, y, z = sym.symbols('x y z')
f = 2 * x ** 2 + 3 * y ** 2 + 4 * z ** 2 + 4 * x * y + 4 * x * z + 2 * y * z + 3 * x - y + 5
# f = 5 * x ** 2 + 2 * y ** 2 + 5 * z ** 2 - 2 * x * y - 4 * x * z - 2 * z + 1
gradient_vector = [sym.diff(f, var) for var in (x, y, z)]
gradient_vector = sym.lambdify((x, y, z), gradient_vector, 'numpy')

hessian_matrix = sym.hessian(f, (x, y, z))
hessian_matrix = sym.lambdify((x, y, z), hessian_matrix, 'numpy')


def compute_t(point):
    # point = np.array(p)
    gradient_values = np.array(gradient_vector(*point))
    hessian_values = np.array(hessian_matrix(*point))

    numerator = np.matmul(gradient_values, gradient_values.T)
    denominator = np.matmul(np.matmul(gradient_values, hessian_values), gradient_values.T)

    return numerator / denominator


# t0 = compute_t([0,0,0])


def compute_next_point(t, point):
    gradient_values = np.array(gradient_vector(*point))
    next_point = point - t * gradient_values
    return next_point


# p = np.array([0,0,0])
# print(compute_next_point(t0, p))

def calculate_error1(p0, p1):
    max_error = -1
    for i in range(p0.size):
        if p0[i] != 0:
            max_error = max(max_error, abs((p0[i] - p1[i]) / p0[i]))
    return max_error


def get_func_value(f, p):
    return f.subs({x: p[0], y: p[1], z: p[2]})


def calculate_error2(p0, p1):
    return abs((get_func_value(f, p0) - get_func_value(f, p1)) / get_func_value(f, p0))


def calculate_error3(p1):
    max_error = -1
    vars = 'xyz'
    for i in range(p1.size):
        if get_func_value(f, p1) != 0:
            max_error = max(max_error, abs(get_func_value(sym.diff(f, vars[i]), p1) / get_func_value(f, p1)))
    return max_error


def main():
    p0 = np.array([0, 0, 0])
    e1 = 1
    e2 = 1
    e3 = 1
    i = 0
    print('%s \t %s \t\t %s \t\t\t %s \t\t %s \t\t %s \t\t %s \t\t %s \t\t %s' % (
        'i', 't', 'x', 'y', 'z', 'f', 'e1', 'e2', 'e3'))
    while e1 > 0.01 or e2 > 0.01 or e3 > 0.01:
        t = compute_t(p0)
        p1 = compute_next_point(t, p0)
        e1 = calculate_error1(p0, p1)
        e2 = calculate_error2(p0, p1)
        e3 = calculate_error3(p1)

        print('%d \t %0.3f \t %0.3f \t %0.3f \t %0.3f \t %0.3f \t %0.3f \t %0.3f \t %0.3f' % (
            i, t, p1[0], p1[1], p1[2], get_func_value(f, p1), e1, e2, e3))

        p0 = p1
        i += 1


main()
