import sympy as sym


class Utils:

    @staticmethod
    def calculate_error1(p0, p1):
        max_error = -1
        for i in range(p0.size):
            if p0[i] != 0:
                max_error = max(max_error, abs((p0[i] - p1[i]) / p0[i]))
        return max_error if max_error != -1 else 1

    @staticmethod
    def calculate_error2(f, p0, p1):
        if Utils.func_value(f, p0) == 0:
            return 1

        return abs((Utils.func_value(f, p0) - Utils.func_value(f, p1)) / Utils.func_value(f, p0))

    @staticmethod
    def calculate_error3(f, p1):
        max_error = -1
        vars = 'xyz'
        for i in range(p1.size):
            if Utils.func_value(f, p1) != 0:
                max_error = max(max_error,
                                abs(Utils.func_value(sym.diff(f, vars[i]), p1) / Utils.func_value(f, p1)))
        return max_error if max_error != -1 else 1

    @staticmethod
    def func_value(f, p):
        x, y, z = sym.symbols('x y z')
        return f.subs({x: p[0], y: p[1], z: p[2]})
