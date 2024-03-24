from functools import lru_cache


@lru_cache
def f(x, y):
    if x > y:
        return 0
    elif x == y:
        return 1
    elif x % 10 != 0:
        return f(x + 5, y) + f(x + 10, y) + f(x * (x % 10), y)
    else:
        return f(x + 5, y) + f(x + 10, y)


print(f(10, 220))