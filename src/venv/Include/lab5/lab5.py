import numpy as np
import math


def f(x):
    x = np.asarray(x)
    return np.power(x, 2) / np.power(x + 1, 2)


def f_2_diff(x):
    x = np.asarray(x)
    return 2 * (1 - 4 * x / (1 + x) + 3 * np.power(x, 2) / np.power((1 + x), 2)) / np.power((1 + x), 2)


def get_max(f_, a: float, b: float, step: float = 10 ** -5) -> float:
    x = np.arange(a, b + step, step)
    return np.max(np.abs(f_(x)))


def correct_step(h: float, a, b):
    # округление в большую сторону
    count_points = math.ceil((b - a) / h) + 1
    # нужное количество точек
    if count_points % 4 == 0:
        return h
    count_points = count_points + 4 - count_points % 4
    return (count_points, (b - a) / count_points)


def get_step_trapeze(a: float, b: float, eps: float):
    M2 = get_max(f_2_diff, a, b)
    h = math.sqrt(12 * eps / (M2 * (b - a)))
    return h


# def get_step_rect(a: float, b: float, eps: float):
#     M2 = get_max(f_2_diff, a, b)
#     return math.sqrt(24 * eps / (M2 * (b - a)))


# def get_step_simpson(a: float, b: float, eps: float):
#     M4 = get_max(f_4_diff, a, b)
#     h = (eps * 2880 / (M4 * (b - a))) ** 0.25
#     return h


def trapeze_quadrature(y: (np.ndarray), step: float):
    sum = 0
    n = len(y)
    for i in range(1, n - 1):
        sum = sum + y[i]
    sum = sum + (y[0] + y[n - 1]) / 2
    return sum * step


def simpson_quadrature(y: (np.ndarray), step: float):
    sum1 = 0
    sum2 = 0
    n = len(y)
    m = n // 2
    # print([2 * i - 1 for i in range(1, m + 1)])
    # print([2 * i for i in range(1, m)])

    for i in range(1, m + 1):
        sum1 = sum1 + y[2 * i - 1]

    for i in range(1, m):
        sum2 = sum2 + y[2 * i]

    sum1 = 4 * sum1
    sum2 = 2 * sum2

    return (sum1 + sum2 + (y[0] + y[n - 1])) * step / 3


def calc_integral(quad_formule_with_calc_step, a: float, b: float, eps: float, info: str = ""):
    h = quad_formule_with_calc_step[0](a, b, eps)
    n, h = correct_step(h, a, b)
    x = [a + h * i for i in range(n)]
    y = f(x)
    y_2h = [y[2 * i] for i in range(y.size // 2)]

    print(info + "    step: " + str(h))
    print(info + " 2x step: " + str(2 * h))
    print("y_h  array size is " + str(len(y)))
    print("y_2h array size is " + str(len(y_2h)))
    print()

    return (quad_formule_with_calc_step[1](y, h), quad_formule_with_calc_step[1](y_2h, 2 * h))


if __name__ == '__main__':
    a = 1
    b = 4
    eps = 10 ** -4

    data1 = calc_integral((get_step_trapeze, trapeze_quadrature), a, b, eps, "trapeze")
    data2 = calc_integral((get_step_trapeze, simpson_quadrature), a, b, eps, "Simpson")

    # %%
    F = lambda x: -2 * math.log(x + 1) - 1 / (x + 1) + x

    integral_val = F(b) - F(a)
    integral_values = []
    integral_values.extend(data1)
    integral_values.extend(data2)

    info = [
        "trapeze formula with step  h ",
        "trapeze formula with step 2h ",
        "Simpson formula with step  h ",
        "Simpson formula with step 2h "
    ]

    print("integral value on trapeze formula with step  h is " + str(data1[0]))
    print("integral value on trapeze formula with step 2h is " + str(data1[1]))
    print()

    print("integral value on Simpson formula with step  h is " + str(data2[0]))
    print("integral value on Simpson formula with step 2h is " + str(data2[1]))
    print()

    print("integral value on Newton-Leibniz  formula " + str(integral_val))
    print()

    min = np.finfo(float).max
    index = 0

    for i in range(4):
        diff = abs(integral_val - integral_values[i])
        if diff < min:
            index = i
            min = diff

        print("Difference in values between " + info[i] + "and Newton-Leibniz: " + str(diff))

    else:
        print()

    print("Best is " + info[index] + " with difference: " + str(min),"\n")

    print("Runge rule for trapeze: " + str(abs(data1[0] - data1[1]) / 3))
    print("Runge rule for Simpson: " + str(abs(data2[0] - data2[1]) / 15))
