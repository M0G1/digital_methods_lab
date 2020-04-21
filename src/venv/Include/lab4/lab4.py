import numpy as np
import pylab


def draw(f, a: float, b: float, eps: float, title: str = "", label: str = "", color="black"):
    x_values = list(np.arange(start=a, stop=b + eps, step=eps))
    func_val = [f(x) for x in x_values]
    print(label, " ", func_val)
    pylab.ylabel("y")
    pylab.xlabel("x")
    pylab.grid(True)
    # pylab.xlim(a - 1, b + 1)
    pylab.title(title)
    # print(2,3, -5,-4, -1, 0)
    print(label)
    print(len(x_values))
    print(len(func_val))
    print()
    pylab.plot(np.asarray(x_values), np.asarray(func_val), label=label, color=color)
    # pylab.plot([a - 1, b + 1], [0, 0], color="black")

    pylab.legend()


def get_linear_spline(x_val: (np.ndarray, list), y_val: (np.ndarray, list), step: float):
    """
        для равно отстоящих узлов интерполяции
    :param x_val:
    :param y_val:
    :param step:
    :return: функцию являющимся линейным сплайном определенной на отрезце [x[0],x[n - 1]]
    где n - количество узлов
    """

    def f(x: float) -> float:
        i: int = 0
        while i < len(x_val) and x >= x_val[i]:
            i = i + 1
        if i is 0:
            i = 1
        if i is len(x_val):
            i = len(x_val) - 1
        return ((x - x_val[i - 1]) * y_val[i] + (x_val[i] - x) * y_val[i - 1]) / step

    return f


def get_parabolic_spline(x_val: (np.ndarray, list), y_val: (np.ndarray, list), step: float, x0_diff_val: float = 0):
    """
        для равно отстоящих узлов интерполяции
    :param x_val:
    :param y_val:
    :param step:
    :param x0_diff_val: значение первой производной функции в точке x0
    :return: функцию являющимся параболичесиким сплайном определенной на отрезце [x[0],x[n - 1]]
    где n - количество узлов
    """
    x_0 = x_val[0]
    del x_val

    b = [x0_diff_val]
    for i in range(1, len(y_val)):
        b.append(2 * (y_val[i] - y_val[i - 1]) / step - b[i - 1])

    c = []
    for i in range(len(y_val) - 1):
        c.append((b[i + 1] - b[i]) / (2 * step))

    # print("len b", len(b))
    # print("len c", len(c))

    def f(x: float) -> float:
        index = int((x - x_0) / step)
        x_sub_x_i = x - index * step - x_0

        # print(x, index, '\n')

        if index < 0:
            return y_val[0]
        if index > len(y_val) - 2:
            return y_val[len(y_val) - 1]

        return y_val[index] + x_sub_x_i * b[index] + x_sub_x_i ** 2 * c[index]

    return f


def get_cubic_spline(x_val: (np.ndarray, list), y_val: (np.ndarray, list), step: float, x0_diff_val: float = 0):
    """
    :param x_val:
    :param y_val:
    :param step:
    :param x0_diff_val:
    :return:
    """
    # len(y_val) = n + 1

    n = len(y_val) - 1
    delta = [0] * n
    delta[0] = -0.25
    # [1 - n] (n)
    for i in range(2, n + 1):
        delta[i - 1] = (-1 / (4 + delta[i - 2]))

    # [0 - n-2] (n-1)
    finite_dif_2 = [
        y_val[i + 2] - 2 * y_val[i + 1] + y_val[i]
        for i in range(len(y_val) - 2)
    ]
    # [1 - n] (n)
    lamda = [0] * n
    lamda[0] = 3 * finite_dif_2[0] / (2 * step) ** 2
    for i in range(2, n + 1):
        lamda[i - 1] = ((3 * finite_dif_2[i - 2] / step ** 2 - lamda[i - 2]) / (4 + delta[i - 2]))

    print()
    print("lambda ", lamda)
    print("delta ", delta)

    print("y-", len(y_val))
    print(y)

    c = [0] * (len(y_val))
    # [0 - n] (n+1)
    for i in range(n, 1, -1):
        # print("\n", i + 1)
        # print(delta[i] * c[i + 1] + lamda[i])
        # print(i, "\n")
        c[i - 1] = delta[i - 1] * c[i] + lamda[i - 1]

    # [1 - n] (n)
    b = [0] * n
    for i in range(1, n + 1):
        # print(y_val[i - 1])
        b[i - 1] = ((y_val[i] - y_val[i - 1]) / step + step * (2 * c[i] + c[i - 1]) / 3)

    # [1 - n] (n)
    d = [0] * n
    for i in range(1, n + 1):
        # d[i -1]
        d[i - 1] = (c[i] - c[i - 1]) / (3 * step)

    x_0 = x_val[0]
    del x_val
    del delta
    del finite_dif_2
    del lamda

    print("b-", len(b))
    print(b)

    print("c-", len(c))
    print(c)

    print("d-", len(d))
    print(d)

    def f(x: float) -> float:
        index = int((x - x_0) / step)
        # print("index-", index)
        x_sub_x_i = x - index * step - x_0

        # print(x, index, '\n')

        if index < 0:
            return y_val[0]
        if index > len(d) - 1:
            return y_val[len(y_val) - 1]

        return y_val[index] + b[index] * x_sub_x_i + d[index] * x_sub_x_i ** 3  # + c[index + 1] * x_sub_x_i ** 2

    return f


def get_cubic_spline2(x_val: (np.ndarray, list), y_val: (np.ndarray, list), step: float, x0_diff_val: float = 0):
    n = len(y_val)
    b = [0] * n
    c = [0] * n
    d = [0] * n
    x_0 = x_val[0]

    delta = [0] * (n - 1)  # al
    lamda = [0] * (n - 1)  # be

    f_const = 0

    # Решение СЛАУ относительно коэффициентов сплайнов c[i] методом прогонки для трехдиагональных матриц

    for i in range(1, n - 1):
        f_const = 6 * ((y_val[i + 1] - 2 * y_val[i] + y_val[i - 1])) / step
        z = (step * delta[i - 1] + 4 * step)
        delta[i] = - step / z
        lamda[i] = (f_const - step * lamda[i - 1]) / z

    c[n - 1] = (f_const - step * lamda[n - 2]) / (4 * step + step * delta[n - 2])
    # Нахождение решения - обратный ход метода прогонки
    for i in range(n - 2, 0, -1):
        c[i] = delta[i] * c[i + 1] + lamda[i]

    # Освобождение памяти, занимаемой прогоночными коэффициентами и ненужным разбиением по
    del x_val
    del lamda
    del delta

    # По известным коэффициентам c[i] находим значения b[i] и d[i]

    for i in range(n - 1, 0, -1):
        d[i] = (c[i] - c[i - 1]) / step
        b[i] = step * (2 * c[i] - c[i - 1]) / 6 + (y_val[i] - y_val[i - 1]) / step

    def f(x: float) -> float:
        # находим нужный сплайн
        index = int((x - x_0) / step)
        # print("index-", index)
        x_sub_x_i = x - index * step - x_0

        # print(x, index, '\n')
        if index < 0:
            return y_val[0]
        if index > len(d) - 1:
            return y_val[len(y_val) - 1]

        # выч знач по схеме горнера
        return y_val[index] + ((b[index] + c[index] / 2 + x_sub_x_i * d[index] / 6) * x_sub_x_i) * x_sub_x_i

    return f


if __name__ == '__main__':
    f = lambda x: np.pi * np.sin(8 * x) / x + x ** 2
    eps = 10 ** -4
    a_glob = 10 ** -3
    b_glob = 2

    a = 0.5
    b = 0.6
    step = (b - a) / 3
    print("step: ", step)
    partit = np.arange(start=a, stop=b + step, step=step)
    print("partit: ", partit)

    y = f(partit)
    print("func val: ", y)

    f_1 = get_linear_spline(partit, y, step)
    f_2 = get_parabolic_spline(partit, y, step)
    f_3 = get_cubic_spline(partit, y, step)

    pylab.figure(0)

    draw(f, a_glob, b_glob, 10 ** -3, label="variant 10", color="blue")
    draw(f_1, a, b, 10 ** -3, label="linear", color="black")
    draw(f_2, a, b, 10 ** -4, label="parabolic", color="red")
    draw(f_3, a, b, 10 ** -3, label="cubic", color="g")
    pylab.grid(True)

    pylab.figure(1)

    err_1 = lambda x: abs(f(x) - f_1(x))
    err_2 = lambda x: abs(f(x) - f_2(x))
    err_3 = lambda x: abs(f(x) - f_3(x))

    draw(err_1, a, b, 10 ** -4, label="linear err", color="orange")
    draw(err_2, a, b, 10 ** -4, label="parabolic err", color="purple")
    draw(err_3, a, b, 10 ** -3, label="cubic err", color="pink")

    pylab.grid(True)

    pylab.show()
