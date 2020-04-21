import numpy as np
import pylab


def get_linear_spline(x_val: (np.ndarray, list), y_val: (np.ndarray, list), h: float):
    A = np.asarray([[1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [1, 0, 0, h, 0, 0],
                    [0, 1, 0, 0, h, 0],
                    [0, 0, 1, 0, 0, h]])

    B = [y[i] for i in range(3)]
    B.extend([y[i] for i in range(1, 4)])
    B = np.asarray(B)

    # Решаем СЛАУ
    coef = np.linalg.solve(A, B).ravel()

    x_0 = x_val[0]
    c_step = 3

    del B
    del A
    del x_val

    def f(x: float) -> float:
        index = int((x - x_0) / h)
        x_sub_x_i = x - index * h - x_0

        if index < 0:
            return y_val[0]
        if index > len(y_val) - 2:
            return y_val[len(y_val) - 1]

        return coef[index] + coef[index + c_step] * x_sub_x_i

    return f


def get_parabolic_spline(x_val: (np.ndarray, list), y_val: (np.ndarray, list), h: float):
    """
         для равно отстоящих узлов интерполяции
     :param x_val:
     :param y_val:
     :param h:
     :return: функцию являющимся параболичесиким сплайном определенной на отрезце [x[0],x[n - 1]]
     где n - количество узлов
     """
    A = np.asarray([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, h, 0, 0, h * h, 0, 0],
                    [0, 1, 0, 0, h, 0, 0, h * h, 0],
                    [0, 0, 1, 0, 0, h, 0, 0, h * h],
                    [0, 0, 0, 1, -1, 0, 2 * h, 0, 0],
                    [0, 0, 0, 0, 1, -1, 0, 2 * h, 0],
                    [0, 0, 0, 0, 0, 0, 2, 0, 0]])
    B = [y[i] for i in range(3)]
    B.extend([y[i] for i in range(1, 4)])
    B.extend([0] * 3)
    B = np.asarray(B)

    # Решаем СЛАУ
    coef = np.linalg.solve(A, B).ravel()

    x_0 = x_val[0]
    c_step = 3

    del B
    del A
    del x_val

    def f(x: float) -> float:
        index = int((x - x_0) / h)
        x_sub_x_i = x - index * h - x_0

        if index < 0:
            return y_val[0]
        if index > len(y_val) - 2:
            return y_val[len(y_val) - 1]

        return coef[index] + coef[index + c_step] * x_sub_x_i + coef[index + 2 * c_step] * x_sub_x_i ** 2

    return f


def get_cubic_spline(x_val: (np.ndarray, list), y_val: (np.ndarray, list), h: float):
    A = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, h, 0, 0, h * h, 0, 0, h ** 3, 0, 0],
                  [0, 1, 0, 0, h, 0, 0, h * h, 0, 0, h ** 3, 0],
                  [0, 0, 1, 0, 0, h, 0, 0, h * h, 0, 0, h ** 3],
                  [0, 0, 0, 1, -1, 0, 2 * h, 0, 0, 3 * h * h, 0, 0],
                  [0, 0, 0, 0, 1, -1, 0, 2 * h, 0, 0, 3 * h * h, 0],
                  [0, 0, 0, 0, 0, 0, 1, -1, 0, 3 * h, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 3 * h, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 3 * h, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3 * h]])

    B = [y[i] for i in range(3)]
    B.extend([y[i] for i in range(1, 4)])
    B.extend([0] * 6)
    B = np.asarray(B)

    # Решаем СЛАУ
    coef = np.linalg.solve(A, B).ravel()

    x_0 = x_val[0]
    c_step = 3

    del B
    del A
    del x_val

    def f(x: float) -> float:
        index = int((x - x_0) / h)
        x_sub_x_i = x - index * h - x_0

        if index < 0:
            return y_val[0]
        if index > len(y_val) - 2:
            return y_val[len(y_val) - 1]

        return coef[index] + coef[index + c_step] * x_sub_x_i + coef[index + 2 * c_step] * x_sub_x_i ** 2 + coef[
            index + 3 * c_step] * x_sub_x_i ** 3

    return f


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
    draw(f_3, a, b, 10 ** -4, label="cubic", color="g")
    pylab.grid(True)

    pylab.figure(1)
    draw(f, a - 10 ** -3, b + 10 ** -3, 10 ** -3, label="variant 10", color="blue")
    draw(f_1, a, b, 10 ** -3, label="linear", color="black")
    draw(f_2, a, b, 10 ** -4, label="parabolic", color="red")
    draw(f_3, a, b, 10 ** -4, label="cubic", color="violet")
    pylab.grid(True)

    pylab.figure(2)

    err_1 = lambda x: abs(f(x) - f_1(x))
    err_2 = lambda x: abs(f(x) - f_2(x))
    err_3 = lambda x: abs(f(x) - f_3(x))

    draw(err_1, a, b, 10 ** -4, label="linear err", color="orange")
    draw(err_2, a, b, 10 ** -4, label="parabolic err", color="purple")
    draw(err_3, a, b, 10 ** -4, label="cubic err", color="pink")

    pylab.grid(True)

    pylab.show()

    x = float(input("Enter point between %f and %f: " % (a, b)))

    print("Linear spline")
    print("f(%-15f) = %f" % (x, f(x)))
    print("spline(%-10f) = %f" % (x, f_1(x)), )
    print("error(%-11f) = %f" % (x, err_1(x)), "\n")

    print("Parabolic spline")
    print("f(%-15f) = %f" % (x, f(x)))
    print("spline(%-10f) = %f" % (x, f_2(x)), )
    print("error(%-11f) = %f" % (x, err_2(x)), "\n")

    print("Cubic spline")
    print("f(%-15f) = %f" % (x, f(x)))
    print("spline(%-10f) = %f" % (x, f_3(x)), )
    print("error(%-11f) = %f" % (x, err_3(x)), "\n")
