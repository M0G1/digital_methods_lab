import numpy as np
import math as m
import pylab


def runge_kutta_4(f, x: float, y: float, h: float) -> float:
    """
    :param f: function that is in the equation y' = f (x,y). f must return float value
    :param x:
    :param y:
    :param h: step
    :return: next value of y on method Runge Kutta(4)
    """
    fi_0 = h * f(x, y)
    fi_1 = h * f(x + h / 2, y + fi_0 / 2)
    fi_2 = h * f(x + h / 2, y + fi_1 / 2)
    fi_3 = h * f(x + h, y + fi_2)

    return (fi_0 + 2 * fi_1 + 2 * fi_2 + fi_3) / 6


def f(x: float, y: float) -> float:
    return y / x * (y * m.log(x) - 1)


if __name__ == '__main__':
    a = 1
    b = 5
    x0 = a
    y0 = 0.5
    eps = 10 ** -4

    iteration = 1
    h = 0.1

    # 1   select the step h
    while True:
        yi1_1 = y0 + runge_kutta_4(f, x0, y0, h)
        yi2_1 = yi1_1 + runge_kutta_4(f, x0 + h, yi1_1, h)

        yi2_2 = y0 + runge_kutta_4(f, x0, y0, 2 * h)
        print("h = " + str(h))
        print("x1 = " + str(x0 + h))
        print("y1_h  = " + str(yi1_1))

        print("x2 = " + str(x0 + 2 * h))
        print("y2_h  = " + str(yi2_1))
        print("y2_2h = " + str(yi2_2))

        print("|y2_h - y2_2h|= " + str(abs(yi2_2 - yi2_1)) + "\n")
        if abs(yi2_2 - yi2_1) < eps:
            iteration = iteration + 1
            h = h * 2
        else:
            break

    n = int((b - a) / h)
    n = n + n % 2
    h = (b - a) / n
    h2 = 2 * h

    # 2    find the solution on [a,b] with step h and 2h Runge Kutta(4)

    x_arr = np.arange(a, b + h, h)
    y_arr_h = [y0]
    y_arr_2h = [y0]
    delta_y = [0]

    # вычисляем значения y с шагом h
    for i in range(n):
        y_arr_h.append(
            y_arr_h[i] + runge_kutta_4(f, x_arr[i], y_arr_h[i], h)
        )

    # вычисляем значения y с шагом 2h и находим разностью
    for i in range(n // 2):
        y_arr_2h.append(
            y_arr_2h[i] + runge_kutta_4(f, x_arr[2 * i], y_arr_2h[i], h)
        )
        delta_y.append(abs(y_arr_2h[i] - y_arr_h[2 * i]))

    max_delta_y = np.max(delta_y)

    # 3    find the solution on [a,b] with step h and 2h Euler
    x_arr_e = np.arange(a, b + h, h)
    y_arr_h_e = [y0]
    y_arr_2h_e = [y0]
    delta_y_e = [0]

    # вычисляем значения y с шагом h
    for i in range(n):
        y_arr_h_e.append(
            y_arr_h_e[i] + h * f(x_arr_e[i], y_arr_h_e[i])
        )

    # вычисляем значения y с шагом 2h и находим разностью
    for i in range(n // 2):
        y_arr_2h_e.append(
            y_arr_2h_e[i] + 2 * h * f(x_arr_e[2 * i], y_arr_2h_e[i])
        )
        delta_y.append(abs(y_arr_2h_e[i] - y_arr_h_e[2 * i]))

    max_delta_y_e = np.max(delta_y_e)

    # 4    exact solution to the Koshi problem

    y = lambda x: 1 / (np.log(x) + x + 1)

    y_arr_koshi = y(x_arr)

    delta_koshi_runge_kutte = np.abs(np.subtract(y_arr_koshi, y_arr_h))
    max_delta_koshi_runge_kutte = np.max(delta_koshi_runge_kutte)

    print("Runge-Kutte\n")
    print("x_arr   :\n" + str(x_arr))
    print("y_arr_h : \n" + str(y_arr_h))
    print("y_arr_2h:\n" + str(y_arr_h))
    print("delta_y:\n" + str(y_arr_h))
    print("max_delta_y =" + str(max_delta_y))

    print("Euler\n")
    print("x_arr_e   :\n" + str(x_arr_e))
    print("y_arr_h_e : \n" + str(y_arr_h_e))
    print("y_arr_2h_e:\n" + str(y_arr_h_e))
    print("delta_y_e:\n" + str(y_arr_h_e))
    print("max_delta_y_e =" + str(max_delta_y_e))

    pylab.plot(np.asarray(x_arr_e), np.asarray(y_arr_h_e), label="Euler", color="g")
    pylab.plot(np.asarray(x_arr), np.asarray(y_arr_h), label="Runge-Kutte", color="b")
    pylab.plot(np.asarray(x_arr), np.asarray(y_arr_koshi), label="solution", color="red", linestyle="--")

    pylab.legend()
    pylab.show()
