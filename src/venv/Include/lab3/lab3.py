import numpy as np
import pylab
import math


def arg_max_dichotomy(f, a: float, b: float, eps: float) -> float:
    while abs(a - b) > eps:
        x_c = (a + b) / 2
        x1 = x_c - eps
        x2 = x_c + eps
        if f(x1) < f(x2):
            a = x2
        else:
            b = x1
    return (a + b) / 2


def is_step_correct(w, a: float, b: float, eps: float, derivative_max: float, derivative_order: int) -> bool:
    """
    :param w: Lagrange polynomial without coefficients
    :param a: left border
    :param b: right border
    :param eps: accuracy
    :param derivative_max: derivative maximum of some order
    :param derivative_order: derivative order
    :return: boolean value of condition on accuracy: |f(x) - P(x)| <=  M_n+1 / (n+1)! * max(w(x))
    """
    arg_max = arg_max_dichotomy(w, a, b, eps)
    factorial = math.factorial(derivative_order)

    expression = (derivative_max / factorial) * w(arg_max)
    return expression < eps


def build_partition(f, a: float, b: float, eps: float, derivative_max: float, derivative_order: int) -> iter:
    n = derivative_order
    # создаем начальный шаг
    step = (b - a) / (n - 1)
    # номер итерации
    num_iter = 0
    x_val = []
    # проверем шаг разбиения
    while True:
        x_val = [a]
        cur_x_val = a
        while abs(b - cur_x_val) > eps:
            cur_x_val = cur_x_val + step
            x_val.append(cur_x_val)

        is_break = False
        segments_num = len(x_val) - 1
        done_iter = 0
        # количество отзков длиной n(derivative_order), которые можно последовательно выделить в отрежке длиной len(x_val)
        for i in range(len(x_val) - n + 1):
            # разбиение
            # отрезки длиной n, которые последовательно выделяем в отрезке длиной len(x_val),
            # (отрезок состоит из строго последовательных точек, где две соседние различаются на step)
            #  идексы массива [0-4),[1-5),[2-6),[3-7), ...
            x_val_i = x_val[i:i + n]
            # получаем функцию для разбиения
            w = get_w(x_val_i)
            l = len(x_val_i)
            if not is_step_correct(w, x_val_i[0], x_val_i[l - 1], eps, derivative_max, n):
                is_break = True
                break

            done_iter = done_iter + 1
        # если апроксимация порядка derivative_order -1 удовлетворяет на всех отрезках
        if is_break:
            num_iter = num_iter + 1
            step = (b - a) / (segments_num + n - 1)
            continue
        else:
            break

        # done_iter = 0
        # segments_num = 2 ** num_iter
        # # для последовательного разбиения(отрезки идут друг за другом и налегаю друг на друга концами)
        # for i in range(segments_num):
        #     # идексы массива  [0-4),[3-7),[6-10),[9-13), ...
        #     x_val_i = x_val[n + (i - 1) * n - i:n + i * n - i]
        #
        #     # получаем функцию для разбиения
        #     w = get_w(x_val_i)
        #     l = len(x_val_i)
        #     if not is_step_correct(w, x_val_i[0], x_val_i[l - 1], eps, derivative_max, n):
        #         break
        #
        #     done_iter = done_iter + 1
        # # если апроксимация порядка derivative_order -1 удовлетворяет на всех отрезках
        # if done_iter != segments_num:
        #     num_iter = num_iter + 1
        #     step = step / 2
        #     continue
        # else:
        #     break
    # возвращаем разбиение, шаг и количестко итераций
    return (step, num_iter, x_val)


# newton interpolation polynomial with finite differences forward
def build_newton_interpol_polynom(f, arg_partition: iter):
    f_val = [f(arg_val) for arg_val in arg_partition]
    finit_diff = [f_val]
    # строим списоки конечных расностей
    for i in range(len(arg_partition) - 1):
        finit_diff_ord_i = []
        # находим текущий список конечных разностей
        for j in range(len(finit_diff[i]) - 1):
            finit_diff_ord_i.append(finit_diff[i][j + 1] - finit_diff[i][j])
        # добавляем
        finit_diff.append(finit_diff_ord_i)
        del finit_diff_ord_i

    # собираем все первые конечные разности
    # данные что будут использовать в функции
    delta_y_val = [finit_diff[i][0] for i in range(len(finit_diff))]
    x0 = arg_partition[0]
    step = arg_partition[1] - arg_partition[0]

    # удаляем не нужные ссылки
    del f
    del arg_partition
    del f_val
    del finit_diff

    def interpol_polynom(x: float) -> float:
        t = (x - x0) / step
        ans = delta_y_val[0]
        factor_t = 1
        factorial = 1
        for k in range(1, len(delta_y_val)):
            factorial = factorial * k
            factor_t = factor_t * t

            ans = ans + (delta_y_val[k] / factorial) * factor_t

            t = t - 1
        return ans

    return interpol_polynom


def get_w(x_val: (list, iter)):
    def w(x: float) -> float:
        ans = 1
        for k, x_k in enumerate(x_val):
            ans = ans * (x - x_k)
        return ans

    return w


def calc_Lagrange_coef(x_val: (list, iter), j: int) -> float:
    ans = 1
    for k, x_k in enumerate(x_val):
        if k != j:
            ans = ans * (x_val[j] - x_k)

    return 1 / ans


def draw(f, a: float, b: float, eps: float, title: str = "", label: str = "", color="black"):
    x_values = np.arange(start=a, stop=b, step=eps)
    func_val = f(x_values)

    pylab.ylabel("y")
    pylab.xlabel("x")
    pylab.grid(True)
    # pylab.xlim(a - 1, b + 1)
    pylab.title(title)

    pylab.plot(x_values, func_val, label=label, color=color)
    pylab.plot([a - 1, b + 1], [0, 0], color="black")

    pylab.legend()


if __name__ == '__main__':
    a = -1
    b = 1
    derivative_max = 3200
    derivative_order = 4
    # a = float(input("Enter a:"))
    # b = float(input("Enter b:"))
    f = lambda x: np.pi * np.sin(8 * x) / x + x ** 2
    f_4th_diff = lambda x: 2 * np.pi * (
            -256 * np.cos(8 * x) -
            (3 * np.sin(8 * x)) / (x ** 3) +
            (24 * np.cos(8 * x)) / (x ** 2) +
            96 * np.sin(8 * x) / x
    ) / x
    eps = 10 ** -4

    pylab.figure(1)
    info = "f := π * sin(8 * x) / x + x ** 2"
    draw(f, a, b, eps, title=info, label="variant 10", color="blue")

    # pylab.figure(2)
    # info = "f_4th_diff:= 2*​π*​((‑256)*​cos(​8*​x)-​3*​sin(​8*​x)/​x^​3+​24*​cos(​8*​x)/​x^​2+​96*​sin(​8*​x)/​x)/​x"
    # draw(f_4th_diff, -0.5, 1, 10 ** -6, title=info)
    # pylab.ylim((-100, 3200))

    # draw(f, a, b, 10 ** -1 / 2, title=info, label="variant 10", color="red")
    res = build_partition(f, a, b, eps, derivative_max, derivative_order)
    print("partitions, step, iter: ", res)
    partitions = res[2]

    P = build_newton_interpol_polynom(f, arg_partition=partitions)

    draw(P, a - 0.01, b + 0.01, 10 ** -3, label="newton interpol", color="red")
    pylab.show()
