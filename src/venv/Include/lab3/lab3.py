import numpy as np
import pylab
import math
import time


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


def check_condition(w, a: float, b: float, eps: float, derivative_max: float, derivative_order: int) -> iter:
    """
    :param w: Lagrange polynomial without coefficients
    :param a: left border
    :param b: right border
    :param eps: accuracy
    :param derivative_max: derivative maximum of some order
    :param derivative_order: derivative order
    :return: boolean value of condition on accuracy: |f(x) - P(x)| <=  M_n+1 / (n+1)! * max(w(x))
    """
    # arg_max = arg_max_dichotomy(w, a, b, eps)
    c = [w(x) for x in np.arange(a, b, 10 ** -2)]
    # print(c)
    maximum = np.max(c)
    factorial = math.factorial(derivative_order)
    # (derivative_max / factorial) *
    expression = (derivative_max / factorial) * maximum

    # print("max", maximum)
    # print("expres", expression)
    return (expression < eps, expression)


def build_partition(f, a: float, b: float, eps: float, derivative_max: float, derivative_order: int) -> iter:
    """

    :param f:
    :param a:
    :param b:
    :param eps:
    :param derivative_max:
    :param derivative_order:
    :return:
    разбивает отрезок по которому можно построить итерполяцию
    """
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
            if not (check_condition(w, x_val_i[0], x_val_i[l - 1], eps, derivative_max, n))[0]:
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


def get_border(f_4, a: float, b: float, derr_ord: int, eps: float, left_step: float, right_step: float):
    """

    :param f_4:
    :param a: начальный левый конец
    :param b: изначальный правый конец
    :param derr_ord:
    :param eps:
    :param left_step: сдвиг левой границы на итерации
    :param right_step: сдвиг правой границы на итерации
    :return:
    ищем отрезок на котором выполняется условия по точности при интерполяции полиномом лагранжа

    """
    # елси слишком большая то сдвигаем границы на шаг
    while b > a:
        step = (b - a) / 3
        partit = [a + i * (b - a) / 3 for i in range(4)]
        # print(partit)
        partit_for_max = np.arange(a, b, step=10 ** -2)
        # максимум 4-й производной
        maximum = np.max([abs(f_4(x)) for x in partit_for_max])
        w = get_w(partit)

        if not (check_condition(w, a, b, eps, maximum, derr_ord))[0]:
            a = a + left_step
            b = b - right_step
        else:
            break
    return (a, b, step)
    # newton interpolation polynomial with finite differences forward


def build_newton_interpol_polynom(f, arg_partition: iter):
    f_val = [f(arg_val) for arg_val in arg_partition]
    # for  arg_val in arg_partition:
    #     if abs(arg_val) < np.finfo(float).eps:
    #         f_val.append(f(10 ** -6))
    #     f_val.append(f(arg_val))

    finit_diff = [f_val]
    # строим списоки конечных расностей
    for i in range(len(arg_partition) - 1):
        finit_diff_ord_i = []
        # находим текущий список конечных разностей
        for j in range(len(finit_diff[i]) - 1):
            finit_diff_ord_i.append((finit_diff[i][j + 1] - finit_diff[i][j]))
        # добавляем
        finit_diff.append(finit_diff_ord_i)
        del finit_diff_ord_i

    print()
    print("\tfor newton func val ", f_val)
    for i in range(len(finit_diff)):
        print("\tfor newton subs %d" % (i + 1), finit_diff[i])
    print()
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


def w_Lagrange(x: float, x_val: (list, iter), j: int) -> float:
    ans = 1
    for k, x_k in enumerate(x_val):
        if k != j:
            ans = ans * (x - x_k)
    return ans


def build_Lagrange(f, n: int, x_val: (list, iter)):
    n = n + 1

    def func(x: float) -> float:
        y_val = [f(x_k) for x_k in x_val]
        # находим промежуток где лежит точка
        # pos = 0
        # for k, x_k in enumerate(x_val):
        #     if x >= x_k:
        #         pos = k
        #         break

        # on_left = None
        # on_right = None
        #
        # if pos < n//2:
        #     on_left = 0
        # else:
        #     on_left = pos - n //2
        #
        # on_right = on_left + n
        # if on_right > len(x_val):
        #     on_left = on_right - n
        #     on_right = len(x_val)

        # sub_arr_x = x_val[on_left:on_right]

        sub_arr_x = x_val
        # print(sub_arr_x)
        sub_arr_y = [
            y_k * calc_Lagrange_coef(sub_arr_x, k)
            for k, y_k in enumerate(y_val)
        ]

        w_val = [w_Lagrange(x, sub_arr_x, i) for i in range(len(sub_arr_x))]
        ans = float(np.matmul(w_val, sub_arr_y))
        return ans

    return func


def draw(f, a: float, b: float, eps: float, title: str = "", label: str = "", color="black"):
    x_values = list(np.arange(start=a, stop=b, step=eps))
    func_val = [f(x) for x in x_values]
    print(label, " ", func_val)
    pylab.ylabel("y")
    pylab.xlabel("x")
    pylab.grid(True)
    # pylab.xlim(a - 1, b + 1)
    pylab.title(title)
    # print(2,3, -5,-4, -1, 0)
    pylab.plot(x_values, func_val, label=label, color=color)
    # pylab.plot([a - 1, b + 1], [0, 0], color="black")

    pylab.legend()


if __name__ == '__main__':
    begin = time.time()
    a = 10 ** -3
    b = 2
    derivative_max = 21000
    derivative_order = 4
    # a = float(input("Enter a:"))
    # b = float(input("Enter b:"))
    f = lambda x: np.pi * np.sin(8 * x) / x + x ** 2
    eps = 10 ** -4

    # Строим разбиение по условию
    res = build_partition(f, a, b, eps, derivative_max, derivative_order)
    # print("(len), step, iter, partitions: ", len(res[2]), res)
    partitions = res[2]

    # Строим разбиение сами

    pylab.figure(3)

    # draw(P, a - 0.01, b + 0.01, 10 ** -3, label="newton interpol", color="red")

    info = "f := π * sin(8 * x) / x + x ** 2"

    arr = [a, (a + b) / 3, 2 * (a + b) / 3, b]
    print(check_condition(get_w(arr), a, b, 10 ** -1, derivative_max, derivative_order))
    draw(f, a, b, 10 ** -2, label="variant 10", color="blue")

    a = 0.5
    b = 0.6
    step = (b - a) / 3
    print("another step", step)
    partit = [a + i * step for i in range(4)]
    print("partit", partit)

    P = build_newton_interpol_polynom(f, arg_partition=partit)
    #
    # step = (b - a) / 4
    # partit = [a + i * step for i in range(int((b - a) / step) + 1)]
    # print(arr)
    L_f = build_Lagrange(f, 3, partit)

    draw(P, a, b, 10 ** -2, label="newton interpol", color="red")
    draw(L_f, a, b, 10 ** -2, title=info, label="Lagrange", color="green")


    def error_func_newton(x: float) -> float:
        return abs(f(x) - P(x))


    def error_func_lagrange(x: float) -> float:
        return abs(f(x) - L_f(x))


    pylab.figure(4)
    info = "ERROR"
    draw(error_func_newton, a, b, 10 ** -4, title=info, label="error_func_newton", color="purple")
    draw(error_func_lagrange, a, b, 10 ** -3, title=info, label="error_func_lagrange", color="orange")

    f_4th_diff = lambda x: 2 * np.pi * (
            -256 * np.cos(8 * x) -
            (3 * np.sin(8 * x)) / (x ** 3) +
            (24 * np.cos(8 * x)) / (x ** 2) +
            96 * np.sin(8 * x) / x
    ) / x

    f_4th_diff2 = lambda x: 8 * np.pi * (
            512 * np.sin(8 * x) -
            96 * np.sin(8 * x) / x ** 2 -
            24 * np.cos(8 * x) / x ** 3 +
            3 * np.sin(8 * x) / x ** 4 +
            256 * np.cos(8 * x) / x) / x

    pylab.figure(5)
    info = "f_4th_diff:= \n8 * pi * (512 * sin(8 * x) -96 * sin(8 * x) / x ** 2 \n-24 * cos(8 * x) / x ** 3 +\n3 * sin(8 * x) / x ** 4 +256 * cos(8 * x) / x) / x"
    draw(f_4th_diff2, -1, 4, 10 ** -3, title=info, label="Wow")
    pylab.ylim(-100,21000)
    # pylab.figure(6)

    pylab.show()

    print("End graf\n")
    res = get_border(f_4th_diff2, a, b, 4, 10 ** -3, 0.1, 0.1)
    print(res)
    res = get_border(f_4th_diff2, a, b, 4, 10 ** -3, 0.001, 0.001)
    print(res)

    print()
    x = float(input("Enter x: "))
    print("funct value  = ", f(x))
    print("newton value =", P(x))
    print("Lagran value =", L_f(x))
# 8 * pi * (512 * sin(8 * x) -96 * sin(8 * x) / x ** 2 -24 * cos(8 * x) / x ** 3 +3 * sin(8 * x) / x ** 4 +256 * cos(8 * x) / x) / x