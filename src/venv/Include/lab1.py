import pylab
import numpy as np


def dichotomy(f, a: float, b: float, eps: float):
    center = 0.0
    n = 0
    while True:
        n += 1
        center = (a + b) / 2

        if abs(b - a) < eps:
            return (center, n)

        f_a = f(a)
        f_b = f(b)
        f_c = f(center)

        if abs(f_c) < eps:
            return (center, n)

        if f_a * f_c < 0:
            b = center
        elif f_b * f_c < 0:
            a = center
        else:
            return None


def nuton_simple(f, x_0: float, eps: float):
    sub = 0.01
    f_ = 3*x_0**2 - 9.6 *x_0 + 3.3

    x_k = x_0
    n = 0
    while True:
        n += 1
        x_k_1 = x_k
        x_k = x_k_1 - f(x_k_1) / f_
        if abs(x_k - x_k_1) < eps:
            return (x_k, n)


def draw(f, a: float, b: float, eps: float):
    x_values = np.arange(start=a, stop=b, step=0.01)
    func_val = f(x_values)

    pylab.ylabel("y")
    pylab.xlabel("x")
    pylab.grid(True)
    # pylab.xlim(a - 1, b + 1)
    info = "x ** 3 - 4.8 * x ** 2 + 3.3 * x + 5"
    pylab.title(info)

    pylab.plot(x_values, func_val, label="variant10")
    pylab.plot([a - 1, b + 1], [0, 0], color='black')

    pylab.legend()
    pylab.show()


def dichotomy_hand(f,eps:float):

    a = float(input("enter left border: "))
    b = float(input("enter right border: "))

    ans = dichotomy(f, a, b, eps)
    print("x: " + str(ans[0]))
    print("iterations: " + str(ans[1]))
    print("f(c): " + str(f(ans[0])))

def nuton_hand(f,eps:float):
    x_0 = float(input("enter the begin value: "))

    ans = nuton_simple(f, x_0, eps)
    print("x: " + str(ans[0]))
    print("iterations: " + str(ans[1]))
    print("f(c): " + str(f(ans[0])))


if __name__ == '__main__':
    f = lambda x: x ** 3 - 4.8 * x ** 2 + 3.3 * x + 5
    a = -1
    b = 4
    eps = 0.001
    # -0.699554443359375
    print("Method of dichotomy")
    dichotomy_hand(f,eps)

    print("\nSimple Newton method")
    nuton_hand(f,eps)



