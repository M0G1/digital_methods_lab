import pylab
import numpy as np


def draw2(f, g, border_x, border_y, eps: float):
    x_val = np.arange(start=border_x[0], stop=border_x[1], step=0.01)
    y_val = np.arange(start=border_y[0], stop=border_y[1], step=0.01)

    pylab.ylabel("y")
    pylab.xlabel("x")
    pylab.grid(True)
    # pylab.xlim(a - 1, b + 1)
    info = "variant10"
    pylab.title(info)

    pylab.plot(x_val, f(x_val), label="sin(y + 1) - 1.2", color="blue")
    pylab.plot(g(y_val), y_val, label="1 + cos(x) / 2", color="red")
    # pylab.plot([a - 1, b + 1], [0, 0], color='black')

    pylab.legend()
    pylab.show()


def cond(x: float, eps: float):
    while x < 1:
        x *= 10


def newton(F, J, x_0, eps: float):
    x_1 = None
    n = 0
    while True:
        if (n == 10_000):
            print("Не сходится", x_0)
            print("x_", n, " = ", x_0)
            return None
        F_ = np.linalg.inv(J(x_0))
        # print("F_ = ", F_)
        # print("x_", n, " = ", x_0)
        n += 1
        # print("F(x_", n - 1, ") = ", F(x_0))
        # print("np.dot(F_, F(x_0)) = ", np.dot(F_, F(x_0)))
        x_1 = np.subtract(x_0, np.dot(F_, F(x_0)))
        # print("x_", n, " = ", x_0)
        # print()
        if np.linalg.norm(np.subtract(x_1, x_0)) < eps:
            return (x_1, n)

        x_0 = x_1


def simple_newton(F, J, x_0, eps: float):
    filename = "res.txt"

    x_1 = None
    n = 0
    F_ = np.linalg.inv(J(x_0))

    write_res_simple_newton(filename, "x_0 = " + str(x_0) + "\n")
    write_res_simple_newton(filename, "J(x_0)^-1 = " + str(F_) + "\n")

    while True:
        if (n == 200):
            print("Не сходится", x_0)
            print("x_", n, " = ", x_0)
            print("F(x_", n - 1, ") = ", F(x_0))
            return None
        # print("F_ = ", F_)
        # print("x_", n, " = ", x_0)
        n += 1
        # print("F(x_", n - 1, ") = ", F(x_0))
        # print("np.dot(F_, F(x_0)) = ", np.dot(F_, F(x_0)))
        x_1 = np.subtract(x_0, np.dot(F_, F(x_0)))

        if any(x_i is None for x_i in x_1) or n == 99:
            write_res_simple_newton(filename,
                                    ("x_%d = %s\n" % (n - 1, str(x_0))) + ("F(x_%d) = %s\n\n" % (n - 1, str(F(x_0)))))

        # print("x_", n, " = ", x_0)
        # print()
        if np.linalg.norm(np.subtract(x_1, x_0)) < eps:
            write_res_simple_newton(filename, ("x_%d = %s\n" % (n, str(x_1))) + (
                        "F(x_%d) = %s\n" % (n, str(F(x_1)))) + "solution in " + str(n) + "\n\n")
            return (x_1, n)

        x_0 = x_1


def write_res_simple_newton(filename: str, val: str):
    file = open("res.txt", mode="a")
    file.write(val)


def simple_iterations(F, x_0, eps: float) -> tuple:
    x_1 = F(x_0)
    n = 0
    while np.linalg.norm(np.subtract(x_1, x_0)) > eps:
        if (n == 100):
            print("Не сходится", x_0)
            print("x_", n, " = ", x_0)
            return None
        # print("x_", n, " = ", x_0)
        # print("||x_", n + 1, " - ", "x_", n, "|| = ", np.linalg.norm(np.subtract(x_1, x_0)), '\n')
        n += 1
        x_0 = x_1
        x_1 = F(x_0)
    return (x_1, n)


def F(x: (np.ndarray, tuple, list)) -> tuple:
    return (np.sin(x[1] + 1) - x[0] - 1.2, 2 * x[1] + np.cos(x[0]) - 2)


def qp(x: (np.ndarray, list, tuple)) -> tuple:
    f1 = lambda y: np.sin(y + 1) - 1.2
    f2 = lambda x: 1 - np.cos(x) / 2
    return (f1(x[1]), f2(x[0]))


def J(x: (np.ndarray, list, tuple)) -> tuple:
    f1_x2 = lambda y: -np.cos(y + 1)
    f2_x1 = lambda x: -np.sin(x)
    return ((-1, f1_x2(x[1])),
            (f2_x1(x[0]), 2))


def simple_iteration_handle(eps: float):
    print("Simple itreation method")
    x0 = float(input("Enter begin the value of x: "))
    y0 = float(input("Enter begin the value of y: "))

    ans = simple_iterations(qp, (x0, y0), eps)
    print("x': " + str(ans[0]))
    print("iterations: " + str(ans[1]))
    print("F(x'): " + str(F(ans[0])))
    print()


def newton_handle(eps: float):
    print("Newton method")
    x0 = float(input("Enter begin the value of x: "))
    y0 = float(input("Enter begin the value of y: "))

    ans = newton(F, J, (x0, y0), eps)
    if ans is not None:
        print("x': " + str(ans[0]))
        print("iterations: " + str(ans[1]))
        print("F(x'): " + str(F(ans[0])))
        print()


def simple_newton_handle(eps: float):
    print("Simple Newton method")
    x0 = float(input("Enter begin the value of x: "))
    y0 = float(input("Enter begin the value of y: "))

    ans = simple_newton(F, J, (x0, y0), eps)
    if ans is not None:
        print("x': " + str(ans[0]))
        print("iterations: " + str(ans[1]))
        print("F(x'): " + str(F(ans[0])))
        print()


if __name__ == '__main__':
    a = -1
    b = 3
    eps = 10 ** -5

    border_x = (-10, 10)
    border_y = (-10, 10)
    # draw2(f, g, border_x, border_y, 0.01)
    # newton_handle()
    # simple_iteration_handle(eps)

    f1 = lambda y: np.sin(y + 1) - 1.2
    f2 = lambda x: 1 - np.cos(x) / 2

    newton_handle(eps)
    simple_newton_handle(eps)
    draw2(f1, f2, [-5, 5], [-5, 5], 10 ** -2)
