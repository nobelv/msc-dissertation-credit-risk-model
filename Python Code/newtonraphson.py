# Newton-Raphson Method implemented in Python
# Credits: https://stackoverflow.com/a/20660005/8784352


def derivative(f, x, h):
    return (f(x + h) -
            f(x - h)) / (2.0 * h)


def solve(f, x0, h):
    last_x = x0
    next_x = last_x + 10 * h  # different than last_x so loop starts OK
    while abs(last_x - next_x) > h:  # this is how you terminate the loop - note use of abs()
        new_y = f(next_x)  # just for debug... see what happens
        print("f(", next_x, ") = ", new_y)  # print out progress... again just debug
        last_x = next_x
        next_x = last_x - new_y / derivative(f, last_x, h)  # update estimate using N-R
    return next_x
