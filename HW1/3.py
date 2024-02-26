import math
import matplotlib as plt

# from sympy import *


def f(x):
    y = math.pow(x-3,2) + 4 * math.pow(math.e,2*x)
    return y

def f_tag(x):
    y = 2 * (x - 3) + 4 * 2 * math.pow(math.e,2*x)
    return y



def local_min(x0, f, f_tag, iterations = 1000,
                     stopping_threshold = 1e-6):
    n = 0.0001
    x = x0
    iter = []
    for i in range(iterations):
        iter.append(x)
        fdf = f_tag(x)
        x_bar = x - n * fdf
        if abs(f(x_bar) - f(x)) <= stopping_threshold:
            break
        elif f(x_bar) <= f(x):
            x = x_bar
    return x, iter





"""
def local_min(x0, f, f_tag):
    iter = []
    smaller_x = False
    x = x0
    n = 0.001
    while True:
        iter.append(x)
        fdf = f_tag(x)
        x_bar = x - n * fdf
        if f(x_bar) <= f(x):
            smaller_x = True
            continue
        elif (f(x_bar) >= f(x)) and (smaller_x == False):
            n -= 0.01
        else:
            break
    return x, iter


def local_min(x0):
    z = symbols('z')
    f = (x - 3) ** 2 + 4 * math.pow(math.e, 2 * x)
    df = diff(f, z)
    smaller_x = False
    x = x0
    n = 0.1
    while True:
        fdf = df.subs(z,x)
        x_bar = x - n * fdf
        if f.subs(z,x_bar) <= f.subs(z,x):
            smaller_x = True
        elif (f.subs(z,x_bar) >= f.subs(z,x)) and (smaller_x == False):
            n -= 0.01
        else:
            break
    return x
"""

print(local_min(3,f,f_tag))

plt.figure(figsize = (8,6))
plt.plot(local_min(3,f,f_tag)[1])
plt.show()