'''
Aditi Nair - Assignment 6
March 5 2017

By replacing the original functions with numpy functions, I was able to achieve the following speed-ups.
(Speed-up is calculated as original time/new time.)

ADD: 474.75 
MULTIPLY: 427.89
SQRT: 399.88
HYPOTENUSE: 271.02

'''


# -----------------------------------------------------------------------------
# calculator.py
# ----------------------------------------------------------------------------- 
import numpy as np

def add(x,y):
    """
    Add two arrays using a Python loop.
    x and y must be two-dimensional arrays of the same shape.
    """
    return x+y


def multiply(x,y):
    """
    Multiply two arrays using a Python loop.
    x and y must be two-dimensional arrays of the same shape.
    """
    return x*y

def sqrt(x):
    """
    Take the square root of the elements of an arrays using a Python loop.
    """
    return np.sqrt(x)


def hypotenuse(x,y):
    """
    Return sqrt(x**2 + y**2) for two arrays, a and b.
    x and y must be two-dimensional arrays of the same shape.
    """
    return np.sqrt(x*x + y*y) 