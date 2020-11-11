#!/usr/bin/env python3

import numpy as np

def GCD(a, b):
    while b != 0:
        a, b = b, a % b
    return a

def LCM(a, b):
    return a*b // GCD(a, b)


class Hypotrochoid:
    '''
    A hypotrochoid is a roulette traced by a point attached to a circle
    of radius r rolling around the inside of a fixed circle of radius R,
    where the point is a distance d from the center of the interior circle.
    '''
    def __init__(
        R:'radius of fixed circle',
        r:'radius of rolling circle',
        d:'point distance from center of rolling circle',
        resolution=256:'points to calculate on curve',
        ):
        k = LCM(r, R) // R  # Revolutions to complete curve
        theta = np.linspace(0, 2*k*np.pi, resolution+1)
        self.curve = (R-r)*np.exp(1j*theta) + d*np.conj(np.exp(1j*theta*(R-r)/r))

    def xy(self):
        return (self.curve.real, self.curve.imag)

    def plot(self):

