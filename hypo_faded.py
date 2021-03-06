#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import bisect

def extrapolate(xs, ys):
    '''Return a function that evaluates as an extrapolation between the arrays.
    xs are expected to be sorted and unique.
    >>> xs = np.linspace(0,10,11)
    >>> ys = xs * xs
    >>> extrapolate(xs, ys)(1.5)  # Square is 2.25, but apprx is 2.5
    2.5
    '''
    def func(x):
        if (x < xs[0]):
            return xs[0]
        if (x > xs[-1]):
            return xs[-1]
        idx = bisect.bisect_left(xs, x)
        if xs[idx] == x:
            return ys[idx]
        return ((xs[idx]-x)*ys[idx-1] + (x-xs[idx-1])*ys[idx])/(xs[idx]-xs[idx-1])
    return func

if __name__ == '__main__':
    import doctest
    doctest.testmod()


fig, ax = plt.subplots()
plt.axis('off')
resolution = 256

num = 3
nloops = 1  # nloops*num should be an integer
k = 5
bound = [-(k+2), (k+2)]
ax.set_xlim(*bound)
ax.set_ylim(*bound)
ax.set_aspect('equal')


theta = np.linspace(0, 2*np.pi*nloops, resolution*nloops+1)

def rotate(theta):
    return np.cos(theta) + 1j*np.sin(theta)

points = rotate(theta)
hypocycloid = lambda theta, k: (k-1)*rotate(theta) + rotate(-(k-1)*theta)
hypo_inner = (k-1)*rotate(theta) + rotate(-(k-1)*theta)
hypo_outer = k*rotate(theta) + rotate(-k*theta)


big_circle, = ax.plot((k+2)*points.real, (k+2)*points.imag, 'b', linewidth=2)
dot, = ax.plot([], [], 'ko', ms=10)
# hypo_inner_path, = ax.plot(hypo_inner.real, hypo_inner.imag, 'r-', linewidth=3.3)
# hypo_outer_path, = ax.plot([], [], 'g-', linewidth=3.3)
# hypo_middle_path, = ax.plot([], [], 'g-', linewidth=3.3)
small_circle, = ax.plot([], [], 'k-', linewidth=2)
# line, = ax.plot([], [], 'k--', linewidth=2)
# line2, = ax.plot([], [], 'k-', linewidth=2)
segments = ax.plot(hypo_inner.real[i:i+2], hypo_inner.imag[i:i+2], 'r-', linewidth=3)[0] for i in range(len(theta)-1)]
gradient = np.linspace[.5,0,len(segments)]

def correct_angle(theta):
    '''
    When np.angle is called, the output is in [-pi,pi].
    This attempts to return an increasing list.
    '''
    for i in range(len(theta)):
        if theta[i] < 0:
            theta[i] += 2 * np.pi
    theta[-1] = 2 * np.pi
    return theta

def frame_generator(step=8):
    for shift in range(0, len(segments), step):
        for i,seg in enumerate(segments)L
        seg.set_alpha(gradient[(i+shift)%len(gradient)])
    yield

def no_op(*args, **kwargs):
    return


ani = FuncAnimation(fig, no_op, frames=frame_generator(), save_count=300, blit=False)

ani.save('hypo_faded.gif', writer='imagemagick', fps=10, dpi=75)

