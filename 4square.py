#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Assumes SolidPython is in site-packages or elsewhwere in sys.path
from solid import *
from solid.utils import *


fig, ax = plt.subplots()
plt.axis('off')
resolution = 512

k = 2
limit = k+2
bound = (-limit, limit)
ax.set_aspect('equal')
ax.set_xlim(*bound)
ax.set_ylim(*bound)

# big circle:
theta = np.linspace(0, 2*np.pi, resolution+1)
circle = np.exp(1j*theta)

def xy(complex_array):
    return complex_array.real, complex_array.imag

square = []
k = 4
for i in range(k):
    square.append(ax.plot([], [], 'k', linewidth=3)[0])

level = []
theta = np.linspace(0,2*np.pi,5)

def frame_generator(step=8):
    yield
    yield
    S = 2+np.exp(1j*(theta+np.pi/4))
    for i in range(k):
        points = S * np.exp(1j*2*np.pi*i/k)
        square[i].set_data(*xy(points))
        yield
    """
    # Drawing path
    for i in range(0, len(theta), step):
        rotate = np.exp(1j*-theta[i]/k)
        hypo_rotate = rotate*hypo_inner + circle[i]
        hypocycloid_a.set_data(hypo_rotate.real, hypo_rotate.imag)
        small.set_data((2*circle+hypo_rotate[resolution//4]).real, (2*circle+hypo_rotate[resolution//4]).imag)
        level.append(hypo_rotate[:-1])
        yield
    """

def no_op(*args, **kwargs):
    return


ani = FuncAnimation(fig, no_op, frames=frame_generator(), save_count=300, blit=False)

ani.save('4square.gif', writer='imagemagick', fps=1, dpi=75)
# plt.show()
"""
for f in frame_generator():
    pass

h_array = np.concatenate(level)
n = resolution
height = .2
points = [(point.real, point.imag, (i//n)*height) for i,point in enumerate(h_array)]
faces = [(i*n + j, (i+1)*n + (j+1)%n, (i+1)*n +j) for i in range(len(level)-1) for j in range(n)]
faces.extend([(i*n + j, i*n + (j+1)%n, (i+1)*n +(j+1)%n) for i in range(len(level)-1) for j in range(n)])
faces.append(list(range(n-1,0,-1)))
N = len(level)
faces.append(list(range((N-1)*n, N*n)))

piece = polyhedron(points=points, faces=faces)
# piece = union()(piece, translate([0,0,height*(N-1)])(piece))
print(scad_render(piece))
"""
