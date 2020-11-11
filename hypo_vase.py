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

k = 5
limit = k+2
bound = (-limit, limit)
ax.set_aspect('equal')
ax.set_xlim(*bound)
ax.set_ylim(*bound)

# big circle:
theta = np.linspace(0, 2*np.pi, resolution+1)
circle = np.exp(1j*theta)

# hypocycloid
hypo_inner = (k-1)*np.exp(1j*theta) + np.conj(np.exp(1j*(k-1)*theta))
hypo_outer = (k)*np.exp(1j*theta) + np.conj(np.exp(1j*(k)*theta))

big_circle, = ax.plot((k+1)*circle.real, (k+1)*circle.imag, 'b', linewidth=2)
hypocycloid_a, = ax.plot([], [], 'r-', linewidth=3.3)
hypocycloid_b, = ax.plot(hypo_outer.real, hypo_outer.imag, 'g-', linewidth=3.3)

level = []

def frame_generator(step=8):
    # Drawing path
    for i in range(0, len(theta), step):
        rotate = np.exp(1j*-theta[i]/k)
        hypo_rotate = rotate*hypo_inner + circle[i]
        hypocycloid_a.set_data(hypo_rotate.real, hypo_rotate.imag)
        level.append(hypo_rotate[:-1])
        yield


def no_op(*args, **kwargs):
    return


# ani = FuncAnimation(fig, no_op, frames=frame_generator(), save_count=300, blit=False)

# ani.save('hypo_vase.svg', writer='imagemagick', fps=10, dpi=75)
# plt.show()
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
piece = scale(10)(piece)
# piece = union()(piece, translate([0,0,height*(N-1)])(piece))
print(scad_render(piece))
"""
class Hypocycloid:

    def __init__(self, ratio=3, frames=100, ncycles=1):
        self.frames = frames
        self.ncycles = ncycles
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        plt.axis('off')


        ##big circle:
        theta = np.linspace(0, 2*np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)

        # self.white_circle, = self.ax.plot(1.0001*x, 1.0001*y, 'w-', linewidth=4)
        self.big_circle, = self.ax.plot(x, y, 'b', linewidth=2)
        # self.dot_circle, = self.ax.plot(x*(1-1/ratio), y*(1-1/ratio), color='gray', linestyle='dashed', linewidth=2)
        self.dot_circle, = self.ax.plot(x*(1-1/ratio), y*(1-1/ratio), 'k--', linewidth=1.5)

        ##small circle:
        self.small_r = 1./ratio
        r = self.small_r
        x = r*np.cos(theta)+1-r
        y = r*np.sin(theta)
        self.small_circle, = self.ax.plot(x,y,'k-',linewidth=2)
        self.small_circle2, = self.ax.plot(-x,-y,'k-',linewidth=2)

        ##line and dot:
        self.line, = self.ax.plot([1-r,1],[0,0],'k-',linewidth=2)
        self.line2, = self.ax.plot([],[],'k-',linewidth=2)
        self.dot, = self.ax.plot([1-r],[0], 'ko', ms=10)
        self.dot2, = self.ax.plot([1-r],[0], 'ko', ms=10)
        ##hypocycloid:
        self.hypocycloid, = self.ax.plot([],[],'r-',linewidth=3.3)
        self.hypocycloid2, = self.ax.plot([],[],'r-',linewidth=3.3)


        self.animation = FuncAnimation(
            self.fig, self.animate,
            frames=self.frames*self.ncycles,
            interval=50, blit=False,
            repeat_delay=2000,
        )

    def update_small_circle(self, phi):
        theta = np.linspace(0,2*np.pi,100)
        x = self.small_r*np.cos(theta)+(1-self.small_r)*np.cos(phi)
        y = self.small_r*np.sin(theta)+(1-self.small_r)*np.sin(phi)
        self.small_circle.set_data(x,y)
        self.small_circle2.set_data(-x,-y)


    def update_hypocycloid(self, phis):
        r = self.small_r
        x = (1-r)*np.cos(phis)+r*np.cos((1-r)/r*phis)
        y = (1-r)*np.sin(phis)-r*np.sin((1-r)/r*phis)
        self.hypocycloid.set_data(x,y)
        self.hypocycloid2.set_data(y,x)

        center = [(1-r)*np.cos(phis[-1]), (1-r)*np.sin(phis[-1])]

        self.line.set_data([center[0],x[-1]],[center[1],y[-1]])
        self.line2.set_data([center[0],-x[-1]],[center[1],-y[-1]])
        self.dot.set_data([center[0]], [center[1]])
        self.dot2.set_data([-center[0]], [-center[1]])

    def animate(self, frame):
        frame = frame+1
        phi = 2*np.pi*frame/self.frames
        self.update_small_circle(phi)
        self.update_hypocycloid(np.linspace(0,phi,frame))

hypo = Hypocycloid(ratio=5, frames=40, ncycles=1)

##un-comment the next line, if you want to save the animation as gif:
hypo.animation.save('hypo.gif', writer='imagemagick', fps=10, dpi=75)

# plt.show()
"""
