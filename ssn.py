import numpy as np
import math
import matplotlib.pyplot as plt



def init(efb,ifb,d):
    params = {'efb':efb,'ifb':ifb,'d':d}
    return params

def angdiff(a1,a2):
    a1 = np.rad2deg(np.array(a1))
    a2 = np.rad2deg(np.array(a2))
    diff = abs(a1 - a2)
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            if diff[i,j] > 180:
                diff[i,j] = 360 - diff[i,j]
    diff = np.deg2rad(diff)
    return diff

params = init(1,1,1)
targetExc = params['efb']
targetInh = params['ifb']
d = params['d']

grid = 360
th_size = 360
theta = np.zeros([1,th_size])
theta = np.deg2rad(np.linspace(0,359,th_size))
deltatheta = np.deg2rad(1)
[X,Y] = np.meshgrid(theta,theta)
minD = abs(angdiff(Y,X))
#plt.imshow(minD)
#plt.show()

Jee = 0.044
Jii = 0.018
Jei = 0.023
Jie = 0.042
sigmadir = np.deg2rad(64)
kappaE = .1
kappaI = .2

if Jei*Jie > Jee*Jii:
    print('Stable')
else:
    print('Unstable')

if Jii-Jei < 0 and Jii-Jei < Jie-Jee:
    print('Nonlinearity Satisfied!')
else:
    print('Not Nonlinear enough :(')

GsigmaDirE = kappaE * np.exp(-(np.square(minD)) / (2 * np.square(sigmadir)))
GsigmaDirI = kappaI * np.exp(-(np.square(minD)) / (2 * np.square(sigmadir)))


Wee = np.random.rand(grid,grid)
Wee[Wee < GsigmaDirE] = 1
Wee[np.not_equal(Wee,1)] = 0
Wee = Wee * np.random.normal(Jee,0.25*Jee,[grid,grid])
cWee = Jee * sum(GsigmaDirE,2)
aWee = sum(Wee,2)
sWee = cWee / aWee
sWee[np.isinf(sWee==1)] = 0
sWee = np.expand_dims(sWee,1)
Wee = Wee * np.tile(sWee,360)
plt.imshow(Wee)
plt.show()