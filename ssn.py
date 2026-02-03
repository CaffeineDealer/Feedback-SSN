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
theta = np.deg2rad(np.linspace(0,359,th_size))
deltatheta = np.deg2rad(1)
[X,Y] = np.meshgrid(theta,theta)
minD = abs(angdiff(Y,X))

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

# Wee - Excitatory to Excitatory
Wee = np.random.rand(grid,grid)
Wee[Wee < GsigmaDirE] = 1
Wee[np.not_equal(Wee,1)] = 0
Wee = Wee * np.random.normal(Jee,0.25*Jee,[grid,grid])
cWee = Jee * np.sum(GsigmaDirE, axis=1)
aWee = np.sum(Wee, axis=1)
sWee = cWee / aWee
sWee[np.isinf(sWee)] = 0
sWee = np.expand_dims(sWee,1)
Wee = Wee * np.tile(sWee, (1, 360))

# Wei - Excitatory to Inhibitory
Wei = np.random.rand(grid,grid)
Wei[Wei < GsigmaDirI] = 1
Wei[np.not_equal(Wei,1)] = 0
Wei = Wei * np.random.normal(Jei,0.25*Jei,[grid,grid])
cWei = Jei * np.sum(GsigmaDirI, axis=1)
aWei = np.sum(Wei, axis=1)
sWei = cWei / aWei
sWei[np.isinf(sWei)] = 0
sWei = np.expand_dims(sWei,1)
Wei = Wei * np.tile(sWei, (1, 360))

# Wie - Inhibitory to Excitatory
Wie = np.random.rand(grid,grid)
Wie[Wie < GsigmaDirE] = 1
Wie[np.not_equal(Wie,1)] = 0
Wie = Wie * np.random.normal(Jie,0.25*Jie,[grid,grid])
cWie = Jie * np.sum(GsigmaDirE, axis=1)
aWie = np.sum(Wie, axis=1)
sWie = cWie / aWie
sWie[np.isinf(sWie)] = 0
sWie = np.expand_dims(sWie,1)
Wie = Wie * np.tile(sWie, (1, 360))

# Wii - Inhibitory to Inhibitory
Wii = np.random.rand(grid,grid)
Wii[Wii < GsigmaDirI] = 1
Wii[np.not_equal(Wii,1)] = 0
Wii = Wii * np.random.normal(Jii,0.25*Jii,[grid,grid])
cWii = Jii * np.sum(GsigmaDirI, axis=1)
aWii = np.sum(Wii, axis=1)
sWii = cWii / aWii
sWii[np.isinf(sWii)] = 0
sWii = np.expand_dims(sWii,1)
Wii = Wii * np.tile(sWii, (1, 360))

# Weight Visualization
fig, axes = plt.subplots(2, 2, figsize=(8, 5))
# Wee
im0 = axes[0, 0].imshow(Wee, cmap='viridis', aspect='auto')
axes[0, 0].set_title('Excitatory-to-Excitatory (Wee)')
axes[0, 0].set_xlabel('Grid')
axes[0, 0].set_ylabel('Grid')
plt.colorbar(im0, ax=axes[0, 0], label='Weight')

# Wei
im1 = axes[0, 1].imshow(Wei, cmap='viridis', aspect='auto')
axes[0, 1].set_title('Excitatory-to-Inhibitory (Wei)')
axes[0, 1].set_xlabel('Grid')
axes[0, 1].set_ylabel('Grid')
plt.colorbar(im1, ax=axes[0, 1], label='Weight')

# Wie
im2 = axes[1, 0].imshow(Wie, cmap='viridis', aspect='auto')
axes[1, 0].set_title('Inhibitory-to-Excitatory (Wie)')
axes[1, 0].set_xlabel('Grid')
axes[1, 0].set_ylabel('Grid')
plt.colorbar(im2, ax=axes[1, 0], label='Weight')

# Wii
im3 = axes[1, 1].imshow(Wii, cmap='viridis', aspect='auto')
axes[1, 1].set_title('Inhibitory-to-Inhibitory (Wii)')
axes[1, 1].set_xlabel('Grid')
axes[1, 1].set_ylabel('Grid')
plt.colorbar(im3, ax=axes[1, 1], label='Weight')

plt.tight_layout()
plt.show()