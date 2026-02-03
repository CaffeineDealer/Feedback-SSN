#%%
import numpy as np
from setup import Stim
from polarmosaic import polarmosaic

dir = np.arange(0, 360, 45)
R = np.zeros([dir.size])
for i in range(dir.size):
    s = Stim('complex',dir[i],1,500) 
    o1, o2 = Stim.makeStim(s,'off')
    mAmp, mAng = Stim.estimatefield(s,o1,o2) 
    s, X = Stim.MTbank(s)
    X = np.maximum(X,0)
    nonlinear = 1 # ON=1 OFF=0
    SpatialPos, PrefSpd, PrefTheta, MT2MSTw, MST = Stim.MSTbank(s,X,'active',nonlinear)
    R[i] = np.max(MST[0,:])
#%%
baseline = np.min(R)  
themax = np.max(R)  
rg = np.array([baseline - (themax - baseline),themax])
polarmosaic(R,rg,1,.35) 
