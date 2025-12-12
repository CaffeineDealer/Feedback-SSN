import numpy as np
import math
import random
import matplotlib.pyplot as plt

class Stim:

    def __init__(h,motion,dir,coh,dur):

        h.screensize = [1000,1000]
        h.dotsize = 1
        h.dotdensity = 0.01
        h.spd = 1000 #100
        h.dir = np.deg2rad(dir)
        h.aptdia = 500 #100
        h.aptlocation = [500,500]
        h.coherence = coh #0.5
        h.lifeTime = 100
        h.duration = dur #500
        h.framerate = 60 
        h.motion = motion
        h.w0 = 2

    def makeStim(h,pltflag):
        
        ndots = np.floor(h.aptdia * h.aptdia * h.dotdensity).astype(int) # number of dots
        pstart = [np.random.randint(0,h.aptdia,[ndots]) + h.aptlocation[0] - h.aptdia/2 , np.random.randint(0,h.aptdia,[ndots]) + h.aptlocation[1] - h.aptdia/2] # dots intial position
        lifestart = np.random.rand(1,ndots) * h.lifeTime
        mv = np.ones([2,ndots])
        mv[0] = mv[0] * h.spd
        mv[1] = mv[1] * h.dir
        nframe = np.floor(h.duration * h.framerate / 1000).astype(int)

        signalIdx = np.random.rand(ndots) < h.coherence
        noiseIdx = ~signalIdx

        dotpos = np.zeros([2,ndots,nframe])
        motvec = np.zeros([2,ndots,nframe])
        
        noiseDir = np.random.rand(ndots) * 2 * np.pi
        for framecount in range(nframe):
            if h.motion == 'simple':
                if framecount == 0:
                    dotpos[:,:,0] = pstart
                    dotlife = lifestart
                    motvec[:,:,0] = mv
                else:
                    dotpos[0,signalIdx,framecount] = np.floor(dotpos[0,signalIdx,framecount-1] + (1/h.framerate) * h.spd * np.cos(h.dir))
                    dotpos[1,signalIdx,framecount] = np.floor(dotpos[1,signalIdx,framecount-1] + (1/h.framerate) * h.spd * np.sin(h.dir))

                    dotpos[0,noiseIdx,framecount] = np.floor(dotpos[0,noiseIdx,framecount-1] + (1/h.framerate) * h.spd * np.cos(noiseDir[noiseIdx]))
                    dotpos[1,noiseIdx,framecount] = np.floor(dotpos[1,noiseIdx,framecount-1] + (1/h.framerate) * h.spd * np.sin(noiseDir[noiseIdx]))

                    dotlife = np.squeeze(dotlife + (1/h.framerate) * 1000)
                    motvec[0,:,framecount] = h.spd
                    motvec[1,signalIdx,framecount] = h.dir
                    motvec[1,noiseIdx,framecount] = noiseDir[noiseIdx]
            elif h.motion == 'complex':
                if framecount == 0:
                    dotpos[:,:,0] = pstart
                    dotlife = lifestart
                    motvec[:,:,0] = mv
                else:
                    u = h.w0 * ((dotpos[0,signalIdx,framecount-1] - h.aptlocation[0]) * np.cos(h.dir)
                                 - (dotpos[1,signalIdx,framecount-1] - h.aptlocation[1]) * np.sin(h.dir))
                    
                    v = h.w0 * ((dotpos[0,signalIdx,framecount-1] - h.aptlocation[0]) * np.sin(h.dir)
                                 + (dotpos[1,signalIdx,framecount-1] - h.aptlocation[1]) * np.cos(h.dir))
                                
                    dotpos[0,signalIdx,framecount] = np.floor(dotpos[0,signalIdx,framecount-1] + (1/h.framerate) * u)
                    dotpos[1,signalIdx,framecount] = np.floor(dotpos[1,signalIdx,framecount-1] + (1/h.framerate) * v)

                    unoise = h.w0 * ((dotpos[0,noiseIdx,framecount-1] - h.aptlocation[0] * np.cos(noiseDir[noiseIdx]))
                                 - (dotpos[1,noiseIdx,framecount-1] - h.aptlocation[1]) * np.sin(noiseDir[noiseIdx]))
                    
                    vnoise = h.w0 * ((dotpos[0,noiseIdx,framecount-1] - h.aptlocation[0] * np.sin(noiseDir[noiseIdx]))
                                 + (dotpos[1,noiseIdx,framecount-1] - h.aptlocation[1]) * np.cos(noiseDir[noiseIdx]))
                    
                    dotpos[0,noiseIdx,framecount] = np.floor(dotpos[0,noiseIdx,framecount-1] + (1/h.framerate) * unoise)
                    dotpos[1,noiseIdx,framecount] = np.floor(dotpos[1,noiseIdx,framecount-1] + (1/h.framerate) * vnoise)

                    dotlife = np.squeeze(dotlife + (1/h.framerate) * 1000)
                    u = np.array(u)
                    v = np.array(v)
                    unoise = np.array(unoise)
                    vnoise = np.array(vnoise)
                    angcheckSig = np.arctan2(u,v)
                    angcheckNoise = np.arctan2(unoise,vnoise)
                    conditionS1 = angcheckSig >= 0
                    conditionS2 = angcheckSig < 0
                    conditionN1 = angcheckNoise >= 0
                    conditionN2 = angcheckNoise < 0
                    motvec[0,signalIdx,framecount] = np.sqrt(u ** 2 + v ** 2) 
                    motvec[0,noiseIdx,framecount] = np.sqrt(unoise ** 2 + vnoise ** 2)
                    motvec[1,signalIdx,framecount] = np.arctan2(v,u) * conditionS1 + (np.arctan2(v,u) + 2 * np.pi) * conditionS2
                    motvec[1,noiseIdx,framecount] = np.arctan2(vnoise,unoise) * conditionN1 + (np.arctan2(vnoise,unoise) + 2 * np.pi) * conditionN2
                    
        # Relocate dead dots    
            deaddotidx = np.squeeze(dotlife > h.lifeTime)
            ndeaddot = sum(deaddotidx)
            dotpos[0,deaddotidx,framecount] = [np.array(random.sample(range(h.aptdia),ndeaddot)) + h.aptlocation[0] - h.aptdia/2]
            dotpos[1,deaddotidx,framecount] = [np.array(random.sample(range(h.aptdia),ndeaddot)) + h.aptlocation[1] - h.aptdia/2]
            if ndeaddot != 0:
                dotlife[deaddotidx] = 0

            if ndeaddot > 0:
                motvec[0,deaddotidx,framecount] = 0
                motvec[1,deaddotidx,framecount] = h.dir
            
            # Replace dots located on the boundry of apt
            currentdotpos = np.squeeze(dotpos[:,:,framecount])
            upidx = currentdotpos[1,:] > [h.aptlocation[1] + h.aptdia/2]
            downidx = currentdotpos[1,:] < [h.aptlocation[1] - h.aptdia/2]
            rightidx = currentdotpos[0,:] > [h.aptlocation[0] + h.aptdia/2]
            leftidx = currentdotpos[0,:] < [h.aptlocation[0] - h.aptdia/2]

            borderidx = upidx | downidx | rightidx | leftidx
            nborderidx = sum(borderidx)
            dotpos[:,borderidx,framecount] = [np.array(random.sample(range(h.aptdia),nborderidx)) + h.aptlocation[0] - h.aptdia/2,
                                                np.array(random.sample(range(h.aptdia),nborderidx)) + h.aptlocation[1] - h.aptdia/2]
            
            if nborderidx != 0:
                dotlife[borderidx] = 0
            
            dotpos = dotpos.astype(int)
        
        # fig = plt.figure(1)
        for i in range(nframe):
            I = np.zeros(h.screensize, dtype=np.int32)
            x = np.squeeze(dotpos[1,:,i])
            y = np.squeeze(dotpos[0,:,i])
            I[x,y] = 1
            if pltflag == 'on':
                plt.imshow(I, cmap='gray')
                plt.title(f"Frame {i + 1}")
                plt.pause(1/h.framerate)
                plt.clf()
            elif pltflag == 'off':
                print('Please wait')
                # plt.close()
        return dotpos,motvec
    
    def estimatefield(h,dotpos,motvec):
        mfratio = 1/10 # Motion Field Resolution Ratio
        mfsize = np.floor(np.array(h.screensize) * mfratio) # Motion Field Size

        avgmotAmp = np.zeros((int(mfsize[0]),int(mfsize[1])))
        avgmotAng = np.zeros((int(mfsize[0]),int(mfsize[1])))
        a = np.zeros((int(mfsize[0]),int(mfsize[1])))
        ii = 0
        for i in range(int(mfsize[0])):
            for j in range(int(mfsize[1])):
                llimitX = (i - 1) * (1/mfratio) + 1
                ulimitX = i * (1/mfratio)

                llimitY = (j - 1) * (1/mfratio) + 1
                ulimitY = j * (1/mfratio)

                dotinX = (dotpos[0,:,:] >= llimitX) & (dotpos[0,:,:] <= ulimitX) & (dotpos[1,:,:] >= llimitY) & (dotpos[1,:,:] <= ulimitY)
                dotinY = dotinX
                dotin = np.stack((dotinX,dotinY))
                
                motvecin = motvec[dotin]
                a[i,j] = motvecin.size

                if np.all(motvecin):
                    avgmotAmp[i,j] = 0
                    avgmotAng[i,j] = 0
                    
                else: 
                    motvecin2row = motvecin.reshape(2,motvecin.size//2)
                    avgmotX = np.sum(motvecin2row[0,:] * np.cos(motvecin2row[1,:]))
                    avgmotY = np.sum(motvecin2row[0,:] * np.sin(motvecin2row[1,:]))
                    avgmotAmp[i,j] = np.sqrt(avgmotX ** 2 + avgmotY ** 2) / (len(motvecin)/2)
                    avgmotAng[i,j] = np.arctan(avgmotY/avgmotX)
                    

                    if avgmotX == 0 and avgmotY == 0:
                        avgmotAng[i,j] = 0
                    if (avgmotX < 0 and avgmotY < 0) or (avgmotX < 0 and avgmotY > 0):
                        avgmotAng[i,j] += np.pi                       
        # np.savetxt('ang.txt',avgmotAng)
        # np.savetxt('amp.txt',avgmotAmp)
        x = np.arange(mfsize[1])
        y = np.arange(mfsize[0])
        X, Y = np.meshgrid(x,y)
 
        fig = plt.figure(2)
        plt.imshow((x,y), cmap='gray')
        U = avgmotAmp * np.sin(avgmotAng)
        V = avgmotAmp * np.cos(avgmotAng)
        plt.quiver(X, Y, U, -V,color='r') 
        h.Xr = np.reshape(np.log(avgmotAmp / 10), -1)
        h.Xt = np.reshape(avgmotAng, -1)
        h.maxspeed = np.float32(np.max(h.Xr[:]))
        np.savetxt('Xr.txt',h.Xr)
        np.savetxt('Xt.txt',h.Xt)   
        return avgmotAmp,avgmotAng
    
    def MTbank(h):
        Xt = np.float32(h.Xt)
        Xt = Xt[np.newaxis,:]
        Xr = np.float32(h.Xr)
        Xr = Xr[np.newaxis,:]
        maxspeed = h.maxspeed
        w = np.float32(np.sqrt(Xt.shape[1]))
        ns = int(np.round((w/2) ** 2))

        if np.all(maxspeed):
            maxspeed = np.max(Xr[:])            
        bwtheta = 1
        bwspeed = 1
        sigmas = np.array([8])
        xi, yi = np.meshgrid(np.arange(0,w), np.arange(0,w))
        xr = []
        yr = []
        iter = 0
        X = np.zeros((1,3,8,ns), dtype=np.float32) 
        filts = np.zeros((int(w ** 2), ns), dtype=np.float32)
        for j in range(int(w/2)):
            for i in range(int(w/2)):
                x0 = np.float32((i-1) * 2 + 1.5) 
                y0 = np.float32((j-1) * 2 + 1.5)
                filt = np.exp(-((xi - x0) ** 2 + (yi - y0) ** 2) / (2 * (sigmas ** 2)))
                filt = filt.reshape(filt.size,order='F')                    
                filts[:,iter] = filt
                xr.append(x0)
                yr.append(y0)
                iter += 1
        filts = filts/36

        spds = np.array([.8,2,3])
        themaxes = np.zeros(spds.shape[0])
        for l in range(0,3):
            prefspeed = spds[l] / 3 * maxspeed
            Xrs = np.arange(0,maxspeed,0.01)
            themaxes[l] = np.max(2 * (np.exp(-bwspeed**-2 / 2 * ((Xrs - prefspeed)**2)) 
                                      - np.exp(-bwspeed**-2 / 2 * ((Xrs + prefspeed)**2))))

        xs = np.zeros((3,8,ns))
        ys = np.zeros((3,8,ns))
        thetas = np.zeros((3,8,ns))
        speeds = np.zeros((3,8,ns))
        bwthetas = np.zeros((3,8,ns))
        bwspeeds = np.zeros((3,8,ns))
        sigmass = np.zeros((3,8,ns))
        kk = 1
        for k in range(0,8):
            preftheta = ((kk-1) / 4 * np.pi)
            kk += 1
            X2 = 1 / np.sinh(bwtheta) * (np.exp(bwtheta * np.cos(Xt - preftheta)) - 1)
            for l in range(0,3):
                prefspeed = spds[l] / 3 * maxspeed
                X1 = 2 * (np.exp(-bwspeed**-2 / 2 * ((Xr - prefspeed)**2)) 
                                      - np.exp(-bwspeed**-2 / 2 * ((Xr + prefspeed)**2)))
                Xf = X1 * X2 / themaxes[l]
                Xfnew = Xf[np.newaxis,:]
                X[:,l,k] = np.dot(Xfnew , filts)
                xs[l,k,:] = xr
                ys[l,k,:] = yr
                thetas[l,k,:] = preftheta
                speeds[l,k,:] = prefspeed
                bwthetas[l,k,:] = bwtheta
                bwspeeds[l,k,:] = bwspeed
                sigmass[l,k,:] = sigmas
        Xout = np.max(np.squeeze(X), axis=2)
        fig = plt.figure(3)
        plt.plot(Xout[0,:],'k')
        plt.plot(Xout[1,:],'r')
        plt.plot(Xout[2,:],'b')
        h.xs = xs
        h.ys = ys
        h.thetas = thetas
        h.speeds = speeds
        h.sigmass = sigmass
        h.bwspeeds = bwspeeds
        h.bwthetas = bwthetas
        return h,X
    
    def MSTbank(h,X,newmst,nonlinear):
        # MT-bank parameter
        MT = X
        MT = np.squeeze(MT)
        h.spatialWidth = np.sqrt(MT.shape[2]).astype(int)
        h.th = np.unique(h.thetas)
        h.numthetas = h.th.size
        h.spd = np.unique(h.speeds)
        h.numspeeds = h.spd.size
        # MST-bank parameter
        h.spatialSigma = 5
        h.numMSTunitPerPos = 20
        h.numMTsubPerMST = 5
        h.kernelW = 5
        h.strideSize = 4

        # Synpatic connection
        if newmst == 'active':
            SpatialPos, PrefSpd, PrefTheta, MT2MSTw = h.connectivity()

        thisMST = np.zeros((np.sqrt(MT.shape[2]).astype(int), np.sqrt(MT.shape[2]).astype(int), h.numMSTunitPerPos))
        for i in range(h.numMSTunitPerPos):
            thisW = MT2MSTw[i,:]
            thisMTsub = np.vstack(([PrefSpd[i,:],PrefTheta[i]]))
            thisSpatialPos = SpatialPos[i,:].astype(int)
            I, J = np.unravel_index(thisSpatialPos, (h.spatialSigma, h.spatialSigma))

            kernel = np.zeros([h.numspeeds,h.numthetas,h.spatialSigma,h.spatialSigma])
            for j in range(h.numMTsubPerMST):
                kernel[thisMTsub[0,j]-1,thisMTsub[1,j]-1,I[j],J[j]] = thisW[j]
            MTreshape = np.reshape(MT, (h.numspeeds, h.numthetas, h.spatialWidth, h.spatialWidth), order='F') 
            thisMST[:,:,i] = h.simpleNdConv(MTreshape, kernel,nonlinear) 

        # Max-pooling
        pooledMST = h.pooling(thisMST,h.spatialWidth, h.kernelW, h.strideSize)
        dwnsmpSpatialW = pooledMST.shape[0]
        MST = np.reshape(pooledMST, (dwnsmpSpatialW ** 2, h.numMSTunitPerPos), order='F').T
        # fig = plt.figure(4)
        # plt.imshow(MST,aspect='auto')
        # for i in range(thisMST.shape[2]):
        #     plt.imshow(np.squeeze(pooledMST[:,:,i]))
        #     plt.pause(0.5)
        return SpatialPos, PrefSpd, PrefTheta, MT2MSTw, MST
    
    def connectivity(h):
        SpatialPos = np.zeros([h.numMSTunitPerPos,h.numMTsubPerMST])
        for i in range(h.numMSTunitPerPos):
            SpatialPos[i,:] = np.random.permutation(h.spatialSigma ** 2)[:h.numMTsubPerMST]

        PrefSpd = np.random.randint(1, h.numspeeds + 1, (h.numMSTunitPerPos, h.numMTsubPerMST))
        PrefTheta = np.random.randint(1, h.numthetas + 1, (h.numMSTunitPerPos, h.numMTsubPerMST))
        MT2MSTw = np.random.rand(h.numMSTunitPerPos, h.numMTsubPerMST) - 0.2
        negwidx = np.where(MT2MSTw < 0)
        MT2MSTw[negwidx] = -np.random.rand(*negwidx[0].shape)
        poswidx = np.where(MT2MSTw >= 0)
        MT2MSTw[poswidx] = np.random.rand(*poswidx[0].shape)
        return SpatialPos, PrefSpd, PrefTheta, MT2MSTw

    def simpleNdConv(h,X,k,nonlinear):
        d1, d2, dx, dy = X.shape
        dk1, dk2, w, w = k.shape

        if d1 != dk1 or d2 != dk2:
            raise ValueError('Kernel and the array don not have the same speed and/or theta dimensions')
        
        padsize = np.ceil(w/2).astype(int)
        padX = np.pad(X, ((0, 0), (0, 0), (padsize, padsize), (padsize, padsize)), mode='constant')

        Y = np.zeros((dx, dy))
        for i in range(dx):
            for j in range(dy):
                 x = padX[:, :, i + padsize - np.floor(w / 2).astype(int):i + padsize + np.ceil(w / 2).astype(int), 
                  j + padsize - np.floor(w / 2).astype(int):j + padsize + np.ceil(w / 2).astype(int)]
                 if nonlinear == 1:
                     xnonlinear = np.maximum(x, 0) ** 0.08
                 elif nonlinear == 0:
                     xnonlinear = np.maximum(x, 0)
                 y = xnonlinear * k
                 Y[i,j] = np.sum(y)
        return Y
    
    def pooling(h, maps, spatialWidth, kernelW, strideSize):
        padsize = np.ceil(h.kernelW / 2).astype(int)
        maps = np.pad(maps, ((padsize, padsize), (padsize, padsize), (0,0)), mode='constant')
        pooledmap = np.zeros(((spatialWidth + padsize * 2 - kernelW) // strideSize + 1,
                          (spatialWidth + padsize * 2 - kernelW) // strideSize + 1, maps.shape[2]))
        k = 0
        for i in range(0,spatialWidth + padsize * 2 - kernelW + 1, strideSize):
            k += 1
            h = 0
            for j in range(0,spatialWidth + padsize * 2 - kernelW + 1, strideSize):
                h += 1
                thispart = maps[i:i+kernelW, j:j+kernelW, :]
                MAXpooled = np.max(np.max(thispart, axis=0), axis=0)
                MINpooled = np.min(np.min(thispart, axis=0), axis=0)
                MAXorMIN = np.abs(MAXpooled) > np.abs(MINpooled)
                pooledmap[k-1, h-1, MAXorMIN] = MAXpooled[MAXorMIN]
                pooledmap[k-1, h-1, ~MAXorMIN] = MINpooled[~MAXorMIN]
        return pooledmap
