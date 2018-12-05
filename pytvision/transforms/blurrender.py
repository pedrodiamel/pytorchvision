
# Render
# Implement the render function for blur problems

import random
import numpy as np
import cv2
from scipy.ndimage import rotate as imrotate
import skimage

from . import functional as F



def blurring(y, h, lambda_=1.0, sigma_=1.0):
    '''
    Blurring image 
    '''

    h = h.astype(float)/np.sum(h)
    y = y.astype(float)/255.0

    hY,wY = y.shape;
    hH,wH = h.shape;

    #y = y*lambda_ 

    # padding kernel
    hh = np.zeros((hY,wY))
    hh[ :hH, :wH ] = h;

    cy = -np.round(float(hH-1)/2.0).astype(int);
    cx = -np.round(float(wH-1)/2.0).astype(int);
    hh = np.roll( hh, shift=(cy,cx), axis=(0,1)) 

    H = np.fft.fft2(hh);
    Y = np.fft.fft2(y);

    yhat = np.fft.ifft2(Y*H).real;   
    
    return yhat



class BlurRender(object):
    

    def __init__(self, 
        pSFsize=64,
        maxTotalLength=64,
        anxiety=0.005,
        numT=2000,
        texp=0.75,
        lmax=100
        ):
        ''' Initialization '''

        self.pSFsize = pSFsize;
        self.maxTotalLength = maxTotalLength;
        self.anxiety = anxiety;
        self.numT = numT;    
        self.texp = texp;
        self.lmax = lmax;
    

    def generatelineal( self, image ):
        '''
        Render lineal blur
         :image: imagen original sin blur
         :lmax: maxima length
         :return: imagen con blur y coeficiente
        '''

        # generate parameters 
        length = int(self.lmax * random.random())
        angle  = 360.0*random.random();        
        # estimate kernel
        psf = self._motionlinalkernel(length=length, angle=angle);
        # blur
        imblur = cv2.filter2D(image, -1, psf )
        coef = length/self.lmax

        return imblur, coef

       


    def generatecurve( self, image ):
        '''
        Render curve blur
         :image: imagen original sin blur
         :return: imagen con blur y coeficiente
        '''

        # create kerners 
        pSFsize = self.pSFsize;
        maxTotalLength = self.maxTotalLength;
        anxiety = self.anxiety;
        numT = self.numT;    
        texp = self.texp;
        
        # create trajectoria 2d
        x, l = self._trajetory_2d( trajSize=pSFsize, anxiety=anxiety, numT=numT, maxTotalLength=maxTotalLength )
        # create kernel
        psf, power = self._motionnolinealkernel(x, texp=texp, pSFsize=pSFsize)                
        # blur 
        imblur = cv2.filter2D(image, -1, psf )            
        # noise
        imnoise = F.gaussian_noise(imblur, sigma=0.001)
                
        # metric reference
        psnr =  self._psnr(image, imnoise);
        
        # coneficient
        coef = (l*power)/(l+power);
        
        return imnoise, psnr, coef


    ## motion blur

    # Motion linear kernel
    def _motionlinalkernel(self, length=9, angle=0):
        '''
        Motion Linal Kernel
        :length 
        :angle 
        '''

        sz = (length + 1, length + 1)
        f = np.zeros(sz, dtype=float)
        f[ int(length / 2), 0:length + 1] = 1;
        f = imrotate(f, angle)
        
        f = f/np.sum(f)
        return f

    # Motion kernel
    def _motionnolinealkernel(
        self,
        x,
        pSFsize = 64,
        texp = 0.01
    ):
        '''
        Motion kernel
        :x: trajetory
        :psfsize: psf size mxn
        :texp: exposition time 
        '''  

        numt = len(x)    
        pSFsize = (pSFsize, pSFsize)   

        # center
        x = x - np.mean(x) + (pSFsize[1] + 1j * pSFsize[0] + 1 + 1j) / 2

        # initialize psf
        PSF = np.zeros(pSFsize)
        M = np.zeros(pSFsize)
        
        def ftriangle(d): return np.maximum(0, (1 - np.abs(d)));
        def ftriangle_prod(d1, d2): return ftriangle(d1) * ftriangle(d2);
        prevT = 0

        # sample the trajectory until time T
        for t in range( numt ):
            
            t_proportion = 0            
            if (texp * numt >= (t+1)) and (prevT * numt < t): t_proportion = 1
            elif (texp * numt >= t) and (prevT * numt < t): t_proportion = (texp * numt) - (t)
            elif (texp * numt >= (t+1)) and (prevT * numt < (t+1)): t_proportion = (t+1) - (prevT * numt)
            elif (texp * numt >= t) and (prevT * numt < (t+1)): t_proportion = (texp - prevT) * numt  

            m2 = int(np.minimum(pSFsize[1] - 1, np.maximum(1, np.floor ( np.real(x[t] )))))
            M2 = m2 + 1
            m1 = int(np.minimum(pSFsize[0] - 1, np.maximum(1, np.floor( np.imag(x[t] )))))
            M1 = m1 + 1

            a1 = t_proportion * ftriangle_prod(np.real(x[t]) - m2, np.imag(x[t]) - m1)
            a2 = t_proportion * ftriangle_prod(np.real(x[t]) - M2, np.imag(x[t]) - m1)
            a3 = t_proportion * ftriangle_prod(np.real(x[t]) - m2, np.imag(x[t]) - M1)
            a4 = t_proportion * ftriangle_prod(np.real(x[t]) - M2, np.imag(x[t]) - M1)

            PSF[m1-1, m2-1] += a1
            PSF[m1-1, M2-1] += a2
            PSF[M1-1, m2-1] += a3
            PSF[M1-1, M2-1] += a4

            M[m1-1, m2-1] = np.maximum(M[m1-1, m2-1], a1)
            M[m1-1, M2-1] = np.maximum(M[m1-1, M2-1], a2)
            M[M1-1, m2-1] = np.maximum(M[M1-1, m2-1], a3)
            M[M1-1, M2-1] = np.maximum(M[M1-1, M2-1], a4)  


        PSF = PSF/numt
        if np.sum(PSF):
            PSF = PSF/(np.sum(PSF) )
            PSF[pSFsize[0]//2,pSFsize[1]//2]=1
        
        power = np.sum(M)

        return PSF, power
    


    ## Trajetory
    # http://home.deib.polimi.it/boracchi/Projects/PSFGeneration.html
    # https://github.com/handong1587/PSF_generation
    # http://home.deib.polimi.it/boracchi/docs/2012_03_Boracchi-Foi-TIP_Modeling.pdf
    # https://arxiv.org/pdf/1612.02583.pdf

    def _trajetory_2d(
        self,
        trajSize=64, 
        anxiety=0.1,
        numT=2000,
        maxTotalLength=60    
    ):
        '''
        Create trajectory
        create trajetory for calculate the psf
        :trajSize: trajetory size
        :anxiety: determining the amount of shake
        :numT: number of samples where the Trajectory is sampled
        :maxTotalLength: maximum length of the trajectory computed as the sum of all distanced between consecuive points
        '''
        # Parameter determining the amount of shake (in the range [0,1] 0 corresponds to rectilinear trajectories)
        anxiety = anxiety * random.random()

        # Initialization
        length = 0
        abruptShakesCounter = 0
        centripetal = 0.7 * random.random()    
        # term determining, at each sample, the random component of the new direction
        gaussianTerm = 10.0 * random.random()
        # probability of having a big shake, e.g. due to pressing camera button or abrupt hand movements
        freqBigShakes = 0.2 * random.random()
        # Generate x(t), Discrete Random Motion Trajectory  in Continuous Domain
        # v is the initial velocity vector, initialized at random direction
        init_angle = 360.0 * random.random()
        init_angle *= np.pi/180.0;

        # initial velocity vector having norm 1
        v0 = np.cos(init_angle) + 1j * np.sin(init_angle)    
        # the speed of the initial velocity vector
        v = v0 * maxTotalLength / (numT - 1)
        if anxiety > 0: v = v0 * anxiety    

        # initialize the trajectory vector
        x = np.zeros((numT, 1), dtype=np.complex)

        for t in range(numT-1):           

            # determine if there is an abrupt (impulsive) shake
            if random.random() < freqBigShakes * anxiety:
                # if yes, determine the next direction which is likely to be opposite to the previous one
                nextDirection = 2 * v * (np.exp(1j * (np.pi + (np.random.rand(1) - 0.5))))
                abruptShakesCounter = abruptShakesCounter + 1
            else:
                nextDirection = 0            

            # determine the random component motion vector at the next step
            dv = nextDirection + anxiety * (gaussianTerm * (random.random() + 1j * random.random() ) - centripetal * x[t]) * (maxTotalLength / (numT - 1))
            v = v + dv

            # update particle position
            x[t + 1] = x[t] + v
            # compute total length
            length += abs(x[t + 1] - x[t])[0]


        # Center the Trajectory
        # set the lowest position in zero
        x = x - 1j * np.min(np.imag(x)) - np.min(np.real(x))       

        # center the Trajectory
        x = x - 1j * np.remainder(np.imag(x[0]), 1) - np.remainder(np.real(x[0]), 1) + 1 + 1j
        x = x + 1j * np.ceil((trajSize - np.max(np.imag(x))) / 2) + np.ceil((trajSize - np.max(np.real(x))) / 2)
    
        return x, length



    ## metrics 

    def _desviation(self, X, texp ):
        '''
        Mide la desviation de la trajetoria 
        :X: coordenadas cartecianas de la trayectoria 
        
        Paper:
        Modeling the Performance of Image Restoration from Motion Blur
        http://home.deib.polimi.it/boracchi/docs/2012_03_Boracchi-Foi-TIP_Modeling.pdf
        '''
        
        # Ecuacion 20
        # (Cl, Cs) = \sqrt Eig( Cov(ht/texp) ) 
        Cov = np.cov(X.T/texp,ddof=0);
        Eig = np.linalg.eigvals( Cov )
        (Cl, Cs) = np.sort( np.sqrt( Eig ) ) 

        return Cl, Cs


    def _harry(self, Cl, Cs, alpha=0.06):
        return Cl*Cs + alpha*(Cl+Cs)**2
    
    def _brow(self, Cl, Cs):
        return Cl*Cs/(Cl+Cs)

    def _shi(self, Cl, Cs):
        return np.maximum(Cl,Cs); # Cs


    ## metricas referenciales 

    def _isnr(self, original, noisy, restore ):
        '''
        ISNR
        Improvement in Signal to Noise Ratio
        '''        
        return 10.0 * np.log10( F.norm_fro(original,noisy)/F.norm_fro(original,restore))


    def _psnr(self, original, restore):
        '''
        PSNR
        ''' 

        # c = 1;
        # if len(original.shape)==3: c=3;
        # mse = F.norm_fro(original, restore)/np.prod(original.shape) ;
        # return 10.0 * np.log10( 255*255.0*c/ mse )

        #c = 1;
        #if len(original.shape)==3: c=3;
        #mse = F.norm_fro(original.astype('float'), restore.astype('float'))/np.prod(original.shape) ;
        #if mse==0: return 200
        #pixel_max= (2 ** (8*c)) -1
        #return 20.0 * np.log10( pixel_max / np.sqrt(mse) )

        return skimage.measure.compare_psnr(original, restore)
    
