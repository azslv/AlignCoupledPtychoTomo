import numpy as np
import scipy.ndimage.filters as filters
import astra

def fourier(image):
    """
    2D discrete fourier transform.        
    Takes 1 argument:
        ND np.array 'image'
    
    Returns the fourier transform of the image in the two last dimensions.
    """
    
    f_image = np.fft.fftshift(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.fft.ifftshift(image,-2),-1)),-2),-1)
    f_image = f_image/(np.sqrt(f_image.shape[-1]*f_image.shape[-2]))
    return f_image

def i_fourier(image):
    """
    2D discrete inverse fourier transform.        
    Takes 1 argument:
        ND np.array 'image'
    
    Returns the fourier transform of the image in the two last dimensions.
    """
    
    if_image = np.fft.fftshift(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(np.fft.ifftshift(image,-2),-1)),-2),-1)
    if_image = if_image*(np.sqrt(if_image.shape[-1]*if_image.shape[-2]))
    return if_image

def mask_circle(radius,n):
    """
    makes a 2D array shaped as sircle size (n,n,n) with radius 'radius'.
    """
    #initialize X,Y and r as euclidian distance from middle
    x = np.arange(n)-int(n/2)
    y = np.arange(n)-int(n/2)
    (Y, X) = np.meshgrid(y, x)
    r = np.sqrt(X**2 + Y**2)
    
    #circle is where 
    obj = r <= int(radius)

    return obj

def create_probe(size,sigma=5,ph_multi=20,ph_exp=2,i_four=False,mask_diameter_ratio=1):
    """
    Probe creation function. Feel free to play around with this function as it is rather arbitrary.
    The function creates a probe which is the sum of a gaussian with std 'sigma' and
    some complex wave proportional to the gaussian and parameters 'ph_multi' for phase multiplier and 'ph_exp' for the exponent.
    """
    #parameters
    flux = 1e8
    expTime = 0.1
    
    totalPhotons = flux*expTime
    
    #needs to be initialized as a deltafunction as we are using a filter to create the gaussian.
    deltafunc = np.zeros(size)
    deltafunc[int(np.ceil(size[0]/2)),int(np.ceil(size[1]/2))] = 1
    
    #dtype is set to complex.
    probe = np.array(filters.gaussian_filter(deltafunc,size[0]/sigma))
    #Adding structure as discussed.
    probe = probe**(1/2)
    
    #normalization 0-1
    probe = probe/np.max(probe)
    
    probe = probe**2*np.exp(ph_multi*1j*probe**ph_exp)
    #here you can optionally add the inverse fourier transform to the probe. We have found this gives good results.
    if i_four:
        probe = i_fourier(probe) #+ probe
    
    #A circualr mask is added around the probe to make it positionally well defined.
    probe = probe*mask_circle(size[0]*0.5*mask_diameter_ratio,size[0])
    
    #normalization to flux of photons
    probe = probe*np.sqrt(totalPhotons)/np.linalg.norm(probe)
    
    return probe

def mis_align(parameters0,mean=0,sigma=1):
    
    parameters = np.zeros(parameters0.shape)
    parameters[:,0] = parameters0[:,0]
    
    parameters[:,1] = parameters0[:,1] + np.random.normal(mean,sigma,(parameters0[:,1].shape))
    parameters[:,2] = parameters0[:,2] + np.random.normal(mean,sigma,(parameters0[:,2].shape))
    
    return parameters

class Ptycho3D:
    
    """Used for 3D ptychography general functions for reconstructions and simulation of data."""
    
    def __init__(self,probe_size,vol_shape,parameters):
        """
        Takes 4 arguments: 
            int 'probe_size'
            tuple 'vol_shape'
            1D np.array 'angles'
            2xN np.array 'scanning positions' (coordinates for the scanning: first row is x-coordinates and second is y-coordinates.)
        """
        self.fourier = fourier
        self.i_fourier = i_fourier
        self.parameters = parameters
        self.vol_shape = vol_shape
        self.probe_size = probe_size
        self.diff_pat = []

    def radon3D(self,volume):
        """
        Makes the 3D radon tranformation of a volume to return the projections.
        Takes one argument: 
            3D np.array 'volume'
        
        Returns frames, which is a list containing the projections in the last two dimensions.
        """
        (M,N,N) = volume.shape
        self.vol_shape = (M,N,N)
        parameters = self.parameters
        
        #initialize vectors
        vectors = np.zeros((np.shape(parameters)[0],12))
        vectors[:,0]=np.sin(parameters[:,0])                                               #rayX
        vectors[:,1]=-np.cos(parameters[:,0])                                              #rayY
        vectors[:,2]=0                                                                     #rayZ
        vectors[:,3]=np.multiply(parameters[:,1],np.cos(parameters[:,0]))                  #dX
        vectors[:,4]=np.multiply(parameters[:,1],np.sin(parameters[:,0]))                  #dY
        vectors[:,5]=parameters[:,2]                                                       #dZ
        vectors[:,6]=np.cos(parameters[:,0])                                               #uX
        vectors[:,7]=np.sin(parameters[:,0])                                               #uY
        vectors[:,8]=0                                                                     #uZ
        vectors[:,9]=0                                    #vX
        vectors[:,10]=0                                   #vY
        vectors[:,11]=1                                   #vZ
        
        #create volume geometry with astra
        vol_geom = astra.create_vol_geom(N,N,M) 
        proj_geom = astra.create_proj_geom('parallel3d_vec',self.probe_size[0],self.probe_size[0],vectors)
        
        
        #create the 3D sinogram for imaginary and real part independently
        [_,frames_r] = astra.create_sino3d_gpu(np.real(volume),proj_geom,vol_geom)
        [_,frames_i] = astra.create_sino3d_gpu(np.imag(volume),proj_geom,vol_geom)
        
        frames = frames_r+1j*np.array(frames_i,dtype=complex)
        
        #transpose because astra
        frames = frames.transpose((1,2,0))
        
        astra.clear()
        return frames
    
    def back_radon3D(self,frames):
        """
        Opposite of radon3D. Does the backprojection.
        Takes one argument: 
            3D np.array 'frames_p' (a stack of the frames that are used to compute the volume)      
        Returns volume
        """
        (M,N,N) = self.vol_shape
        parameters = self.parameters
        
        #initialize vectors
        vectors = np.zeros((np.shape(parameters)[0],12))
        vectors[:,0]=np.sin(parameters[:,0])                                               #rayX
        vectors[:,1]=-np.cos(parameters[:,0])                                              #rayY
        vectors[:,2]=0                                                                     #rayZ
        vectors[:,3]=np.multiply(parameters[:,1],np.cos(parameters[:,0]))                  #dX
        vectors[:,4]=np.multiply(parameters[:,1],np.sin(parameters[:,0]))                  #dY
        vectors[:,5]=parameters[:,2]                                                       #dZ
        vectors[:,6]=np.cos(parameters[:,0])                                               #uX
        vectors[:,7]=np.sin(parameters[:,0])                                               #uY
        vectors[:,8]=0                                                                     #uZ
        vectors[:,9]=0                                    #vX
        vectors[:,10]=0                                   #vY
        vectors[:,11]=1                                   #vZ
        
        #create volume geometry with astra
        vol_geom = astra.create_vol_geom(N,N,M) 
        proj_geom = astra.create_proj_geom('parallel3d_vec',self.probe_size[0],self.probe_size[0],vectors)
        
        #revert frames back in to (N,n_angles,M)
        frames = np.transpose(frames,(2,0,1))
        
        #do the transform for real part and imaginary part independently
        (_,volume_r) = astra.create_backprojection3d_gpu(np.real(frames),proj_geom,vol_geom)
        (_,volume_i) = astra.create_backprojection3d_gpu(np.imag(frames),proj_geom,vol_geom)
        
        volume = volume_r + 1j*np.array(volume_i,dtype=complex)
        
        astra.clear()
        return volume
    
    def probe_object(self,scans,probe):
        """
        Mutiply each individual scan by probe function
        """
        PO = np.zeros([(np.shape(self.parameters)[0]),self.probe_size[0],self.probe_size[1]],dtype=complex)
        
        for i,scan in enumerate(scans):
          
            PO[i] = scan*probe       
        
        return PO
    
    def norm(self,vf):
        """
        Takes the L_2 norm of an array.
        """
        n = np.sqrt(np.inner(np.conj(np.complex128(vf)),np.complex128(vf)))
        
        return n
        
class Ptycho3D_RI:
    
    """Used for 3D ptychography using levenberg marquardt simulation of data with the refractive index model"""
    def __init__(self,ptycho,p_opti = False):
        self.Ptycho3D = ptycho
        if p_opti:
            self.noise_opti = 1/np.sqrt(self.Ptycho3D.diff_pat+np.min(self.Ptycho3D.diff_pat[np.nonzero(self.Ptycho3D.diff_pat)]))
        else:
            self.noise_opti = 1
        

        
    def forward(self,vol,probe,k):
        """
        The forward model for 3D-ptycography. Used to simulate data for testing reconstruction. 
        
        Takes two arguments: 
            3D np.array 'vol' (volume with values: 1 - refractive indexes)
            integer 'k' (wavenumber with correction for pixel size)
            
        Returns the resulting stack of stacks of diffraction patterns as a 4D np.array (nTheta,nScans,[img_size]).
        """
            
        #multiply the probe function with the object function - which in turn includes the radon3D of at the scanning locations
        diff_pat = self.Ptycho3D.probe_object(np.exp(1j*k*self.Ptycho3D.radon3D(vol)),probe)
        #now 'PO' is a 4D array with dimensions (nTheta,nScans,[img_size])
        
        #do the discrete 2D fourier transform over the two last dimensions (default):
        diff_pat = self.Ptycho3D.fourier(diff_pat)
        
        #the resulting diffraction pattern of intensities is then the norm square
        diff_pat = np.abs(diff_pat)**2
        #diff_pat = f_PO
                            
        return diff_pat
    
    def jacobian(self,vol,probe,hn,k,J_c = []):
        """
        The jacobian of the forward model for 3D-ptycography. 
        
        Takes three or four arguments: 
            3D np.array 'vol' (volume with values: 1 - refractive indexes)
            3D np.array 'hn0' (change in volume)
            integer 'k' (wavenumber with correction for pixel size)
            3D np.array 'J_c' which is the constant part of J in some cases ( conj(fourier{P*S{exp(ikn')}}) )
        
        Returns the resulting projections as an np.array.
        """
        if not J_c.any():
            J_c = np.conj(self.Ptycho3D.fourier(self.Ptycho3D.probe_object(np.exp(1j*k*self.Ptycho3D.radon3D(vol)),probe)))
        
        J_scans = self.noise_opti*2*np.real(J_c*self.Ptycho3D.fourier(self.Ptycho3D.probe_object((np.exp(1j*k*self.Ptycho3D.radon3D(vol))*self.Ptycho3D.radon3D(hn)*1j*k),probe)))
        
        return np.nan_to_num(J_scans)

    def jacobian_adjoint(self,vol,probe,hng,k,Ja_c = [],Ja_c2 = []):
        """
        The adjoint of the jacobian of the forward model for 3D-ptychography. 
        
        Takes three or five arguments: 
            3D np.array 'vol' (1 - refractive index)
            3D np.array 'hng' (change in volume)
            integer 'k' (wavenumber with correction for pixel size)
            4D np.array 'Ja_c' which is the constant part of Ja in some cases ( fourier{P*S{exp(ikn')}} )
            4D np.array 'Ja_c2' which is another constant part of Ja in some cases ( conj{ikP*S{exp(ik*R(vol))}} )            
        
        Returns the resulting image as an np.array.
        """
        if not Ja_c.any():
            Ja_c = self.Ptycho3D.fourier(self.Ptycho3D.probe_object(np.exp(1j*k*self.Ptycho3D.radon3D(vol)),probe))
        
        if not Ja_c2.any():
            Ja_c2 = np.conj(self.Ptycho3D.probe_object(np.exp(1j*k*self.Ptycho3D.radon3D(vol)),1j*k*probe))
        
        JA_image = 2*self.Ptycho3D.back_radon3D(Ja_c2*self.Ptycho3D.i_fourier(Ja_c*hng.real*self.noise_opti))
    
        return np.nan_to_num(JA_image)
    
    def jacobian_probe(self,vol,probe,hp,k,J_c = [],obj = []):
        """
        The jacobian of the forward model for 3D-ptycography with respect to the probe.
        
        Takes three or five arguments: 
            3D np.array 'vol' (volume with values: 1 - refractive indexes)
            3D np.array 'hp' (change in probe)
            integer 'k' (wavenumber with correction for pixel size)
            3D np.array 'J_c' which is the constant part of J in some cases ( conj(fourier{P*S{exp(ikn')}}) )
            3D np.array 'obj' which is the object 
            
        Returns the resulting projections as an np.array.
        """
        if not J_c.any():
            J_c = np.conj(self.Ptycho3D.fourier(self.Ptycho3D.probe_object(np.exp(1j*k*self.Ptycho3D.radon3D(vol)),probe)))

        if not obj.any():
            obj = np.exp(1j*k*self.Ptycho3D.radon3D(vol))

        J_probe = self.noise_opti*2*np.real(J_c*self.Ptycho3D.fourier(self.Ptycho3D.probe_object(obj,hp)))
        
        return J_probe
    
    def jacobian_probe_adjoint(self,vol,probe,hpg,k,Ja_c = [],obj_p = []):
        """
        The adjoint of the jacobian of the forward model for 3D-ptychography w.r.t the probe. 
        
        Takes three or five arguments: 
            3D np.array 'vol' (1 - refractive index)
            3D np.array 'hng' (change in volume)
            integer 'k' (wavenumber with correction for pixel size)
            4D np.array 'Ja_c' which is the constant part of Ja in some cases ( fourier{P*S{exp(ikn')}} )        
            3D np.array 'obj' which is the object
            
        Returns the resulting image as an np.array.
        """
        if not Ja_c.any():
            Ja_c = self.Ptycho3D.fourier(self.Ptycho3D.probe_object(np.exp(1j*k*self.Ptycho3D.radon3D(vol)),probe))
        
        if not obj_p.any():
            obj_p = np.exp(1j*k*self.Ptycho3D.radon3D(vol))

        JA_probe = 2*np.conj(obj_p)*self.Ptycho3D.i_fourier(Ja_c*hpg.real*self.noise_opti)
        JA_probe = np.sum(np.sum(JA_probe,axis=0),axis=0)
        return JA_probe