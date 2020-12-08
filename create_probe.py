import numpy as np
import scipy.ndimage.filters as filters

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

