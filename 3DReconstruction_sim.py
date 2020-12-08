from ptycho3D import Ptycho3D, Ptycho3D_RI, create_probe, mis_align
import numpy as np
import scipy.io as io
import time
from skimage.transform import resize
from pathlib import Path
import os
import matplotlib.pyplot as plt

from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph.output import GraphvizOutput

config = Config(max_depth=5)
graphviz = GraphvizOutput(output_file='filter_max_depth_1.png')

update_probe = False
noise = 'None'
musave = True
#used to time the initialization
tht = time.time()

#loadinig custom volume to simulate data
volume = io.loadmat('layer_het300.mat')['volume']
binn = 1
volume = volume[::binn, ::binn, ::binn]

volume = -1e-5*volume + 1e-7j*volume #phrased as a complex refractive index

vol_shape=volume.shape
(M,N,N) = vol_shape
probe_size = (100, 100)
probe0 = create_probe(probe_size)

#define the scanning positions
nx = 10
ny = 10

angles = np.linspace(0,np.pi,180)
angles_param = np.repeat(angles,nx*ny)

parameters = np.zeros([len(angles)*nx*ny,3])
parameters[:,0] = angles_param

np.random.seed(1234)

for i in range(len(parameters[:,0])):
    parameters[i,1] = (np.random.rand(1)-0.5)*N
    parameters[i,2] = (np.random.rand(1)-0.5)*M

np.save("parameters_sigma0.npy",parameters)

#k-value for the simulated x-ray source
wavelength = 12.389e-9
dist = 5
pxsize = 172e-6
rec_pxsize = wavelength*dist/(pxsize*probe_size[0])
wavenum = 2*np.pi/wavelength*rec_pxsize

recstart = time.time()
mean = 0
sigma = 0

parameters0 = mis_align(parameters,mean,sigma) #parameters0 are the ones misaligned
np.save("parameters0_sigma0.npy",parameters0)

with PyCallGraph(output=graphviz, config=config):

    #initialize Ptycho3D objects.
    ptycho = Ptycho3D(probe_size,vol_shape,parameters0)
    ptycho_ri = Ptycho3D_RI(ptycho)

    #simulate the diffraction patterns from the imported volume
    diff_pat0 = ptycho_ri.forward(volume,probe0,wavenum)

    recon0 = Path(os.path.abspath('recon_ma_multiscale_sigma0.npy'))
    if os.path.exists('recon_ma_multiscale_sigma0.npy'):
        os.remove('recon_ma_multiscale_sigma0.npy')

    s_s = [1]
    for s0 in s_s:

        #reset data
        parameters_ms = parameters0
        probe = probe0
        diff_pat = diff_pat0

        #multi-scale low resolution
        s1 = np.size(diff_pat[0,0]) #size of the raw diffraction pattern
        pixel_ratio = 1

        if s0 < 1:
            cut = np.int((1-s0)/2*s1) #margins to cut
            diff_pat = diff_pat[:,cut:-cut,cut:-cut] #
            s2 = np.size(diff_pat[0,0]) #scaled size of raw diffration pattern
            pixel_ratio = s2/s1
            wavenum = wavenum/pixel_ratio
            print("initialized data in {} seconds".format(time.time()-tht))

            #Probe rescaling
            probe_f = ptycho.fourier(probe)
            probe_f = probe_f[cut:-cut,cut:-cut]
            probe = ptycho.i_fourier(probe_f)

        parameters_ms[:,[1,2]] = (parameters_ms[:,[1,2]]*pixel_ratio)

        #parameters
        im_size = np.max(parameters_ms[:,1]).astype(int),np.max(parameters_ms[:,2]).astype(int)
        proj_size = tuple(np.add(np.add(im_size,probe.shape),1))


        if recon0.is_file():
            recon = np.load('recon_ma_multiscale_sigma0.npy')
            recon_real = np.real(recon)
            recon_imag = np.imag(recon)
            recon_real_rs = resize(recon_real, (int(M*pixel_ratio),int(N*pixel_ratio),int(N*pixel_ratio)))
            recon_imag_rs = resize(recon_imag, (int(M*pixel_ratio),int(N*pixel_ratio),int(N*pixel_ratio)))
            recon = recon_real_rs + 1j*recon_imag_rs
            recon = recon/pixel_ratio
        else:
            recon = (-0+0j)*np.ones((int(M*pixel_ratio),int(N*pixel_ratio),int(N*pixel_ratio)),dtype=complex)

        #initialize Ptycho3D objects.
        ptycho = Ptycho3D(probe.shape,recon.shape,parameters_ms)
        ptycho_ri = Ptycho3D_RI(ptycho)
        ptycho_ri.Ptycho3D.diff_pat = diff_pat

        volume = None #memory management

        #parameters for LMA:
        mu0 = -1 #set negative to calculate automatically
        muP0 = -1 #set negative to calculate automatically
        ni_LMA = 1
        ni_CGM = 1

        #tracking residual values over iterations
        resid_meas = np.zeros([ni_CGM,ni_LMA]) #initialize residual measure

        #optional save dampening parameter
        if musave == True:
            muArray = np.zeros(ni_LMA)

        #start the actual algorithm
        stop = False #not used
        mu = mu0
        muP = muP0
        musave = np.zeros([ni_LMA,1])
        k = 0
        v = 2 #used for mu update
        r_LMA = diff_pat-ptycho_ri.forward(recon,probe,wavenum) #initial residual

        print("ready to start algorithm after {} s.".format(np.round(time.time()-tht)))

        while not stop and k<ni_LMA:
            try:           
                t_start = time.time()
                #predefining parts of the (jacobian) and jacobian adjoint that will stay constant during an iteration of LMA
                Ja_c = np.nan_to_num(ptycho.fourier(ptycho.probe_object(np.exp(1j*wavenum*ptycho.radon3D(recon)),probe)))
                Ja_c2 = np.nan_to_num(np.conj(ptycho.probe_object(np.exp(1j*wavenum*ptycho.radon3D(recon)),1j*wavenum*probe)))

                #for CGM (add term for total varriation)
                g = ptycho_ri.jacobian_adjoint(recon,probe,r_LMA,wavenum,Ja_c,Ja_c2)

                if mu<0:
                    mu = np.max([np.max(np.abs(g.flatten()))*1,1e-25])#ptycho.norm(g.flatten())
                print("Mu is now",mu)
                musave[k] = mu.real
                np.save("musave_ma_multiscale_sigma0.npy",musave)

                #anonymous function as the operator for CGM (laplacian for total varriation)
                A = lambda h1: ptycho_ri.jacobian_adjoint(recon,probe,ptycho_ri.jacobian(recon,probe,h1,wavenum,np.conj(Ja_c)),wavenum,Ja_c,Ja_c2) + mu*h1

                #CGM initialization
                h = recon*0        
                r = g-A(h)
                p = r

                for j in range(ni_CGM):
                    #this operation takes a long time, so it makes sense to save it.
                    AP = A(p)

                    alpha = np.inner(np.conj(r.flatten()),r.flatten())/np.real(np.inner(np.conj(p.flatten()),AP.flatten()))
                    dh = alpha*p
                    h = h + dh
                    r2 = r - alpha*AP

                    if np.max(np.abs(dh))<1e-8:
                        break

                    beta = np.inner(np.conj(r2.flatten()),(r2.flatten()-r.flatten()))/np.inner(np.conj(r.flatten()),r.flatten()) #Polak-Ribiére
    #                 beta=np.max([-np.inner(np.conj(r2.flatten()),(r2.flatten()-r.flatten()))/np.inner(np.conj(p.flatten()),(r2.flatten()-r.flatten())),0]) #Hestens-Stiefel
                    r=r2
                    p = r + beta*p
                    rf = r.flatten()

                    resid_meas[j,k] = np.inner(np.transpose(np.conj(rf)),rf).real

                    if resid_meas[j,k] > resid_meas[j-1,k] and resid_meas[j-1,k] > resid_meas[j-2,k]:
                        break

                    print("Did iteration {} of {} in CGM".format(j+1,ni_CGM))
                    np.save("resid_ma_multiscale_sigma0.npy",resid_meas)

    #                 plt.ion() #interactive mode "on"              
    #                 plt.figure(1,figsize=(15,10)) #pop up only 1 window

    #                 plt.subplot(2, 3, 1)
    #                 plt.imshow(np.real(h[:,np.int(M/2*pixel_ratio),:]))
    #                 plt.title('k='+str(k)+','+'j='+str(j))
    #                 if j==0 and k==0: #fix colorbar
    #                     plt.colorbar()

    #                 plt.subplot(2, 3, 2)
    #                 plt.imshow(np.imag(h[:,np.int(M/2*pixel_ratio),:]))
    #                 plt.title('k='+str(k)+','+'j='+str(j))
    #                 if j==0 and k==0: #fix colorbar
    #                     plt.colorbar()

    #                 plt.subplot(2, 3, 3)
    #                 plt.plot(resid_meas,'o')
    #                 plt.yscale('log')

    #                 plt.subplot(2, 3, 4)
    #                 plt.imshow(np.real(recon[:,np.int(M/2*pixel_ratio),:]))
    #                 plt.title('k='+str(k)+','+'j='+str(j))
    #                 if j==0 and k==0: #fix colorbar
    #                     plt.colorbar()

    #                 plt.subplot(2, 3, 5)
    #                 plt.imshow(np.imag(recon[:,np.int(M/2*pixel_ratio),:]))
    #                 plt.title('k='+str(k)+','+'j='+str(j))
    #                 if j==0 and k==0: #fix colorbar
    #                     plt.colorbar()

    #                 plt.subplot(2, 3, 6)
    #                 plt.plot(musave,'o')
    #                 plt.yscale('log')

    #                 plt.pause(1) #update image after each CGM iteration
    #                 plt.draw()


                #here the reconstruction is updated completing the iteration of LMA
                recon = recon + h

                #apply constraints:
                #pos_recon_real = np.real(recon)>0
                #pos_recon_real=np.array(pos_recon_real,dtype=complex)
                #recon = recon + pos_recon_real*np.min(np.real(recon))/2  
                #nonnegativity
                recon = -np.abs(np.real(recon))+1j*np.maximum(np.imag(recon),0)

                #memory management
                del r 
                del r2
                del p 
                del alpha 
                del beta 
                del AP

                #here rho is updated. Rho is used to update mu
                r_LMA2 = diff_pat-ptycho_ri.forward(recon,probe,wavenum)

                #if ptycho.norm(r_LMA2.flatten()) > ptycho.norm(r_LMA.flatten()):
                #    print('r_LMA2 is bigger than r_LMA')
                #    break

                rho = (ptycho.norm(r_LMA.flatten())**2 - ptycho.norm(r_LMA2.flatten())**2)/np.real(np.inner(np.conj(h.flatten()),(mu*h+g).flatten()))

                del h #memory management

                #mu update for next iteration.
                if rho > 0:
                    mu=mu*np.max([1/3,1-(2*rho-1)**3])
                    v=2
                    muArray[k] = np.real(mu)
                    print("Mu is now",mu)
                else:
                    mu = mu*v
                    v=2*v
                    muArray[k] = np.real(mu)
                    print("Mu is now",mu)

                r_LMA = r_LMA2

                #saving results

                np.save("recon_ma_multiscale_sigma0.npy",recon)
                print("Did iteration {} of {} in LMA.".format(k+1,ni_LMA))
                print("This iteration took {} s. Reconstruction has been saved as recon_ma_multiscale_sigma0.npy".format(np.round(time.time()-t_start)))
                k=k+1
            except KeyboardInterrupt:
                print("KEYBOARD INTERUPT DETECTED! Current reconstruction is saved as recon_ma_multiscale_sigma0.npy")
                raise

