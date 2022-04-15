from my_fft import *
import para
import numpy as np
import copy

#x_mesh, y_mesh, z_mesh, kx_mesh, ky_mesh, kz_mesh, ksqr = my_fft.set_meshrid(rank, nprocs, my_fft.y, my_fft.z)

def initcond1(G, comm):

    '''
    Initial wavefunction
    '''
#    x, kx = def_x_kx(comm)
 #   y, ky = def_y_ky()
  #  z, kz = def_z_kz()
    x, y, z = get_xyz(comm)
    kx, ky, kz = get_kx_ky_kz(comm)
    #print(comm.Get_rank(), x)

    G.comm = comm
    x_mesh, y_mesh, z_mesh, kx_mesh, ky_mesh, kz_mesh, ksqr = set_meshrid(x, y, z, kx, ky, kz)
    G.x = x
    G.y = y
    G.z = z
    G.ksqr = ksqr
    G.kx_mesh = kx_mesh
    G.ky_mesh = ky_mesh
    G.kz_mesh = kz_mesh
    
    # G.psi = (1/np.pi**(1/4))*np.exp(-(my_fft.z_mesh**2/2+my_fft.y_mesh**2+my_fft.x_mesh**2/2))+0j  #1D initial condition
    #x_mesh, y_mesh, z_mesh, kx_mesh, ky_mesh, kz_mesh, ksqr = my_fft.set_meshrid(rank, nprocs, my_fft.y, my_fft.z)
   # G.psi = (1/(np.pi)**(1/2))*np.exp(-(my_fft.z_mesh**2/2+my_fft.y_mesh**2+my_fft.x_mesh**2/2))+0j    #2D initial condition
    #G.psi = (1/(np.pi)**(1/2))*np.exp(-(z_mesh**2/2+y_mesh**2+x_mesh**2/2))+0j    #2D initial condition
    #G.psi = (1/(np.pi)**(3/4))*np.exp(-(my_fft.z_mesh**2+my_fft.y_mesh**2+my_fft.x_mesh**2)/2)+0j  # 3D initial condition
    G.psi = (1/(np.pi)**(3/4))*np.exp(-(z_mesh**2+y_mesh**2+x_mesh**2)/2)+0j  # 3D initial condition
    #print(comm.Get_rank(), "	", G.psi)
    #print(comm.Get_rank(), "	", G.psi.shape)
    #G.psik=my_fft.forward_transform(G.psi)
    G.psik=forward_transform(G.psi, comm)
    #print(comm.Get_rank(), "	", G.psik)
    #print(comm.Get_rank(), "	", G.psik.shape)
    G.tempF=copy.deepcopy(G.psik)
    G.tempR=copy.deepcopy(G.psi)



    '''
    potential
    '''
#    G.V=(my_fft.x_mesh**2+my_fft.y_mesh**2+my_fft.z_mesh**2)/2
    G.V=(x_mesh**2+y_mesh**2+z_mesh**2)/2
#    del my_fft.x_mesh,my_fft.y_mesh,my_fft.z_mesh
    del x_mesh,y_mesh,z_mesh
