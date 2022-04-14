import numpy as np
import para
import pyfftw.interfaces.numpy_fft as pyfftw

#from mpi4py_fft import PFFT
############################################################################
from mpi4py import MPI
from mpi4py_fft import PFFT

#comm=MPI.COMM_WORLD
#rank=comm.Get_rank()
#nprocs=comm.Get_size();
############################################################################

############################################################################

def forward_transform(psi, para.fft):
#    N = np.array([para.Nx, para.Ny, para.Nz], dtype=int)
#    fft = PFFT(comm, N, axes=(0, 1, 2), dtype=np.complex128, grid=(-1,))

    #N = np.array([para.Nx, para.Ny, para.Nz], dtype=int)
    #fft = PFFT(comm, N, axes=(0, 1, 2), dtype=np.float, grid=(-1,))
    #psik = (pyfftw.fftn(psi)/(para.Nx*para.Ny*para.Nz))
    psik = para.fft.forward(psi, normalize=True)
    return psik

def inverse_transform(psik, para.fft):
#    N = np.array([para.Nx, para.Ny, para.Nz], dtype=int)
#    fft = PFFT(comm, N, axes=(0, 1, 2), dtype=np.complex128, grid=(-1,))
    # psik=dealias(psik)
    #N = np.array([para.Nx, para.Ny, para.Nz], dtype=int)
    #fft = PFFT(comm, N, axes=(0, 1, 2), dtype=np.float, grid=(-1,))
    #psi = (pyfftw.ifftn(psik)*(para.Nx*para.Ny*para.Nz))
    #psi = fft.backward(psik, psi)
    psi = para.fft.backward(psik)
    return psi


## meshgrid

def def_x_kx(comm):
	Dx = np.linspace(comm.Get_rank()*para.Nx/comm.Get_size(), ((comm.Get_rank()+1)*para.Nx/comm.Get_size())-1, para.Nx//comm.Get_size() ) - ( para.Nx//2 -1 ) 
#	kx = (2*np.pi/para.Lx)*np.roll(Dx, para.Nx//(2*comm.Get_size()) -1)
	kx=2*np.pi*np.roll(Dx,para.Nx//2+1)/para.Lx
	x = Dx*para.Lx/para.Nx		# To make it INTEGER
	return x, kx
def def_y_ky():
	Dy=np.arange(-para.Ny//2+1,para.Ny//2+1)
#	ky=2*np.pi*np.roll(Dy,para.Ny//2+1)/para.Ly
	y=Dy*para.Ly/para.Ny
	return y, ky
def def_z_kz():
	Dz=np.arange(-para.Nz//2+1,para.Nz//2+1)
	kz=2*np.pi*np.roll(Dz,para.Nz//2+1)/para.Lz
	z=Dz*para.Lz/para.Nz
	return z, kz


#Dx=np.arange(-para.Nx//2+1,para.Nx//2+1)
#kx=2*np.pi*np.roll(Dx,para.Nx//2+1)/para.Lx
#x=Dx*para.Lx/para.Nx
#print(x)

'''
Dy=np.arange(-para.Ny//2+1,para.Ny//2+1)
ky=2*np.pi*np.roll(Dy,para.Ny//2+1)/para.Ly
y=Dy*para.Ly/para.Ny
#print(Dy)

Dz=np.arange(-para.Nz//2+1,para.Nz//2+1)
kz=2*np.pi*np.roll(Dz,para.Nz//2+1)/para.Lz
z=Dz*para.Lz/para.Nz
#print(rank, z)
'''
###################################################################
def set_meshrid(x, y, z, kx, ky, kz):

##Dx =  np.linspace(rank*para.Nx/nprocs, ((rank+1)*para.Nx/nprocs)-1, para.Nx//nprocs ) - ( para.Nx//2 -1 ) 
##kx = (2*np.pi/para.Lx)*np.roll(Dx, para.Nx//(2*nprocs) -1)
#	kx=2*np.pi*np.roll(Dx,para.Nx//2+1)/para.Lx
##x = Dx*para.Lx/para.Nx		# To make it INTEGER
#	return x, kx
#print(rank,x)
	
##n = para.Nx//nprocs
##kx = kx[(rank*n):((rank*n)+n)]
##x = x[(rank*n):((rank*n)+n)]
#	print(rank, "	", x)
	
	x_mesh,y_mesh,z_mesh=np.meshgrid(x,y,z,indexing='ij')
	kx_mesh, ky_mesh, kz_mesh = np.meshgrid(kx, ky, kz,indexing = 'ij')
	ksqr = kx_mesh**2 + ky_mesh**2 + kz_mesh**2

	return x_mesh, y_mesh, z_mesh, kx_mesh, ky_mesh, kz_mesh, ksqr

##################################################################
#x_mesh, y_mesh, z_mesh, kx_mesh, ky_mesh, kz_mesh, ksqr = set_meshrid(rank, nprocs, y, z)


