import numpy as np
'''
For 1D set Nx=1,Ny=1,Lx=1,Ly=1
For 2D set Ny=1,Ly=1
'''

Lx=2		#16
Ly=2			# 3D-Data
Lz=2		#16

dimension=3	#1

Nlin=12.52

Nx = 4		#64
Ny = 4
Nz = 4		#64

#nprocs = 4

tinit = 0
tstepMov=np.array([0.001,1.0,4.0,6.0,10.0]) #points at which we have to print Norm and Energy
volume=Lx*Ly*Lz


tMax=10 # time 
dt=0.001 
Ngt=int(tMax/dt)

N = np.array([Nx, Ny, Nz], dtype=int)
fft = PFFT(comm, N, axes=(0, 1, 2), dtype=np.complex128, grid=(-1,))