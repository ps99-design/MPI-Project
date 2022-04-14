from sch import Wavefunction
import numpy as np
import para 
import init_cond
import copy
import time
import my_fft

#import my_fft

from mpi4py import MPI


#t1 = time.time()
###############################################################
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
nprocs=comm.Get_size();
###############################################################
#t1 = time.time()

t=para.dt
G=Wavefunction()
G.set_arrays(nprocs)

###############################################################################
#my_fft.MeshGrid(rank, nprocs)
#init_cond.initcond1(G)		# Initialization.
init_cond.initcond1(G, comm)		# Initialization.

#t2 = time.time()
#print(t2 - t1)
def compute_RHS(psik):
    psi=my_fft.inverse_transform(psik, comm)
    #ff = my_fft.forward_transform(((self.V+para.Nlin*np.abs(self.psi)**2)*self.psi), comm)
    #print(ksqr.shape)
    #self.psik=-1j*(self.ksqr*self.psik/2+my_fft.forward_transform(((self.V+para.Nlin*np.abs(self.psi)**2)*self.psi), comm))
j=0
for i in range(para.Ngt):
    # G.sstep_RK2()
    # RK2
    G.tempF=copy.deepcopy(G.psik)
    compute_RHS(G.psik)
    G.psik=G.tempF+G.psik*para.dt/2
    compute_RHS(G.psik)
    G.psik=G.tempF+para.dt*G.psik


#    if (para.tstepMov[j]-t)/para.dt<para.dt:
 #       print('\n----------------------')
  #      print('t: ',t)
   #     print('norm: ',G.norm())
    #    print('energy: ',G.energy())
     #   j=j+1
    t =t+para.dt
