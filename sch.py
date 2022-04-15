import numpy as np
import para
import my_fft
import fns
import copy
#import init_cond

class Wavefunction():
    def __init__(self):
        self.psi=[]
        self.V=[]
        self.psik=[]
        self.tempR=[] #temp in real space
        self.tempF=[] #temp in fourier space 
    
    def set_arrays(self, n):
#	self.n = n
        self.psi=np.zeros(((para.Nx//n),para.Ny,para.Nz),dtype=np.complex128)
        self.psik=np.zeros_like(self.psi)
	#print(rank, self.psi.shape)
#        self.psik=np.zeros((para.Nx,para.Ny,para.Nz),dtype=np.complex128)
        #self.psik=np.zeros((para.Nx,para.Ny,(para.Nz//n)),dtype=np.complex128)
#        self.tempR=np.zeros((para.Nx,para.Ny,para.Nz),dtype=np.complex128)
        self.tempR=np.zeros_like(self.psi)
#        self.tempF=np.zeros((para.Nx,para.Ny,para.Nz),dtype=np.complex128)
        self.tempF=np.zeros_like(self.psi)
        #self.V = np.zeros((para.Nx,para.Ny,para.Nz),dtype=np.float64)
        self.V = np.zeros(((para.Nx//n),para.Ny,para.Nz),dtype=np.float64)

    #Calculation of  integral of space derivative part is done by using the fft
    def energy(self):
  ##      derivative2=para.volume*np.sum(np.abs(1j*(my_fft.kx_mesh+my_fft.ky_mesh+my_fft.kz_mesh)*self.psik)**2)
        derivative2=para.volume*np.sum(np.abs(1j*(self.kx_mesh+self.ky_mesh+self.kz_mesh)*self.psik)**2)
        z=np.conjugate(self.psi)*((self.V+para.Nlin*np.abs(self.psi)**2/2)*self.psi)
        
  ##      return fns.my_integral(z.real)+derivative2.real/2
        return fns.my_integral(z.real, self.x, self.y, self.z)+derivative2.real/2

    def norm(self):
        self.psik=my_fft.forward_transform(self.psi, self.comm) #for sstep_strang
        # self.psi=my_fft.inverse_transform(self.psik) #for RK2 or RK4
        return para.volume*np.sum(np.abs(self.psik)**2)
    
     
    def chempot(self):  # When Non-lineartity is present
       
        derivative2=para.volume*np.sum(np.abs(1j*(my_fft.kx_mesh+my_fft.ky_mesh+my_fft.kz_mesh)*self.psik)**2)
        z=np.conjugate(self.psi)*((self.V+para.Nlin*np.abs(self.psi)**2)*self.psi)
        return fns.my_integral(z.real)+derivative2.real/2
    
    def compute_RHS(self):
        self.psi=my_fft.inverse_transform(self.psik, self.comm)
        self.psik=-1j*(self.ksqr*self.psik/2+my_fft.forward_transform( (self.V+para.Nlin*np.abs(self.psi)**2)*self.psi, self.comm ))

    
    # Schemes for time evolution
    # def sstep_RK2(self):
    #     self.tempF=copy.deepcopy(self.psik)
    #     self.compute_RHS()
    #     self.psik=self.tempF+self.psik*para.dt/2
    #     self.compute_RHS()
    #     self.psik=self.tempF+para.dt*self.psik
    
    
    #TSSP
    def sstep_strang(self):
        self.psi= self.psi*np.exp(-1j*(self.V+para.Nlin*np.abs(self.psi)**2)*para.dt/2)
        self.psik=my_fft.forward_transform(self.psi)
        self.psik= self.psik*np.exp(-1j*my_fft.ksqr/2*para.dt)
        self.psi=my_fft.inverse_transform(self.psik)
        self.psi= self.psi*np.exp(-1j*(self.V+para.Nlin*np.abs(self.psi)**2)*para.dt/2)
