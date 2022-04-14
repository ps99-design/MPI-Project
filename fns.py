import para
import my_fft
from scipy import integrate as st
#import scipy.integrate


def my_integral(psi):
    if para.Nx==1 and para.Ny==1:
        return(st.simps(psi,my_fft.z).reshape(-1))
    elif para.Ny==1:
        return st.simps(st.simps(psi,my_fft.z).reshape(-1),my_fft.x)
    return st.simps(st.simps(st.simps(psi,my_fft.z),my_fft.y),my_fft.x)
