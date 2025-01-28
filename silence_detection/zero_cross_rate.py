import numpy as np

#takes difference of signs
def zc_diff(a,b):
    result = a*b
    if result>0:
        return 0
    else:
        return 1
    

#calculates to zero cross rate. Input must be at minimum length 2
def zcr_calc(signal):
    norm_signal = signal/np.max(signal)
    zcr = 0
    n_factor = 1/(len(norm_signal)-1)
    for i in range(1,len(norm_signal)):
        zcr += zc_diff(norm_signal[i],norm_signal[i-1])
    return n_factor*zcr