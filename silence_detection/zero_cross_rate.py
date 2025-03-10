import numpy as np
from math import log2, ceil

#takes difference of signs
def zc_diff(a,b):
    result = a*b
    if result>0:
        return 0
    else:
        return 1
    

#calculates to zero cross rate. Input must be at minimum length 2
def zcr_calc(signal):
    zcr = 0
    #n_factor = 1/(len(signal)-1)
    # print(len(signal))
    for i in range(1,len(signal)):
        zcr += zc_diff(signal[i],signal[i-1])
    # print(f"zcr is {zcr>>(int(log2(len(signal))))}")
    return zcr>>(int(log2(len(signal))))