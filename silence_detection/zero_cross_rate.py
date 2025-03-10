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
    #This shift factor exists to stop the ZCR from being a decimal.
    #the original zcr algoithm had a decimal threshold of around 0.14
    #but I do not want to deal with decimals on the FPGA. With the shift
    #at 6 silent signals tend to have a value of 127 and silent signals one of
    #50
    shift_offset = 6
    #n_factor = 1/(len(signal)-1)
    # print(len(signal))
    for i in range(1,len(signal)):
        zcr += zc_diff(signal[i],signal[i-1])

    #take the average of the ZCR using shifts assuming the sample
    #length is a power of 2. shift_offset to stop the value from getting 
    #too small (decimals)
    return zcr>>(int(log2(len(signal)))-shift_offset)