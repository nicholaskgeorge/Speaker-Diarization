from zero_cross_rate import zcr_calc

#returns 1 if voice present 0 if not
def zcr_classify(signal, threshold=0.14):
    zcr = zcr_calc(signal)
    decision = 0
    if zcr <= threshold:
        decision = 1
    else:
        decision = 0
    return decision
    
