from signal_energy_calculation import signal_energy
#returns 1 if voice present 0 if not
def sig_classify(signal, threshold=1000):
    zcr = signal_energy(signal)
    decision = 0
    if zcr >= threshold:
        decision = 1
    else:
        decision = 0
    return decision