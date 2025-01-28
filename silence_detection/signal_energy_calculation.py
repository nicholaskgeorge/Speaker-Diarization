def signal_energy(signal):
    energy = 0
    for i in signal:
        energy += i*i
    energy = energy/len(signal)
    return energy