# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

def alpha_n(V):
    return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))

def beta_n(V):
    return 0.125 * np.exp(-(V + 65) / 80)

def alpha_m(V):
    return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

def beta_m(V):
    return 4.0 * np.exp(-(V + 65) / 18)

def alpha_h(V):
    return 0.07 * np.exp(-(V + 65) / 20)

def beta_h(V):
    return 1 / (1 + np.exp(-(V + 35) / 10))

def hh_step(V, m, h, n, I_ext, dt):
    g_Na = 120.0
    g_K = 36.0
    g_L = 0.3
    E_Na = 50.0
    E_K = -77.0
    E_L = -54.387

    I_Na = g_Na * m**3 * h * (V - E_Na)
    I_K = g_K * n**4 * (V - E_K)
    I_L = g_L * (V - E_L)

    dV = (I_ext - I_Na - I_K - I_L) / 1.0  # Cm = 1 uF/cm^2
    V_new = V + dt * dV

    m += dt * (alpha_m(V) * (1 - m) - beta_m(V) * m)
    h += dt * (alpha_h(V) * (1 - h) - beta_h(V) * h)
    n += dt * (alpha_n(V) * (1 - n) - beta_n(V) * n)

    return V_new, m, h, n

def simulate(duration, I_func, dt=0.01):
    steps = int(duration / dt)
    t = np.arange(0, duration, dt)
    V = -65.0
    m = alpha_m(V) / (alpha_m(V) + beta_m(V))
    h = alpha_h(V) / (alpha_h(V) + beta_h(V))
    n = alpha_n(V) / (alpha_n(V) + beta_n(V))
    Vs = np.empty(steps)
    spike_times = []
    for i in range(steps):
        I_ext = I_func(t[i])
        V, m, h, n = hh_step(V, m, h, n, I_ext, dt)
        Vs[i] = V
        if i > 0 and Vs[i - 1] < 0 and V >= 0:
            spike_times.append(t[i])
    return t, Vs, spike_times

def step_current(t):
    return 10.0 if 10.0 <= t <= 60.0 else 0.0

def run_exp1():
    t, V, spikes = simulate(100.0, step_current)
    plt.figure()
    plt.plot(t, V)
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')
    plt.title('Step current response')
    plt.savefig('voltage_step_current.png')
    plt.close()

def run_exp2():
    duration = 500.0
    currents = np.arange(0, 21, 1)  # µA/cm^2
    freqs = []
    for I in currents:
        def I_func(t, I=I):
            return I
        t, V, spikes = simulate(duration, I_func)
        # Use all spikes over the full duration
        freq = len(spikes) / (duration / 1000.0)  # Hz
        freqs.append(freq)
    plt.figure()
    plt.plot(currents, freqs, marker='o')
    plt.xlabel('Injected current (µA/cm²)')
    plt.ylabel('Firing frequency (Hz)')
    plt.title('Current–frequency (f–I) curve')
    plt.savefig('frequency_vs_current.png')
    plt.close()
    return currents, freqs

def main():
    run_exp1()
    currents, freqs = run_exp2()
    answer = freqs[-1]
    print('Answer:', answer)

if __name__ == '__main__':
    main()
