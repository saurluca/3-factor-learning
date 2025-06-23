# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from types import SimpleNamespace

rng = np.random.default_rng(42)

def poisson_spike_times(rate, T, rng=rng):
    n_spikes = rng.poisson(rate * T)
    return np.sort(rng.random(n_spikes) * T)


# plot your spike data
def plot_spikes(spike_trains, T):
    """
    Quick raster plot.

    Parameters
    ----------
    spike_trains : array-like of 1-D numpy arrays
        spike_trains[i] contains the spike times (s) for neuron i.
    T : float
        Maximum time for x-axis (s).
    """
    plt.eventplot(spike_trains, linelengths=0.3, color="k")
    plt.xlim(0, T)
    plt.ylim(-0.5, len(spike_trains) - 0.5)
    plt.xlabel("time (s)")
    plt.ylabel("neuron index")
    plt.show()

def plot_weight_trajectories(weight_trajectories):
    plt.figure()
    weight_traj_array = np.array(weight_trajectories)
    for i in range(cfg.N_exc):
        plt.plot(weight_traj_array[:, i], label=f"Weight {i + 1}")
    plt.xlabel("Iteration")
    plt.ylabel("Weight")
    plt.title("Weight Trajectories Over Iterations")
    plt.legend()
    plt.show()
    

def plot_first_spike_latencies(spike_trajectories):
    plt.figure()
    plt.plot([spikes[0] if len(spikes) > 0 else None for spikes in spike_trajectories], marker="o")
    plt.gca().invert_yaxis()  # earlier = up
    plt.xlabel("Cycle")
    plt.ylabel("Post-spike latency (ms)")
    plt.title("Output spike marches earlier")

# %%

#  Simulation & model parameters
cfg = SimpleNamespace(**{})

cfg.dt = 1e-3  # simulation time step [s]
cfg.tau_membrane = 20e-3  # membrane time constant [s]
cfg.v_rest = -0.070  # resting potential [V]
cfg.v_reset = -0.060  # reset potential [V]
cfg.v_threshold = -0.054  # spike threshold [V]
cfg.tau_exc_decay = 5e-3  # excitatory synaptic decay [s]
cfg.tau_inh_decay = 5e-3  # inhibitory synaptic decay [s]

cfg.weight_max_exc = 0.5  # max excitatory weight (dimensionless w.r.t. leak conductance)
cfg.weight_fixed_inh = 0.05  # max inhibitory weight (dimensionless w.r.t. leak conductance)

# STDP constants
cfg.tau_plus = 20e-3 # time constant for potentiation [s]
cfg.tau_minus = 20e-3 # time constant for depression [s]
cfg.weight_plus = 0.005  # fraction of g_max added per causal pair
cfg.weight_minus = 1.05 * cfg.weight_plus # fraction of g_max added per causal pair

# Network parameters
cfg.N_exc = 5 # number of excitatory neurons
cfg.N_inh = 1 # number of inhibitory neurons
cfg.N_total = cfg.N_exc + cfg.N_inh

# %%


def run_simulation(exc_spike_trains, inh_spike_trains, weights, sim_time, cfg):
    """Simulate the LIF neuron for T seconds.
    exc_spike_trains – list of numpy arrays with spike times for each excitatory synapse.
    Returns (postsynaptic spike times, final weights list)."""
    
    # ignoring inhibitory neurons for now

    n_steps = int(sim_time / cfg.dt)
    
    ltp_trace = np.zeros(len(weights)) # trace for LTP, og: P
    ltd_trace = 0.0 # trace for LTD, og: M
    
    post_spk = [] # post-synaptic spike times

    exc_conductance = 0.0 # og: g_ex
    # inh_conductance = 0.0 # og: g_in
    
    v = cfg.v_rest # membrane potential, begins at resting potential
    
    for step in tqdm(range(n_steps)):
        t = step * cfg.dt
        
        # iterate over all neurons to check if they have spiked
        # excitatory neurons
        for i, spike_times in enumerate(exc_spike_trains):
            # check if spike time is within current time step
            if spike_times[0] <= t and spike_times.size:
                # update conductance
                exc_conductance += weights[i]
                # update ltp trace
                ltp_trace[i] += cfg.weight_plus
                # update weights based on LTD trace, and clip at 0
                weights[i] = max(weights[i] + ltd_trace * cfg.weight_max_exc, 0.0)
                # remove spike time
                np.delete(exc_spike_trains[i], 0)  # todo check working
        
        # inhibitory neurons
        # for i, spike_times in enumerate(inh_spike_trains):
        #     pass

        # update conductances
        exc_conductance -= (cfg.dt / cfg.tau_exc_decay) * exc_conductance
        # inh_conductance -= (cfg.dt / cfg.tau_inh_decay) * inh_conductance
        
        # update ltp and ltd traces
        ltp_trace -= ltp_trace / cfg.tau_plus * cfg.dt
        ltd_trace -= ltd_trace / cfg.tau_minus * cfg.dt
        
        # update voltage v
        dv = (cfg.v_rest - v - v * exc_conductance) / cfg.tau_membrane * cfg.dt
        v += dv
        
        # check for spikes
        if v >= cfg.v_threshold:
            # add spike time to post-synaptic spike times
            post_spk.append(t)
            # reset voltage1
            v = cfg.v_reset
            # update LTD trace
            ltd_trace -= cfg.weight_minus
            # update weights based on LTP trace
            weights += ltp_trace * cfg.weight_max_exc
            # clip weights at max value
            weights = np.clip(weights, 0, cfg.weight_max_exc)
        
    return post_spk, weights
        

# %%

# experiment parameters
exc_spike_trains = np.array([[0.005], [0.010], [0.015], [0.020], [0.025]])
inh_spike_trains = np.array([[0.5]])
sim_time = 0.050  # 50 ms cycle
cycles = 10

weights = np.full(cfg.N_exc, 0.1*cfg.weight_max_exc)

spike_trajectories = []
weight_trajectories = [weights]

print(f"Initial weights: {weights}")

for cycle in range(cycles):
    spikes, weights = run_simulation(exc_spike_trains, inh_spike_trains, weights, sim_time, cfg)

    spike_trajectories.append(spikes)
    weight_trajectories.append(weights)

print(f"Final weights: {weights}")


# plot spikes
plot_spikes(spike_trajectories, sim_time)

plot_weight_trajectories(weight_trajectories)

plot_first_spike_latencies(spike_trajectories)