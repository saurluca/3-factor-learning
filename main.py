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
    plt.plot(
        [spikes[0] if len(spikes) > 0 else None for spikes in spike_trajectories],
        marker="o",
    )
    plt.gca().invert_yaxis()  # earlier = up
    plt.xlabel("Cycle")
    plt.ylabel("Post-spike latency (ms)")
    plt.title("Output spike marches earlier")
    

def plot_final_weights(weight_trajectories, neuron_names):    # Plot results
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 3)
    final_weights = weight_trajectories[-1]
    colors = ["red" if "City" in name else "blue" for name in neuron_names]
    bars = plt.bar(neuron_names, final_weights, color=colors, alpha=0.7)
    plt.ylabel("Final Weight")
    plt.title("Final Synaptic Weights")
    plt.xticks(rotation=45)

    # Add text annotations
    for bar, weight in zip(bars, final_weights):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{weight:.3f}",
            ha="center",
            va="bottom",
        )

    plt.subplot(2, 2, 4)
    weight_changes = final_weights - weight_trajectories[0]
    colors = [
        "red" if "City" in name else "green" if change > 0 else "orange"
        for name, change in zip(neuron_names, weight_changes)
    ]
    bars = plt.bar(neuron_names, weight_changes, color=colors, alpha=0.7)
    plt.ylabel("Weight Change")
    plt.title("Weight Changes (Final - Initial)")
    plt.xticks(rotation=45)
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)

    # Add text annotations
    for bar, change in zip(bars, weight_changes):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.01 if change >= 0 else -0.02),
            f"{change:+.3f}",
            ha="center",
            va="bottom" if change >= 0 else "top",
        )

    plt.tight_layout()
    plt.show()


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

cfg.weight_max_exc = (
    2.0  # max excitatory weight (increased from 0.5 to make spikes possible)
)
cfg.weight_fixed_inh = (
    0.05  # max inhibitory weight (dimensionless w.r.t. leak conductance)
)

# STDP constants
cfg.tau_plus = 20e-3  # time constant for potentiation [s]
cfg.tau_minus = 20e-3  # time constant for depression [s]
cfg.weight_plus = 0.01  # fraction of g_max added per causal pair (increased)
cfg.weight_minus = 1.05 * cfg.weight_plus  # fraction of g_max added per causal pair

# Network parameters
cfg.N_exc = 5  # number of excitatory neurons
cfg.N_inh = 1  # number of inhibitory neurons
cfg.N_total = cfg.N_exc + cfg.N_inh

# %%


def run_simulation(exc_spike_trains, inh_spike_trains, weights, sim_time, cfg):
    """Simulate the LIF neuron for T seconds.
    exc_spike_trains – list of numpy arrays with spike times for each excitatory synapse.
    Returns (postsynaptic spike times, final weights list)."""

    # ignoring inhibitory neurons for now

    n_steps = int(sim_time / cfg.dt)

    ltp_trace = np.zeros(len(weights))  # trace for LTP, og: P
    ltd_trace = 0.0  # trace for LTD, og: M

    post_spk = []  # post-synaptic spike times

    exc_conductance = 0.0  # og: g_ex
    # inh_conductance = 0.0 # og: g_in

    v = cfg.v_rest  # membrane potential, begins at resting potential

    # print(f"len(exc_spike_trains): {len(exc_spike_trains)}")
    # print(f"len(exc_spike_trains[0]): {len(exc_spike_trains[0])}")

    voltage_trace = []

    for step in range(n_steps):
        t = step * cfg.dt

        # iterate over all neurons to check if they have spiked
        # excitatory neurons
        for i, spike_times in enumerate(exc_spike_trains):
            # check if spike time is within current time step
            if len(spike_times) > 0 and spike_times[0] <= t:
                # update conductance
                exc_conductance += weights[i]
                # update ltp trace
                ltp_trace[i] += cfg.weight_plus
                # update weights based on LTD trace, and clip at 0
                weights[i] = max(weights[i] + ltd_trace * cfg.weight_max_exc, 0.0)
                # remove spike time
                # np.delete(exc_spike_trains[i], 0)  # todo check working
                spike_times = spike_times[1:]  # remove first spike
                exc_spike_trains[i] = spike_times

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

        voltage_trace.append(v)

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

    return post_spk, weights, voltage_trace


# %%


def toy_experiment():
    # experiment parameters
    exc_spike_trains = np.array([[0.005], [0.010], [0.015], [0.020], [0.025]])
    inh_spike_trains = np.array([[0.5]])
    sim_time = 0.050  # 50 ms cycle
    cycles = 10

    weights = np.full(cfg.N_exc, 0.1 * cfg.weight_max_exc)

    spike_trajectories = []
    weight_trajectories = [weights]

    print(f"Initial weights: {weights}")

    for cycle in range(cycles):
        spikes, weights = run_simulation(
            exc_spike_trains, inh_spike_trains, weights, sim_time, cfg
        )

        spike_trajectories.append(spikes)
        weight_trajectories.append(weights)

    print(f"Final weights: {weights}")

    # plot spikes
    plot_spikes(spike_trajectories, sim_time)

    plot_weight_trajectories(weight_trajectories)

    plot_first_spike_latencies(spike_trajectories)


# %%

# The goal is to evaluate the ability of our model to learn the causal relationships
# The data should be generated based on a simple graph

# Smoking -> Tar in lungs -> Cancer
# City (indpendent of Cancer, occurs randomly)

# Check if model can differentiate between the two relationships


def generate_causal_spike_dataset(T):
    """Generate spike trains for causal relationship experiment.

    Smoking -> Tar -> Cancer (causal chain)
    City -> random spikes (confounder)
    """
    # Dataset parameters
    smoke_burst_interval = 0.004  # seconds between smoking bursts
    smoke_burst_rate = 500  # Hz during burst
    smoke_burst_duration = 0.001  # 10ms bursts
    tar_delay = 0.002  # 20ms delay after smoking
    cancer_delay = 0.003  # 30ms delay after tar
    city_rate = 1000  # Hz for random city spikes

    # Generate smoking spike bursts
    smoke_spikes = []
    t = 0
    while t < T:
        # Generate burst
        burst_spikes = poisson_spike_times(smoke_burst_rate, smoke_burst_duration, rng)
        smoke_spikes.extend(burst_spikes + t)
        t += smoke_burst_interval

    smoke_spikes = np.array(smoke_spikes)
    smoke_spikes = smoke_spikes[smoke_spikes < T]  # keep only spikes within T

    # Generate tar spikes (triggered by smoking with delay)
    tar_spikes = smoke_spikes + tar_delay
    tar_spikes = tar_spikes[tar_spikes < T]

    # Generate cancer spikes (triggered by tar with delay)
    cancer_spikes = tar_spikes + cancer_delay
    cancer_spikes = cancer_spikes[cancer_spikes < T]

    # Generate random city spikes (confounder)
    city_spikes = poisson_spike_times(city_rate, T, rng)

    return {
        "smoke": smoke_spikes,
        "tar": tar_spikes,
        "cancer": cancer_spikes,
        "city": city_spikes,
    }


