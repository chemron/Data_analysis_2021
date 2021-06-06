# imports
import csv
import urllib.request as request
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pystan as stan


# get data
url = "http://astrowizici.st/teaching/phs5000/decay.csv"
response = request.urlopen(url)
lines = [l.decode('utf-8') for l in response.readlines()]
cr = csv.reader(lines)
cr_data = list(cr)
data = np.array(cr_data[1:])
t, N, N_err, d_name = data.T
# time in seconds
t = t.astype(np.float)
# grams of radioactive material measured
N = N.astype(np.float)
# uncertainty in grams measured
N_err = N_err.astype(np.float)
# Detector name
d_name = d_name.astype(str)

# define useful quantities
days_to_s = 60 * 60 * 24
s_to_days = 1/days_to_s
# known parameters:
N_widgets = 100
N_initial_max = 20
manufacturing_time_span = 35 * days_to_s
t_delay = 14 * days_to_s
measurement_time_span = 90 * days_to_s
N_observations = 1
# time in days
t_days = t * s_to_days
# minimum measured N
min_N = np.min(N)
# Minimimum time span:
min_dt = t_delay
# maximum possible alpha value
max_alpha = -1/(min_dt) * np.log(min_N/N_initial_max)


# plot figure
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(t_days, N, c="k", s=10)
ax.errorbar(t_days, N, yerr=N_err, fmt="o", lw=1, c="k")
ax.axhline(0, linestyle='--', c='gray')
ax.set_xlabel(r"Time [days since manufacturing]")
ax.set_ylabel(r"Measured Radioactive material [grams]")
ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(7))
fig.tight_layout()
plt.savefig('q1_data.png', dpi=300)


# pystan utils
def sampling_kwds(**kwargs):
    r"""
    Prepare a dictionary that can be passed to Stan at the sampling stage.
    Basically this just prepares the initial positions so that they match the
    number of chains.
    """

    kwds = dict(chains=4)
    kwds.update(kwargs)

    if "init" in kwds:
        kwds["init"] = [kwds["init"]] * kwds["chains"]

    return kwds


# define model
model_filename = "model1.stan"
model_str = """
data {
    int<lower=1> N_widgets; // number of widgets

    // Time of measurement.
    vector[N_widgets] t_measured;

    // Amount of material measured is uncertain.
    vector[N_widgets] N_measured;
    vector[N_widgets] sigma_N_measured;

    // Maximum amount of initial material.
    real N_initial_max;

    //  last possible time of manufacture.
    real t_initial_max;

    //  max value of alpha
    real alpha_max;
}

parameters {
    // Time of manufacture.
    vector<lower=0, upper=t_initial_max>[N_widgets] t_initial;

    // The decay rate parameter.
    real<lower=0, upper=alpha_max> alpha;

    // The amount of initial material is not known.
    vector<lower=0, upper=N_initial_max>[N_widgets] N_initial;
}

model {
    for (i in 1:N_widgets) {
        N_measured[i] ~ normal(
          N_initial[i] * exp(-alpha * (t_measured[i] - t_initial[i])),
          sigma_N_measured[i]
        );
    }
}
"""
# make model
model = stan.StanModel(model_code=model_str)

# Data.
data_dict = dict(
    N_widgets=N_widgets,
    t_measured=t,
    N_measured=N,
    sigma_N_measured=N_err,
    N_initial_max=N_initial_max,
    t_initial_max=manufacturing_time_span,
    alpha_max=max_alpha,
)

# initial guess
alpha_guess = max_alpha/2
t_init_guess = np.full(N_widgets, manufacturing_time_span/2)

init_dict = dict(
    t_initial=t_init_guess,
    alpha=alpha_guess,
)

# Run optimisation.
opt_stan = model.optimizing(
    data=data_dict,
    init=init_dict
)

# Run sampling.
samples = model.sampling(**sampling_kwds(
    chains=2,
    iter=2000,
    data=data_dict,
    init=opt_stan
))


# plot initial N
fig = samples.traceplot(("N_initial", ))
fig.set_size_inches(10, 5)
plt.savefig('q11_N_init.png', dpi=300)

# plot initial t
fig = samples.traceplot(("t_initial", ))
fig.set_size_inches(10, 5)
plt.savefig('q11_t_init.png', dpi=300)


# get alpha chain and remove burn in
alpha_chain = samples["alpha"][1000:]
# calculate mean and std
alpha_mean = np.mean(alpha_chain)
alpha_err = np.std(alpha_chain)

# plot
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

ax = axs[0]
ax.plot(alpha_chain, c='k')
ax.set_xlabel("Step Number")
ax.set_ylabel("Alpha")

ax = axs[1]
ax.hist(alpha_chain, bins=30, color='gray', label='Posterior Distribution')
ax.axvline(alpha_mean, c='k', linestyle='--', label='Mean')
ax.set_ylabel("Posterior Probability")
ax.set_xlabel("Alpha")
ax.legend()
plt.savefig('q11_alpha.png', dpi=300)


# print mean and std
print(f'Alpha: {alpha_mean:.2}, Standard Deviation: {alpha_err:.0}')


# Part 2


# make model
model_filename = "model2.stan"
model_str = """
data {
    int<lower=1> N_widgets; // number of widgets

    // Time of measurement.
    vector[N_widgets] t_measured;

    // Amount of material measured is uncertain.
    vector[N_widgets] N_measured;
    vector[N_widgets] sigma_N_measured;

    // Maximum amount of initial material.
    real N_initial_max;

    //  last possible time of manufacture.
    real t_initial_max;

    //  max value of alpha
    real alpha_max;

    //detector
    int k[N_widgets];
}

parameters {
    // Time of manufacture.
    vector<lower=0, upper=t_initial_max>[N_widgets] t_initial;

    // The decay rate parameter.
    real<lower=0, upper=alpha_max> alpha;

    // The detector bias.
    real bias;

    // The amount of initial material is not known.
    vector<lower=0, upper=N_initial_max>[N_widgets] N_initial;

    // multinomial
    simplex[3] theta;
}

model {
    for (i in 1:N_widgets) {
        N_measured[i] ~ normal(
          N_initial[i] * exp(-alpha * (t_measured[i] - t_initial[i]))
          + theta[k[i]] * bias,
          sigma_N_measured[i]
        );
    }
}
"""
model = stan.StanModel(model_code=model_str)


# convert detector names to numbers
name_to_k = dict(
    A=1,
    B=2,
    C=3,
)
k = np.vectorize(name_to_k.get)(d_name)

# Data.
data_dict = dict(
    N_widgets=N_widgets,
    t_measured=t,
    N_measured=N,
    sigma_N_measured=N_err,
    N_initial_max=N_initial_max,
    t_initial_max=manufacturing_time_span,
    alpha_max=max_alpha,
    k=k,
)

# initial guess
alpha_guess = max_alpha/2
t_init_guess = np.full(N_widgets, manufacturing_time_span/2)
theta_guess = [0.33, 0.33, 0.34]
bias_guess = 0.

init_dict = dict(
    t_initial=t_init_guess,
    alpha=alpha_guess,
    theta=theta_guess,
    bias=bias_guess,
)

# Run optimisation.
opt_stan = model.optimizing(
    data=data_dict,
    init=init_dict
)

# Run sampling.
samples = model.sampling(**sampling_kwds(
    chains=2,
    iter=2000,
    data=data_dict,
    init=opt_stan,
    control=dict(max_treedepth=30),
))


# plot initial N
fig = samples.traceplot(("N_initial", ))
fig.set_size_inches(10, 5)
plt.savefig('q12_N_init.png', dpi=300)

# plot initial t
fig = samples.traceplot(("t_initial", ))
fig.set_size_inches(10, 5)
plt.savefig('q12_t_init.png', dpi=300)


# get theta chains and remove burn in
theta_chains = samples["theta"][1000:].T


# plot theta chains
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

ax = axs[0]
for i in range(3):
    ax.plot(theta_chains[i])
ax.set_xlabel("Step Number")
ax.set_ylabel(r"$\theta_k$")

ax = axs[1]
for i in range(3):
    ax.hist(theta_chains[i], bins=30, label=rf'$\theta${i+1}')
ax.legend()
ax.set_ylabel("Posterior Probability")
ax.set_xlabel(r"$\theta_k$")
plt.savefig('q12_theta.png', dpi=300)


# get bias chain and remove burn in
bias_chain = samples["bias"][1000:]
# calculate mean and std
bias_mean = np.mean(bias_chain)
bias_err = np.std(bias_chain)

# plot
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

ax = axs[0]
ax.plot(bias_chain, c='k')
ax.set_xlabel("Step Number")
ax.set_ylabel("Bias")

ax = axs[1]
ax.hist(bias_chain, bins=30, color='gray', label='Posterior Distribution')
ax.axvline(bias_mean, c='k', linestyle='--', label='Mean')
ax.set_ylabel("Posterior Probability")
ax.set_xlabel("Bias")
ax.legend()
plt.savefig('q12_bias.png', dpi=300)


# print mean and std
print(f'Bias: {bias_mean:.2}, Standard Deviation: {bias_err:.0}')


# get alpha chain and remove burn in
alpha_chain = samples["alpha"][1000:]
# calculate mean and std
alpha_mean = np.mean(alpha_chain)
alpha_err = np.std(alpha_chain)

# plot
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

ax = axs[0]
ax.plot(alpha_chain, c='k')
ax.set_xlabel("Step Number")
ax.set_ylabel("Alpha")

ax = axs[1]
ax.hist(alpha_chain, bins=30, color='gray', label='Posterior Distribution')
ax.axvline(alpha_mean, c='k', linestyle='--', label='Mean')
ax.set_ylabel("Posterior Probability")
ax.set_xlabel("Alpha")
ax.legend()
plt.savefig('q12_alpha.png', dpi=300)


# print mean and std
print(f'Alpha: {alpha_mean:.2}, Standard Deviation: {alpha_err:.0}')
