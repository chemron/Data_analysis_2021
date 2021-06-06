import scipy.optimize as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
from scipy.signal import lombscargle
from george import kernels
import george

# load data
with open("assignment2_gp.pkl", "rb") as fp:
    data = pickle.load(fp)
t = data['t']
y = data['y']
y_err = data['yerr']

# plot the data
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(t, y, c="k", s=2)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$y$")
ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(7))
fig.tight_layout()
plt.savefig('q2_data.png', dpi=300)


# get periodogram
f = np.linspace(0.001, 2, 1000)
pgram = lombscargle(t, y, f)


# find index of peaks
peak_i = np.where(
    np.r_[True, pgram[1:] > pgram[:-1]] & np.r_[pgram[:-1] > pgram[1:], True]
)[0]
# power of peaks
peak_p = pgram[peak_i]

# exclude peaks with <100 power
peak_i = peak_i[np.where(peak_p > 100)]
peak_p = pgram[peak_i]
# frequency of peaks
peak_f = f[peak_i]

# plot
plt.figure(figsize=(8, 5))
plt.plot(f, pgram, c='k')
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.vlines(peak_f, 0, peak_p, linestyle='--', colors='gray')
plt.savefig('periodogram.png', dpi=300)

# get periods (for later)
periods = 2*np.pi / peak_f
print('Frequency: \t Power: \t Periods:')
for freq, power, T in zip(peak_f, peak_p, periods):
    print(f'{freq:.3}\t \t {int(power)} \t \t {int(T)}')


# Part 2


# define our kernal
k1 = kernels.ExpSine2Kernel(gamma=1, log_period=np.log(periods[1]))
k2 = kernels.ExpSine2Kernel(gamma=1, log_period=np.log(periods[3]))

kernel = k1 + k2
kernel *= 1 * kernels.ExpSquaredKernel(1000)

# compute the gaussian process
gp = george.GP(kernel)
gp.compute(t, y_err)

t_min = np.min(t)
t_max = np.max(t)
t_pred = np.linspace(t_min, t_max + 60, 500)
pred, pred_var = gp.predict(y, t_pred, return_var=True)


plt.figure(figsize=(8, 5))
plt.fill_between(t_pred, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var),
                 color="k", alpha=0.2)
plt.plot(t_pred, pred, "k", lw=1.5, alpha=0.5)
plt.scatter(t, y, c="k", s=2)
plt.xlim(t_min, t_max + 60)
plt.xlabel("Day")
plt.ylabel("Brightness")
plt.savefig('gp1.png', dpi=300)

print(f"Initial ln-likelihood: {gp.log_likelihood(y):.2f}")


# Define the objective function (negative log-likelihood in this case).
def nll(p):
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else 1e25


# And the gradient of the objective function.
def grad_nll(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y, quiet=True)


# Run the optimization routine.
p0 = gp.get_parameter_vector()
results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")

# Update the kernel and print the final log-likelihood.
gp.set_parameter_vector(results.x)
print(f"Final ln-likelihood: {gp.log_likelihood(y):.2f}")

t_min = np.min(t)
t_max = np.max(t)
t_pred = np.linspace(t_min, t_max + 60, 500)
mu, var = gp.predict(y, t_pred, return_var=True)


# plot
plt.figure(figsize=(8, 5))
plt.fill_between(t_pred, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var),
                 color="k", alpha=0.2)
plt.plot(t_pred, pred, "k", lw=1.5, alpha=0.5)
plt.scatter(t, y, c="k", s=2)
plt.xlim(t_min, t_max + 60)
plt.xlabel("Day")
plt.ylabel("Brightness")
plt.savefig("gp2.png", dpi=300)
