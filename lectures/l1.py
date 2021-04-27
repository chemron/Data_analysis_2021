import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse

# For reproducibility
np.random.seed(0)

# Load the data.
_, x, y, y_err, x_err, rho_xy = data = np.array([
    [1,  201, 592, 61,  9, -0.84],
    [2,  244, 401, 25,  4, +0.31],
    [3,   47, 583, 38, 11, +0.64],
    [4,  287, 402, 15,  7, -0.27],
    [5,  203, 495, 21,  5, -0.33],
    [6,   58, 173, 15,  9, +0.67],
    [7,  210, 479, 27,  4, -0.02],
    [8,  202, 504, 14,  4, -0.05],
    [9,  198, 510, 30, 11, -0.84],
    [10, 158, 416, 16,  7, -0.69],
    [11, 165, 393, 14,  5, +0.30],
    [12, 201, 442, 25,  5, -0.46],
    [13, 157, 317, 52,  5, -0.03],
    [14, 131, 311, 16,  6, +0.50],
    [15, 166, 400, 34,  6, +0.73],
    [16, 160, 337, 31,  5, -0.52],
    [17, 186, 423, 42,  9, +0.90],
    [18, 125, 334, 26,  8, +0.40],
    [19, 218, 533, 16,  6, -0.78],
    [20, 146, 344, 22,  5, -0.56],
]).T

# Helpful function.
def _ellipse(x, y, cov, scale=2, **kwargs):
    vals, vecs = np.linalg.eig(cov)
    theta = np.degrees(np.arctan2(*vecs[::-1, 0]))
    w, h = scale * np.sqrt(vals)

    kwds = dict(lw=0.5, color="r")
    kwds.update(**kwargs)

    ellipse = Ellipse(xy=[x, y], 
                      width=w, height=h, angle=theta,
                      **kwds)
    ellipse.set_facecolor("none")
    return ellipse


covs = np.array([[[x_e**2, x_e*y_e*rho],
                  [x_e*y_e*rho, y_e**2]] \
                  for y_e, x_e, rho in zip(*data[3:])])

fig, ax = plt.subplots(figsize=(4, 4))
    
ax.scatter(x, y, c="k", s=10)

for xi, yi, cov in zip(x, y, covs):
    ax.add_artist(_ellipse(xi, yi, cov))

# least squares
Y = np.atleast_2d(y).T

A = np.vstack([np.ones_like(x), x]).T
C = np.diag(y_err * y_err)

C_inv = np.linalg.inv(C)

G = np.linalg.inv(A.T @ C_inv @ A)
X = G @ (A.T @ C_inv @ Y)

b, m = X.T[0]

xl = np.array([0, 300])
ax.plot(xl, xl * m + b, 
        "-", c="tab:blue",
        lw=2, zorder=-1)


draws = np.random.multivariate_normal(X.T[0], G, 50)
for b_, m_ in draws:
    ax.plot(xl, xl * m_ + b_,
            "-", c="tab:blue", 
            lw=0.5, zorder=-1,
            alpha=0.1)
    
ax.set_xlim(0, 300)
ax.set_ylim(0, 700)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
    
ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(7))
    
fig.tight_layout()
plt.savefig("plot.png")
plt.show()


bins = 250
m_bounds = (0.5, 1.5)
b_bounds = (150, 300)
    
m_bins = np.linspace(*m_bounds, bins)
b_bins = np.linspace(*b_bounds, bins)
    
M, B = np.meshgrid(m_bins, b_bins)
    
# Calculate chi^2
chi_sq = np.sum([
    (y_ - (M * x_ + B))**2 / y_err_**2 for (x_, y_, y_err_) in zip(x, y, y_err)
], axis=0)
    
# Pro-Tip(tm): The 'origin' and 'extent' keywords in plt.imshow can give unexpected behaviour.
#              You should *always* make sure you are plotting the right orientation.
imshow_kwds = dict(
    origin="lower",
    # extent=(*m_bounds, *b_bounds),
    # aspect=np.ptp(m_bounds)/np.ptp(b_bounds),
)
    
fig_chi_sq, ax = plt.subplots(figsize=(5, 4))
imshow_chi_sq = ax.imshow(chi_sq, 
                          cmap="Blues_r",
                          **imshow_kwds)
cbar_chi_sq = plt.colorbar(imshow_chi_sq)
cbar_chi_sq.set_label(r"$\chi^2$")

ax.set_xlabel(r"$m$")
ax.set_ylabel(r"$b$")
ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(6))

# Plot the error ellipse we found from linear algebra.
color = "#000000"
ax.scatter(*X.T[0],
           facecolor=color,
           s=10, zorder=10)
ax.add_artist(_ellipse(m, b, G[::-1, ::-1], 
                       scale=3,
                       color=color))
fig_chi_sq.tight_layout()
plt.show()
