import numpy as np
import matplotlib.pyplot as plt


alpha = 1.
r0 = 1.
d0 = 1.
rc = 3.
rvals = np.linspace(0,5, 100)


def phi(r):
    value = d0*(np.exp(-2.*alpha*(r-r0)) - 2.*np.exp(-1.*alpha*(r-r0)))
    return value

def dphi_dr(r):
    value = 2.*d0*alpha*(np.exp(-1.*alpha*(r-r0)) - np.exp(-2.*alpha*(r-r0)))
    return value

def calc_e(r):

    phi_r = phi(r)
    phi_rc = phi(rc)
    dphi_dr_rc = dphi_dr(rc)
    value = phi_r - phi_rc - (r-rc)*dphi_dr_rc
    return value

#phi_r = phi(r)
#phi_rc = phi(rc)
#dphi_dr_rc = dphi_dr(rc)

energy = calc_e(rvals)


fig, ax = plt.subplots()
ax.set_yticks([0.0], minor=False)
ax.yaxis.grid(True, which='major')
ax.yaxis.grid(True, which='minor')

plt.plot(rvals, energy, 'k-')
plt.savefig("morse_plot.png", dpi=500)