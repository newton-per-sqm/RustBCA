from libRustBCA import *
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(__file__)+'/../scripts')
sys.path.append('scripts')
from materials import *
from formulas import *


'''
This script is intended to serve as an example of using the Python 
interface for a multi-layer, multi-species simulation.

In this simulation, 50 He and 50 H ions are incident on a target 
of two layers, one titanium boride and one aluminum:

_____________
|           |
|    TiB2   | dx = 100 A
_____________
|           |
|     Al    | dx = 1 um
|           |

'''
number_ions = 10000
angle = 45.0 # angles in BCA codes are typically measured from the surface normal.
energy = 1000.0 # eV

ux = [np.cos(angle*np.pi/180.0)]*number_ions
uy = [np.sin(angle*np.pi/180.0)]*number_ions
uz = [0.0]*number_ions

energies = [energy]*number_ions
ion1 = hydrogen
ion2 = helium

# Ion properties are per-ion; so we have a list of number_ions/2 of each:
Z1 = [ion1["Z"]]*(number_ions//2) + [ion2["Z"]]*(number_ions//2)
m1 = [ion1["m"]]*(number_ions//2) + [ion2["m"]]*(number_ions//2)
Ec1 = [ion1["Ec"]]*(number_ions//2) + [ion2["Ec"]]*(number_ions//2)
Es1 = [ion1["Es"]]*(number_ions//2) + [ion2["Es"]]*(number_ions//2)

# Material properties are per species; so we have a list of 3 species:
Z2 = [titanium["Z"], boron["Z"], aluminum["Z"]]
m2 = [titanium["m"], boron["m"], aluminum["m"]]
Ec2 = [titanium["Ec"], boron["Ec"], aluminum["Ec"]]
Es2 = [titanium["Es"], boron["Es"], aluminum["Es"]]
Eb2 = [titanium["Eb"], boron["Eb"], aluminum["Eb"]]

# Densities (n2) are specified as a list of layers,
# each of which has a list of the densities per-species:

#top layer, titanium diboride:
# n_i calculated from ni = rho_TiB2 / (m_Ti + 2 * m_B)
ni = 3.8e28 # 1/m3

nTi1 = ni/10**30 # 1/A^3
nB1 = 2 * ni/10**30  # 1/A^3
nAl1 = 0.0

# bottom layer, pure aluminum:
nTi2 = 0.0
nB2 = 0.0
nAl2 = aluminum["n"]/10**30 # 1/A^3

n2 = [
    [nTi1, nB1, nAl1], # top layer
    [nTi2, nB2, nAl2] # bottom layer
]

dx = [
    100.0, # Angstrom, top layer
    1.0*1e-6/1e-10 # Angstrom; bottom layer
]

# compound_bca_list_py provides three return values
output, incident, stopped = compound_bca_list_1D_py(
    ux,
    uy,
    uz,
    energies,
    Z1,
    m1,
    Ec1,
    Es1,
    Z2,
    m2,
    Ec2,
    Es2,
    Eb2,
    n2,
    dx
)

# output columns = [Z, m (amu), E (eV), x, y, z, (angstrom), ux, uy, uz]
output = np.array(output)
Z = output[:, 0]
E = output[:, 0]

# implanted ions can be selected using the stopped value
x_He = output[np.logical_and(Z == helium["Z"], stopped), 3]
x_H = output[np.logical_and(Z == hydrogen["Z"], stopped), 3]

num_bins = 50
bins = np.linspace(0.0, 400.0, num_bins)
plt.hist(x_He, bins=bins, histtype='step', label='He')
plt.hist(x_H, bins=bins, histtype='step', label='H')
plt.plot([dx[0], dx[0]], [0.0, number_ions], linestyle='--', color='gray')
plt.ylim([0.0, 375])
plt.xlabel('x [A]')
plt.ylabel('f(x) [counts]')
plt.title('H/He implantation in TiB2 on Al')
plt.legend()
plt.show()