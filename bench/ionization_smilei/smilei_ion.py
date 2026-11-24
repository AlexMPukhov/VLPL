# Smilei 5.1 input of the field-ionization benchmark (bench/ionization_smilei/README.md): the same setup as ion.deck in
# 1Dcartesian. Normalized units of lambda0 = 0.8 um (lengths 1/k0, times 1/omega0), as VLPL's code units.
import math
lam = 0.8e-6
um = 2 * math.pi / 0.8                 # 1 um in 1/k0
fs = 2 * math.pi * 299792458.0 / lam * 1e-15
dx = 0.0125 * um
a0, tau, tc = 0.03, 10 * fs, 25 * fs    # tau: FWHM of the intensity

def env(t):                            # VLPL's laser: a0 exp(-2 ln2 (t - tc)^2 / tau^2) sin(t - tc)
    return a0 * math.exp(-2 * math.log(2) * (t - tc) ** 2 / tau ** 2) * math.sin(t - tc)

def slab(n):
    return lambda x: n if 12 * um <= x < 28 * um else 0.0

Main(
    geometry = "1Dcartesian",
    interpolation_order = 2,
    cell_length = [dx],
    grid_length = [44 * um],
    number_of_patches = [32],
    timestep = 0.95 * dx,
    simulation_time = 140 * fs,
    EM_boundary_conditions = [["silver-muller"]],
    reference_angular_frequency_SI = 2 * math.pi * 299792458.0 / lam,
    print_every = 1000,
)

Laser(box_side = "xmin", space_time_profile = [lambda t: 0.0, env])

Species(name = "hydrogen", ionization_model = "tunnel", ionization_electrons = "electrons", atomic_number = 1,
        position_initialization = "regular", momentum_initialization = "cold", particles_per_cell = 4,
        mass = 1836.0 * 1e6, charge = 0.0, number_density = slab(0.1), boundary_conditions = [["remove", "remove"]])
Species(name = "carbon", ionization_model = "tunnel", ionization_electrons = "electrons", atomic_number = 6,
        position_initialization = "regular", momentum_initialization = "cold", particles_per_cell = 64,
        mass = 1836.0 * 12 * 1e6, charge = 0.0, number_density = slab(1e-5), boundary_conditions = [["remove", "remove"]])
Species(name = "electrons", position_initialization = "regular", momentum_initialization = "cold", particles_per_cell = 0,
        mass = 1.0, charge = -1.0, charge_density = 0.0, boundary_conditions = [["remove", "remove"]])

DiagScalar(every = 20)
DiagParticleBinning(deposited_quantity = "weight", every = 200, species = ["hydrogen"], axes = [["charge", -0.5, 1.5, 2]])
DiagParticleBinning(deposited_quantity = "weight", every = 200, species = ["carbon"], axes = [["charge", -0.5, 6.5, 7]])
