# Smilei 5.1 side of the solid-density benchmark (bench/collisions_epoch/README.md): hydrogen at 100 n_c, T_e = 100 eV,
# T_i = 10 eV, lnL = 5 for all pairs, frozen species and fields (only the collisions act), 0.5 ps; the same time step as VLPL.
import math
L0 = 2. * math.pi
dt = 0.02 / L0                         # 0.02 / omega in units of L0
Main(geometry="1Dcartesian", number_of_patches=[4], interpolation_order=2, timestep=dt * L0, simulation_time=149.9 * L0,
     time_fields_frozen=1e12, cell_length=[0.01 * L0], grid_length=[48 * 0.01 * L0], EM_boundary_conditions=[["periodic"]],
     reference_angular_frequency_SI=L0 * 3e8 / 1.e-6, print_every=5000, random_seed=7)
for name, m, q, T in [("electron", 1., -1., 100. / 510998.95), ("proton", 1836.15267, 1., 10. / 510998.95)]:
    Species(name=name, position_initialization="regular", momentum_initialization="maxwell-juettner", particles_per_cell=500,
            mass=m, charge=q, number_density=100., temperature=[T], time_frozen=1e12, boundary_conditions=[["periodic", "periodic"]])
for a, b in [("electron", "electron"), ("electron", "proton"), ("proton", "proton")]:
    Collisions(species1=[a], species2=[b], coulomb_log=5.)
for nm in ["electron", "proton"]:
    for q in ["weight", "weight_px", "weight_vx_px", "weight_vy_py", "weight_vz_pz", "weight_ekin", "weight_charge"]:
        DiagParticleBinning(deposited_quantity=q, every=500, species=[nm], axes=[["x", 0., Main.grid_length[0], 1]])
