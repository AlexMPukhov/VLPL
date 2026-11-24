# Smilei 5.1 namelist of the collision benchmarks (bench/collisions_smilei/README.md): CASE=thermal|iso|beam|stop|ion smilei smilei_coll.py
# The parameters are those of Smilei's own benchmarks (benchmarks/collisions/thermalisation_ei1.py, temperature_isotropization1.py,
# beam_relaxation1.py, Stopping_power1.py, ionization_rate1.py), with fewer cells; all species frozen (time_frozen), fields frozen:
# only the collisions act. Diagnostics: per species one ParticleBinning over the whole box for each of the moments in MOMENTS.
import math, os
L0 = 2. * math.pi
CASE = os.environ.get("CASE", "thermal")
P = {
    #          dt     time   cell  ncells
    "thermal": (0.2,  20.,   10.,  100),
    "iso":     (0.01, 0.4,   4.,   1000),
    "beam":    (0.2,  40.,   2.,   56),
    "stop":    (1.,   50.,   0.2,  120),
    "ion":     (0.05, 100.,  20.,  24),
}[CASE]
dt, T, cell, nc = P
Main(geometry="1Dcartesian", number_of_patches=[4], interpolation_order=2, timestep=dt * L0, simulation_time=T * L0,
     time_fields_frozen=1e12, cell_length=[cell * L0], grid_length=[nc * cell * L0], EM_boundary_conditions=[["periodic"]],
     reference_angular_frequency_SI=L0 * 3e8 / 1.e-6, print_every=100, random_seed=7)
def species(name, m, q, n, ppc, T, v=0., Z=0):
    kw = dict(atomic_number=Z) if Z else {}
    Species(name=name, position_initialization="regular", momentum_initialization="maxwell-juettner", particles_per_cell=ppc,
            mass=m, charge=q, number_density=n, mean_velocity=[v, 0., 0.], temperature=T if isinstance(T, list) else [T],
            time_frozen=1e12, boundary_conditions=[["periodic", "periodic"]], **kw)
names = []
if CASE == "thermal":
    species("ion", 10., 1., 10., 5000, 0.00018); species("electron", 1., -1., 10., 5000, 0.0002)
    Collisions(species1=["electron"], species2=["ion"], coulomb_log=5)
    Collisions(species1=["electron"], species2=["electron"], coulomb_log=1000)
    Collisions(species1=["ion"], species2=["ion"], coulomb_log=1000)
    names, every = ["electron", "ion"], 5
elif CASE == "iso":
    species("electron", 1., -1., 10., 2000, [0.000011, 0.00001, 0.00001])
    Collisions(species1=["electron"], species2=["electron"], coulomb_log=2)
    names, every = ["electron"], 2
elif CASE == "beam":
    species("ion", 10., 1., 10., 1000, 0.00002); species("electron", 1., -1., 10., 1000, 0.0000002, 0.05)
    Collisions(species1=["electron"], species2=["ion"], coulomb_log=3)
    names, every = ["electron", "ion"], 10
elif CASE == "stop":
    species("background", 1., -1., 10., 1000, 0.01)
    for k, v in enumerate([0.2, 0.27, 0.33]):
        species("beam%d" % k, 1., -1., 0.00001, 1000, 0.0000001, v)
        Collisions(species1=["background"], species2=["beam%d" % k], coulomb_log=3.)
    names, every = ["beam0", "beam1", "beam2"], 2
elif CASE == "ion":
    species("electron", 1., -1., 1., 400, 0.0000001, 0.03)
    species("ion", 1836. * 13., 3., 1. / 3., 400, 0.00000001, 0., 13)
    Collisions(species1=["electron"], species2=["ion"], coulomb_log=3, ionizing=True)
    names, every = ["electron", "ion"], 40
MOMENTS = ["weight", "weight_px", "weight_vx_px", "weight_vy_py", "weight_vz_pz", "weight_ekin", "weight_charge"]
for nm in names:
    for q in MOMENTS:
        DiagParticleBinning(deposited_quantity=q, every=every, species=[nm], axes=[["x", 0., Main.grid_length[0], 1]])
