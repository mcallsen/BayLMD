
import numpy as np
from forcefield_ml.settings import DynamicsSettings, DynamicsEnum, ThermostatEnum

from forcefield_ml.temperature import create_bins

from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.md.md import MolecularDynamics
from ase.md.npt import NPT

from ase import units, Atoms
from ase.io import read

import ase.parallel as parallel


class TrajectoryReader(object):
    def __init__(self, atoms: Atoms, trajectory: str, settings: DynamicsSettings) -> None:
        # read the trajectory
        self.atoms = atoms
        self.trajectory = read(trajectory, format="vasp-xdatcar", index=":")
        self.iteration = settings.start_iteration

    def run(self, nsteps: int) -> None:
        # increment the current iteration by nsteps.
        self.iteration += nsteps

        # Update the positions from the trajectory.
        self.atoms.set_scaled_positions(self.trajectory[self.iteration].get_scaled_positions())

        # call atoms.get_forces, which will also calculate the phi matrix for the Bayesian error.
        self.atoms.get_forces()

        # TODO: Update the velocities.

class LocalLangevin(Langevin):
    def updatevars(self):
        dt = self.dt
        T = self.temp
        fr = self.fr
        masses = self.masses.flatten()
        sigma = np.sqrt(2 * T * fr / masses)
        sigma.shape = (-1, 1)

        self.c1 = dt / 2. - dt * dt * fr / 8.
        self.c2 = dt * fr / 2 - dt * dt * fr * fr / 8.
        self.c3 = np.sqrt(dt) * sigma / 2. - dt**1.5 * fr * sigma / 8.
        self.c5 = dt**1.5 * sigma / (2 * np.sqrt(3))
        self.c4 = fr / 2. * self.c5

class NoseHoover(NPT):
    def initialize(self):
        """Initialize the dynamics.

        The dynamics requires positions etc for the two last times to
        do a timestep, so the algorithm is not self-starting.  This
        method performs a 'backwards' timestep to generate a
        configuration before the current.

        This is called automatically the first time ``run()`` is called.
        """
        # print "Initializing the NPT dynamics."
        dt = self.dt
        atoms = self.atoms
        self.h = self._getbox()
        self.inv_h = np.linalg.inv(self.h)
        # The contents of the q arrays should migrate in parallel simulations.
        # self._make_special_q_arrays()
        self.q = np.dot(self.atoms.get_positions(), self.inv_h) - 0.5
        # zeta and eta were set in __init__
        self._initialize_eta_h()
        deltazeta = dt * self.tfact * (atoms.get_kinetic_energy() -
                                       self.desiredEkin)
        self.zeta_past = self.zeta - deltazeta
        self._calculate_q_past_and_future()
        self.initialized = 1

    def _count_degrees_of_freedom(self):
        n_atoms = len(self.atoms)
        return 3 * n_atoms - 3

    def _calculateconstants(self):
        """(Re)calculate some constants when pfactor,
        ttime or temperature have been changed."""

        degrees_of_freedom = self._count_degrees_of_freedom()
        if self.ttime is None:
            self.tfact = 0.0
        else:
            self.tfact = 2.0 / (degrees_of_freedom * self.temperature *
                                self.ttime * self.ttime)
        if self.pfactor_given is None:
            self.pfact = 0.0
        else:
            self.pfact = 1.0 / (self.pfactor_given * np.linalg.det(self._getbox()))
            # self.pfact = 1.0/(n * self.temperature * self.ptime * self.ptime)
        self.desiredEkin = 0.5 * degrees_of_freedom * self.temperature


def dynamics_factory(atoms, settings: DynamicsSettings):
    """ Create a suitable 'dynamics' object based on the settings. """
    if settings.dynamics_mode is DynamicsEnum.FROM_FILE:
        return TrajectoryReader(atoms, trajectory = settings.trajectory_file, settings = settings)

    # We are doing molecular dynamics so decide which thermostat to use.
    if settings.thermostat is ThermostatEnum.LANGEVIN:
        return Langevin(atoms, settings.time_step * units.fs,
            trajectory = settings.trajectory_file,
            append_trajectory = settings.dynamics_mode is DynamicsEnum.CONTINUE,
            temperature_K = settings.temperature,
            friction = settings.langevin_friction,
            fixcm = settings.langevin_fix_com,
            loginterval = settings.write_interval
        )
    elif settings.thermostat is ThermostatEnum.NONE:
        return VelocityVerlet(atoms, settings.time_step * units.fs,
            trajectory = settings.trajectory_file,
            append_trajectory = settings.dynamics_mode is DynamicsEnum.CONTINUE,
            loginterval = settings.write_interval
        )
    elif settings.thermostat is ThermostatEnum.NOSE_HOOVER:
        return NoseHoover(atoms, settings.time_step * units.fs,
            trajectory = settings.trajectory_file,
            append_trajectory = settings.dynamics_mode is DynamicsEnum.CONTINUE,
            temperature_K = settings.temperature,
            ttime = settings.nose_ttime * units.fs,
            externalstress = 1.0 * units.bar,
            pfactor = None,
            loginterval = settings.write_interval
        )
    return None
