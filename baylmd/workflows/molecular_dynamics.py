from typing import Optional, Type
from numpy.typing import ArrayLike

from ase import Atoms
from ase.io import Trajectory

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

from baylmd import parallel

from baylmd.context import Context
from baylmd.estimator import GaussianEstimator
from baylmd.helpers import initialize_file, count_lines
from baylmd.forcefields import ForcefieldModel
from baylmd.forcefield_calculator import ForcefieldCalculator
from baylmd.dynamics import dynamics_factory
from baylmd.settings import Settings, ThermostatEnum, DynamicsSettings, DynamicsEnum
from baylmd.workflow import Workflow

import numpy as np
import os

BAYESIAN_HEADER: str = "#Iteration    Total Energy (eV)    Bayesian Error (eV/A)    Avg. Bayesian Error (eV/A)    Temperature (K)\n"
SUPERCELL_HEADER: str = "#Iteration    Temperature L (K)    Temperature R (K)    Delta T (K)    Temperature (K)\n"

class MolecularDynamics(Workflow):
    """ A workflow performing molecular dynamics. """
    def __init__(self, structure: Atoms, settings: Settings, model:ForcefieldModel = None, forcefield_class: Type = ForcefieldModel, parent: Optional[Workflow] = None) -> None:
        super().__init__(parent=parent)
        self.settings: DynamicsSettings = settings.dynamics

        # Read the ForceModel and setup the corresponding calculator.
        self.model = model if model is not None else forcefield_class(settings)

        # Read the parameters from a file.
        parameters = np.loadtxt(settings.training.parameters_file)
        self.model.set_parameters(parameters)

        # Read the equilibrium structure and set the forcefield calculator.
        self.equilibrium_structure: Atoms = structure
        self.calculator = ForcefieldCalculator(structure, self.model)

        # Create the log file, if it does not exist.
        self.log_file = self.settings.log_file + f"_{parallel.index}"

        header = BAYESIAN_HEADER if settings.internal.calculate_phi else SUPERCELL_HEADER

        # initialise the log- and trajectory_file.
        if parallel.group_comm.rank == 0: 
            message = initialize_file(self.log_file, default_content=header)
            if message is not None:
                print(f"[{self.__class__.__name__}]", message)

        if not os.path.isfile(settings.dynamics.trajectory_file):
            with Trajectory(settings.dynamics.trajectory_file, mode="w") as trajectory:
                trajectory.write(structure)
                trajectory.close()

        # NOTE: the log file might not exist yet at this point. Wait for others to catch up 
        parallel.group_comm.Barrier()

        # Set the current iteration based on the number of lines in the log file. 
        self.iteration = count_lines(self.log_file) - 1
        self.settings.start_iteration = self.iteration

        # Check whether iteration number is consistent with the dynamics mode.
        if self.iteration > 0 and self.settings.dynamics_mode is DynamicsEnum.NEW:
            self.settings.dynamics_mode = DynamicsEnum.CONTINUE

        # Set the atoms object the calculator is running on.
        trajectory = Trajectory(self.settings.trajectory_file)
        self.atoms: Atoms = trajectory[-1].copy()
        trajectory.close()

        self.atoms.calc = self.calculator

        # Set initial velocities if we start from scratch.
        if self.settings.dynamics_mode is DynamicsEnum.NEW and self.settings.thermostat is ThermostatEnum.NOSE_HOOVER:
            # Starting from scratch. Initialise the velocities.
            MaxwellBoltzmannDistribution(self.atoms, temperature_K = self.settings.initial_temperature)

            # set center of mass movement and angular momentum to 0.
            Stationary(self.atoms)
            ZeroRotation(self.atoms)

        self.dynamics = dynamics_factory(self.atoms, self.settings)

        # Read the GaussianEstimator from a file for the Bayesian error estimation.
        self.estimator = GaussianEstimator.from_file('./_tmp/smatrix.dat')

    @property
    def displacements(self) -> ArrayLike:
        return self.calculator.displacements

    def update(self, context: Context, nsteps: int = 1) -> None:

        if self.settings.thermostat is ThermostatEnum.NONE:
            # Remove center of mass velocity, which for some reason ASE does not do.
            Stationary(self.atoms, preserve_temperature=False)

        self.dynamics.run(nsteps)

        # Update the context.
        self.iteration += nsteps
        context.iteration = self.iteration

        context.structure = self.atoms
        context.displacements = self.calculator.displacements

        context.temperature = self.atoms.get_temperature()
        context.total_energy = self.atoms.get_total_energy()

    @classmethod
    def print_settings(cls, settings: Settings) -> None:
        print(f"\n{cls.__name__}\n")
        print("    Dynamics mode:             ", settings.dynamics.dynamics_mode.value)
        if settings.dynamics.dynamics_mode is DynamicsEnum.CONTINUE:
            print("    Start iteration:           ", settings.dynamics.start_iteration)
        print("    Max iterations:            ", settings.dynamics.max_iterations)
        print("    Time step (fs):            ", settings.dynamics.time_step)
        print("    Write interval (steps)     ", settings.dynamics.write_interval)
        print("")
        print("    Thermostat:                ", settings.dynamics.thermostat.value)
        if settings.dynamics.thermostat is not ThermostatEnum.NONE:
            print("    Temperature (K):           ", settings.dynamics.temperature)
            print("    Initial temperature (K):   ", settings.dynamics.initial_temperature)
        if settings.dynamics.thermostat is ThermostatEnum.LANGEVIN:
            print("    Langevin friction:         ", settings.dynamics.langevin_friction)
            print("    Fix center of mass:        ", settings.dynamics.langevin_fix_com)
        if settings.dynamics.thermostat is ThermostatEnum.NOSE_HOOVER:
            print("    Nose-Hoover time scale:    ", settings.dynamics.nose_ttime)
        print("")

    def set_temperature(self, temperature):
        self.dynamics.set_temperature(temperature_K=temperature)

    def get_displacements(self, atoms: Atoms) -> ArrayLike:
        return self.calculator.get_displacements(atoms)

    def print_context(self, context: Context, function):
        if parallel.group_comm.rank == 0:
            with open(self.log_file, "a") as fo:
                fo.write(function(context))
        parallel.group_comm.barrier()

    def bayesian_error(self) -> float:
        """ Compute the Bayesian error for the current Phi."""
        return self.estimator.score(self.model.current.toarray())
       
    def sigma(self) -> float:
        """ Compute the Bayesian error for the current Phi."""
        return self.estimator.get_sigma(self.model.current.toarray())
 
