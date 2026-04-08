from http.client import CONTINUE
from typing import Optional

from enum import Enum
from pydantic import BaseModel, Field
from yaml import safe_load

class DynamicsEnum(Enum):
    NEW = "new"
    CONTINUE = "continue"
    FROM_FILE = "from_file"

class ThermostatEnum(Enum):
    LANGEVIN = "langevin"
    NOSE_HOOVER = "nose-hoover"
    NONE = "none"

class ParametersEnum(Enum):
    TRAIN = "train"
    READ = "read"

class ForcefieldSettings(BaseModel):
    """ A pydantic model for describing a forcefield. """
    description: str = "Training cell"
    forcefield_file: str = "ml_forcefield.dat"
    structure_file: str = "POSCAR.vasp"

class TrainingSettings(BaseModel):
    """ All the settings relevant for training the force-field. """
    parameters_mode: ParametersEnum = ParametersEnum.TRAIN     # 'train' or 'read'.
    data_file: str = "ml_training_data.dat"
    parameters_file: str = "ml_parameters.dat"
    lasso_keywords: dict = dict(
        fit_intercept = False,
        n_splits = 5)
    #glmnet_keywords: dict = dict(
    #    parallel = True, 
    #    nfolds = 10,
    #    intr = False)
    csld_settings: dict = dict(
        description = "BAs conventional 3x3x3",
        prim = "POSCAR_primitive.vasp", 
        model_type = "LD",
        max_order = 6,
        cluster_diameter = "8.35 3.5 3.5 3.5 3.5",
        cluster_filter = "lambda cls: (True)", 
        cluster_in = "csld_clusters.dat", 
        symC_in = "csld_symc.dat")

class DynamicsSettings(BaseModel):
    """ All settings relevant for molecular dynamics. """
    dynamics_mode: DynamicsEnum = DynamicsEnum.NEW        # 'new' | 'continue' | 'from_file'
    thermostat: ThermostatEnum = ThermostatEnum.LANGEVIN  # Thermostat for the MD. One of [langevin, nose-hoover, none]
    start_iteration: int = 0                              # Current iteration for a restart calculation.
    trajectory_file: str = "md.traj"
    log_file: str = "ml_std.out"
    time_step: float = 1.0                          # Time step for the MD in fs.
    temperature: float = 100                        # Temperature for the MD in K.
    initial_temperature: float = 10                 # Initial temperature in K.
    write_interval: int = 1                         # interval between structures in the trajectory.
    max_iterations: int = 1000                      # Maximum number of iterations.
    langevin_fix_com: bool = True                   # Fix the center of mass when using the Langevin thermostat.
    langevin_friction: float = 0.01                 # Friction parameter for the Langevin thermostat.
    nose_ttime: float = 11.0                        # time factor for the Nose-Hoover thermostat.

class AbinitioSettings(BaseModel):
    """ Settings for the Abinitio calculations. """
    executable: str = ""
    mpi_procs: int = 1
    log_file: str = "std.out"
    encut: float = 500.0
    setups: dict = Field(default_factory = dict)
    kpoints: list = Field(default_factory = list)
    updates: dict = Field(default_factory = dict)

class ActiveLearningSettings(BaseModel):
    initial_iterations: int = 5          # Number of randomised structure for the initial force-field.
    bayesian_factor: float = 3.0         # Prefactor for the Bayesian error as threshold for refitting.
    bayesian_iterations: int = 10        # Number of iterations before another refit can occur.
    max_size: int = 1000                 # Number of iterations over which the Bayesian error is averaged.
    skip_iterations: int = 0             # How many iterations to skip before training the first time.
    bayesian_threshhold: float = 0.02
    group: int = 0

class InternalSettings(BaseModel):
    calculate_phi: bool = False
    parallel: bool = True
    mpi_group: int = 0

class Settings(BaseModel):
    """ A pydantic model for all the settings. """
    forcefield: ForcefieldSettings = ForcefieldSettings()
    supercell: Optional[ForcefieldSettings] = None
    active_learning: ActiveLearningSettings = ActiveLearningSettings()
    dynamics: DynamicsSettings = DynamicsSettings()
    training: TrainingSettings = TrainingSettings()
    abinitio: AbinitioSettings = AbinitioSettings()
    internal: InternalSettings = InternalSettings()

def parse_settings(file_name: str) -> Settings:
    with open(file_name, "r") as input_file:
        inputs = safe_load(input_file)
        return Settings(**inputs)

INCAR_DEFAULTS = dict (
    xc = 'pbe',
    gamma = True,
    istart = 0,
    icharg = 2,
    lreal = 'F',
    algo = 'Normal',
    addgrid = True,
    ispin = 1,
    nelmin = 7,
    ismear = 0,
    sigma = 0.05,
    isif = 2,
    ibrion = -1,
    potim = 0.1,
    nsw = 0,
    lcharg = False,
    lwave = False,
    lplane = True,
    npar = 4,
    nsim = 4,
    ediff = 1e-8,
    ediffg = -1e-3,
    lasph = True,
    prec = 'Accurate'
)
