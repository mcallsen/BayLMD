from ase import Atoms
from typing import Optional

from ase.calculators.vasp import Vasp

from baylmd.context import Context
from baylmd.helpers import make_directories, copy_files, count_files
from baylmd.settings import INCAR_DEFAULTS, Settings
from baylmd.workflow import Workflow

from os import getcwd, chdir, path

import ase.parallel as parallel

_WORK_DIRECTORY = "_work"
_RESULTS_DIRECTORY = "output"

class VaspParallel(Vasp):
    def _run(self, command=None, out=None, directory=None):
        """Method to explicitly execute VASP"""
        if command is None:
            command = self.command
        if directory is None:
            directory = self.directory

        comm = parallel.world.comm

        comm2 = comm.Spawn(command, args=[], maxprocs=comm.size)
        comm2.Disconnect()

        return -1


class AbinitioCalculation(Workflow):
    """ A Workflow performing an ab-initio calculation. """
    def __init__(self, settings: Settings, structure: Optional[Atoms] = None, parent: Optional[Workflow] = None) -> None:
        super().__init__(parent=parent)

        self.settings = settings.abinitio
        self.structure: Optional[Atoms] = structure

        self.my_rank = parallel.world.rank

        # Create the working directory and copy all required input files.
        self.current_directory: str = getcwd()
        self.work_directory: str = path.join(self.current_directory, _WORK_DIRECTORY)
        self.output_directory: str = path.join(self.current_directory, _RESULTS_DIRECTORY)
        make_directories([_WORK_DIRECTORY, _RESULTS_DIRECTORY])

        # Index of the current structure, used for labeling the output files.
        self.structure_index: int = count_files(self.output_directory, "OUTCAR_*") + 1

        # Setup the input for the VASP calculation.
        self.inputs: dict = INCAR_DEFAULTS
        self.inputs.update(settings.abinitio.updates)

        command = f"mpirun -np {settings.abinitio.mpi_procs} {settings.abinitio.executable} > {settings.abinitio.log_file}"

        self.inputs["command"] = command
        self.inputs["encut"] = settings.abinitio.encut
        self.inputs["setups"] = settings.abinitio.setups
        self.inputs["kpts"] = settings.abinitio.kpoints

    def update(self, context: Context) -> None:
        """ Perform an ab-initio calculation to determine the forces. """
        chdir(self.work_directory)

        self.structure = context.structure.copy()

        # Calculate the forces acting on the atoms using VASP.
        calculation = Vasp(**self.inputs)

        self.structure.set_calculator(calculation)

        print("Call get_forces")
        context.forces = self.structure.get_forces().flatten()

        # Save copies of the VASP output files.
        if self.my_rank == 0:
            copy_files(self.work_directory, self.output_directory, ["POSCAR", "OUTCAR"], f"_{self.structure_index}")

            # Write the current sigma to a file.
            # TODO: find another place for this.
            #if context.sigma is not None:
            #    np.savetxt(self.output_directory + f"/sigma_{self.structure_index}", context.sigma)

        self.structure_index += 1

        chdir(self.current_directory)

    @classmethod
    def print_settings(cls, settings: Settings) -> None:
        print(f"\n{cls.__name__}\n")
        print("    Cutoff energy (eV):   ", settings.abinitio.encut)
        kpoints = " ".join(str(k) for k in settings.abinitio.kpoints)
        print("    K-point grid:         ", kpoints)
        pseudos = " ".join([ element + postfix for element, postfix in settings.abinitio.setups.items()])
        print("    Pseudo potentials:    ", pseudos)
        print("    VASP executable:      ", settings.abinitio.executable)
        print("    Log file:             ", settings.abinitio.log_file)
        print("")
