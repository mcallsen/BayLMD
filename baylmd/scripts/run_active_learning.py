from ase import Atoms

from baylmd import parallel

from baylmd.context import Context, to_string_bayesian
from baylmd.parsing import parse_structure
from baylmd.settings import parse_settings

from baylmd.workflows.active_learning import ActiveLearning


def main() -> None:
    settings = parse_settings("settings.yaml")
    settings.internal.calculate_phi = True

    # TODO: Figure out how many groups there are.

    parallel.initialize_groups(1)

    # Read the structure file for each group.
    structure: Atoms = parse_structure(settings.forcefield.structure_file)

    if parallel.group_comm.rank == 0:
        print("[run active learning] setup ")
    parallel.group_comm.barrier()

    learning = ActiveLearning(structure, settings)

    if parallel.group_comm.rank == 0:
        print("[run active learning] setup done ")
    parallel.group_comm.barrier()

    context = Context()
    learning.start(context)

    refit_required: bool = False

    if parallel.group_comm.rank == 0:
        print("[run active learning] start done ")
    parallel.group_comm.barrier()

    if parallel.group_comm.rank == 0:
        print("[run active learning]", refit_required, context.iteration, settings.dynamics.max_iterations)
    parallel.group_comm.barrier()

    # Run one iteration to update the VASP criterion.
    learning.update(context)
    parallel.inter_comm.allreduce(refit_required, parallel.MPI.LOR)
    learning.molecular_dynamics.print_context(context, to_string_bayesian)    
    if parallel.group_comm.rank == 0:
        with open("dummy.dat", "a") as fo:
             fo.write(f" {context.iteration} {context.bayesian_error} {context.vasp_criterion}\n")
    parallel.group_comm.barrier()

    # run the active learning until either a new structure is found or the max number of steps has been reached.
    while not refit_required and context.iteration < settings.dynamics.max_iterations:
        learning.update(context)

        refit_required = context.refit_required

        # Use MPI_Allreduce to let all processes know whether a refit is required.
        # NOTE: minor case MPI functions for python objects (e.g. single variable)
        #       major case MPI functions for buffer objects (e.g. numpy array)
        parallel.inter_comm.allreduce(refit_required, parallel.MPI.LOR)
        
        learning.molecular_dynamics.print_context(context, to_string_bayesian)

    # Write the current state of active learning to a file.
    learning.end()

    if parallel.group_comm.rank == 0:
        print("[run active learning] end ")
    parallel.group_comm.barrier()

if __name__ == "__main__":
    main()
