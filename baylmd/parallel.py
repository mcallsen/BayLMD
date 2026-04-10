from mpi4py import MPI

# Some global variables for the MPI processes.
world: MPI.Comm = MPI.COMM_WORLD
group_comm: MPI.Comm = world
inter_comm: MPI.Comm = world
index = 0

def initialize_groups(number: int) -> None:
    """ Split the world communicator into a number of smaller groups."""
    index = world.rank % world.size // number
    group_comm = world.Split(index, world.rank)
    inter_comm = set_intercomm(group_comm)

def set_intercomm(comm: MPI.Comm) -> MPI.Comm:
    """ Setup a communicator between the rank 0 processes in comm. """
    color = 0 if comm.rank == 0 else MPI.UNDEFINED
    return world.Split(color, world.rank)

