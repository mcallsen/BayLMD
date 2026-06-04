#pragma once

#include <optional>
#include <span>
#include <utility>

#include <mpi.h>

namespace mpi {
    // MPI Datatypes for the C++ type system. NOTE 1: Would be great to constexpr these but alas.
    // NOTE 2: Template specialisation violate ODR without inline.
    template<typename T> MPI_Datatype MPItype;
    template<> inline MPI_Datatype MPItype<int> = MPI_INT;
    template<> inline MPI_Datatype MPItype<double> = MPI_DOUBLE;

    // Optional MPI_Comm. No value means serial execution.
    class communicator {
        public:
        communicator() = default;
        communicator(MPI_Comm comm) : _comm(comm) {
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &size);
        }

        auto operator == (const communicator & other) const -> bool = default;

        static communicator world() { return communicator(MPI_COMM_WORLD); }

        explicit operator bool () const { return _comm.has_value(); }

        // Implicit conversion to MPI_Comm. NOTE: not sure whether this is good practice.
        operator MPI_Comm () const { return _comm.value(); }
        
        // Get a reference to the MPI_Comm. NOTE: I could not figure out how to do this with implicit conversion.
        // So I follow the STD implementation.
        auto value() & -> MPI_Comm & { return _comm.value(); }
        auto value() const & -> const MPI_Comm & { return _comm.value(); }

        auto value() && -> MPI_Comm && { return std::forward<MPI_Comm>(_comm.value()); }
        auto value() const && -> const MPI_Comm && { return std::forward<const MPI_Comm>(_comm.value()); }

        int rank = 0;
        int size = 1;

        private:
        std::optional<MPI_Comm> _comm;
    };

    inline auto free(communicator & comm) -> void {
        if (comm) {
            MPI_Comm_free(& comm.value());
        }
    }

    // split a communicator into a number of new comms. Processes providing the same colour
    // will end up in the same communicator.
    inline auto split(const communicator & comm, size_t colour) -> communicator {
        if (comm) {
            MPI_Comm new_comm;
            MPI_Comm_split(comm, colour, comm.rank, &new_comm);
            return {new_comm};
        }
        return {};
    }

    template<typename T>
    auto broadcast(const communicator & comm, std::span<T> vector, size_t process) -> void {
        if (comm) {
            MPI_Bcast(vector.data(), vector.size(), MPItype<T>, process, comm);
        }
    }

    template<typename T>
    auto all_reduce(const communicator & comm, std::span<T> vector, const MPI_Op & operation = MPI_SUM) -> void {
        if (comm) {
            MPI_Allreduce(MPI_IN_PLACE, vector.data(), vector.size(), MPItype<T>, operation, comm);
        }
    }

    template<typename T>
    auto sum(const communicator & comm, std::span<T> vector) -> void {
        all_reduce(vector, comm, MPI_SUM);
    }

    template<typename T>
    auto sum(const communicator & comm, T number) -> T {
        T result {};
        if (comm) {
            MPI_Allreduce(number, & result, 1, MPItype<T>, MPI_SUM, comm);
        }
        return result;
    }

    struct Interval {
        int first {0};
        int last {0};

        auto contains(int value) const -> bool { return (first <= value) && (value <= last); }
        auto size() const -> size_t { return last - first; }
    };
    
    struct Parallelisation {
        Interval interval;
	    std::vector<int> counts;
	    std::vector<int> displacements;
    };

    // Get the interval in a total number of tasks that this process is working on.
    inline auto get_parallelisation(const communicator & communicator, size_t total_number, size_t columns) -> Parallelisation {
        Parallelisation parallel;

        // Compute for each rank the number of elements and the starting index in the array
        for (int rank = 0; rank < communicator.size; rank++) {
            int count = total_number / communicator.size;
            int remaining = total_number % communicator.size;

            int start = rank * count;
            if (rank < remaining) {
                start += rank;
                count += 1;
            } else {
                start += remaining;
            }

            // compute the number of elements and the starting index for this rank. Required for MPI_Allgatherv.
	        parallel.counts.push_back(3 * count * columns);
            parallel.displacements.push_back(3 * start * columns);

	    if (rank == communicator.rank)
		    parallel.interval = {start, start + count - 1};
	    }

        return parallel;
    }

    template<typename T>
    auto all_gather_v(const communicator & comm, std::span<T> vector, const Parallelisation & parallel) -> void {
        if (comm) {
            MPI_Allgatherv(MPI_IN_PLACE, 1, MPItype<T>, vector.data(), parallel.counts.data(), parallel.displacements.data(), MPItype<T>, comm);
        }
    }
}
