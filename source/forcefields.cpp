#include "forcefields.h"

#include <chrono>
#include <fstream>
#include <string>

#include "mpi.h"
#include "reader.h"

namespace Forcefields {
    Orbit::Orbit(const Orbits::Orbit & orbit, const structures::Structure & primitive, const structures::StructureTransformer & transformer, const Groups::TranslationGroup & translation_group) : Orbit(orbit.cluster.size(), orbit.row_offset, orbit.cluster.prefactor) {
        for (const auto & fiber: orbit) {
            size_t rotation_index = fiber.inverse_rotation().index;

            // Find the defining cluster in the supercell.
            auto positions = primitive.ijkls_to_fractional(fiber.cluster.sites);
            auto indices = transformer.get_indices(positions);

            // Add all translated copies to the list of clusters.
            for (const auto & translation: translation_group) {
                clusters.push_back(Cluster { translation.transform(indices), rotation_index });
            }
        }
    }

    Forcefield::Forcefield(const std::string & file_name, bool parallel, size_t group): my_group(group) {
        if (parallel) {
            // Get the communicator for this processes group. All processes which provide
            // the same value for group passed from Python will be in the same communicator. 
            _communicator = mpi::split(mpi::communicator::world(), my_group);
        }

        // Read the ForceModel from a file.
        reader::Reader input(file_name);
        reader::Line line;

        // First two lines are the description of the force-field and the cutoffs, which will be ignored.
        input.skip_line();
        input.skip_line();

        n_components = reader::read_tag<size_t>(input, "components");
        n_tensors = reader::read_tag<size_t>(input, "parameters_total"); 
        n_parameters = reader::read_tag<size_t>(input, "parameters_unique"); 

        _is_zero_at = std::vector<bool> (n_tensors, false);

        // MPI: Find the interval of atoms that belongs to this process.
        _parallel_phi = mpi::get_parallelisation(_communicator, n_components / 3, n_parameters); 
        _parallel_forces = mpi::get_parallelisation(_communicator, n_components / 3, 1);

	    my_start = 3 * _parallel_phi.interval.first;
        my_end = 3 * (_parallel_phi.interval.last + 1);
	    my_rows = my_end - my_start;

        // Print some information.
        if (_communicator.rank == 0) {
            std::cout << "Group: " << my_group
                    << ", MPI processes: " << _communicator.size 
                    << ", Atoms per MPI process: " << _parallel_phi.interval.size() 
                    << std::endl;
        }

        //std::cout << "MPI proc: " << _communicator.rank << ", Interval: " << _parallel_phi.interval.first << " " << _parallel_phi.interval.last << std::endl;
        size_t n_orbits = reader::read_tag<size_t>(input, "orbits");

        // temp variables for reading the orbits.
        std::vector<size_t> numbers;
        for (size_t i = 0; i < n_orbits; i++) {
            input.skip_line();  // Name of the orbit.

            auto order = reader::read_tag<size_t>(input, "order");
            auto tensor_index = reader::read_tag<size_t>(input, "tensor_index");
            auto prefactor = reader::read_tag<size_t>(input, "prefactor");

            input.skip_line();  // Radius.

            Orbit orbit(order, tensor_index, prefactor);

            size_t n_clusters = reader::read_tag<size_t>(input, "clusters");
            for (size_t index = 0; index < n_clusters; index++) {
                numbers = reader::read_vector<size_t>(input);
                Cluster cluster { std::vector<size_t>(numbers.begin() + 1, numbers.end()), numbers[0] };

		        // Only add the clusters that contain at least one index belonging to this process. The others do not contribute
                // to the forces computed by this process.
                if (std::any_of(cluster.indices.begin(), cluster.indices.end(), [&](size_t i) { return _parallel_phi.interval.contains(i); } )) {
		    
		        // store the indices belonging to this process for convenience later.	
		        //for (size_t q = 0; q < cluster.indices.size(); q++) {
		        //    if (_parallel_phi.interval.contains(cluster.indices[q])) { 
	            //        cluster.my_indices.push_back(q);
                //    }
	            //}

                    orbit.clusters.push_back(cluster);
                }
            }
            //std::cout << "id: " << _communicator.rank << ", orbit: " << i << ", clusters: " << orbit.clusters.size() << "/" << n_clusters << std::endl;
            orbits.push_back(orbit);
        }

        reader::read_tag<size_t>(input, "Rotations");

        rotations = reader::read_vector<int>(input);
        n_symmetries = rotations.size() / 9;
        
        numbers = reader::read_vector_tag<size_t>(input, "Tensors");
        tensors = sparse::Matrix<double>(numbers[0], numbers[1]);

        // Read the C Matrix.
        for (size_t i = 0; i < tensors.rows; i++) {
            size_t n_entries = reader::read_vector<size_t>(input)[1];
            if (n_entries == 0) {
                // This row is empty so we do not have to compute the corresponding columns of A.
                _is_zero_at[i] = true;
            }
            for (size_t j = 0; j < n_entries; j++) {
                auto expression = reader::read_expression<size_t, double>(input);
                tensors.insert_quick(i, expression.first, expression.second);
            }
        }

        _is_zero_at_cache = _is_zero_at;

        // Transpose the C matrix for efficiency resons when computing AC.
        tensors = sparse::transpose(tensors);

        // allocate working arrays.
        forces = std::vector<double> (n_components, 0.0);
        phi_matrix = std::vector<double> (n_components * n_parameters, 0.0);
        a_matrix = std::vector<double> (my_rows * n_tensors, 0.0);
        rotated_displacements = std::vector<double> (n_components * n_symmetries, 0.0);

        input.close();
    }

    auto Forcefield::get_forces(const std::vector<double> & displacements) -> std::vector<double> {

	    std::fill(forces.begin(), forces.end(), 0.0);

	    //auto t1 = std::chrono::high_resolution_clock::now();

        calculate_phi(displacements);

        //auto t2 = std::chrono::high_resolution_clock::now();

        // only rows in _interval will be non zero.
        for (size_t i = my_start; i < my_end; i++) {
            for (size_t j = 0; j < n_parameters; j++) {
                forces[i] += phi_matrix[i * n_parameters + j] * parameters[j];
            }     
        }

	    //auto t3 = std::chrono::high_resolution_clock::now();

        mpi::all_reduce(forces, _communicator);

	    //auto t4 = std::chrono::high_resolution_clock::now();

        //if (_communicator.rank == 0) {
        //    std::cout << "    Get Phi          " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << std::endl
	    //              << "    Multiplication   " << std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count() << std::endl
	    //	            << "    allreduce        " << std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count() << std::endl;
        //}


        return forces;
    }

    auto Forcefield::get_fit_matrix(const std::vector<double> & displacements) -> std::vector<double> {
        calculate_phi(displacements);
        // Collect the results from all the MPI processes.
        mpi::all_reduce(phi_matrix, _communicator);
        return phi_matrix;
    }

    auto Forcefield::write(const std::string & file_name) const -> void {
        std::ofstream output(file_name);

        output << n_components << " " << n_tensors << " " << n_parameters << std::endl;
        output << "Orbits: " << orbits.size() << std::endl;

        for (const auto & orbit: orbits) {
            output << orbit.clusters.size() << " " << orbit.order << " " << orbit.dimension << " " << orbit.tensor_index << std::endl;
            output << orbit.prefactor << std::endl;
            for (const auto & cluster: orbit.clusters) {
                output << cluster.rotation_index << " ";
                String::operator<<(output, cluster.indices) << std::endl;
            }
        }

        output << "Rotations: " << rotations.size() << std::endl;
        String::operator<<(output, rotations) << std::endl;

        output << "Tensors: " << tensors.rows << " " << tensors.columns << std::endl;
        for (size_t i = 0; i < tensors.size(); i++) {
            auto & row = tensors[i];
            output << i << "     " << row.size() << std::endl;
            for (size_t j = 0; j < row.size(); j++) {
                output << "    " << row.indices[j] << ": ";
                output << std::setprecision(12) << row.values[j] << std::endl;
            }
        }
        
        output.close();
    }

    auto Forcefield::calculate_phi(const std::vector<double> & displacements) -> void {

	    //auto t1 = std::chrono::high_resolution_clock::now();

        calculate_a_matrix(displacements);

        //auto t2 = std::chrono::high_resolution_clock::now();

        //size_t n_size = displacements.size();
        //std::vector<double> result(n_size * n_parameters, 0.0);

        std::fill(phi_matrix.begin(), phi_matrix.end(), 0.0);

        //auto t3 = std::chrono::high_resolution_clock::now();

        // Evaluate the Matrix product explicitely. Because tensors is already transposed, this is the scalar product of
        // row vectors of 'matrix' and 'tensors'. Only rows in _interval will be non zero.
        for (size_t i = 0; i < my_rows; i++) {
	    size_t row_index = (my_start + i) * n_parameters;
            for (size_t j = 0; j < n_parameters; j++) {
                auto & tensor = tensors[j];
                for (size_t k = 0; k < tensor.size(); k++) {
                    phi_matrix[row_index + j] += a_matrix[i * n_tensors + tensor.indices[k]] * tensor.values[k];
                }
            }
        }

	    //auto t4 = std::chrono::high_resolution_clock::now();

        //if (_communicator.rank == 0) {
        //    std::cout << "        calculate_a_matrix  " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << std::endl
	    //	            << "        std::fill           " << std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count() << std::endl
        //              << "        Matrix product      " << std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count() << std::endl;
        //}
    }

    auto Forcefield::calculate_a_matrix(const std::vector<double> & displacements) -> void {
        // This subroutine implements formula (18) from the CSLD paper:
        //
        //     A'(a, \alpha I) = - 1 / \alpha! \sum_{s\alpha in S\alpha } \Gamma_IJ (s) \partial_a u_I^{s\alpha}
        //
        // In this particular formulation the displacements are rotated instead of the tensors, which 
        // has some advantages in terms of performance. Hiphive rotates the tensors instead. MPI parallelisation
        // works over the rows of A, which are the atom indices assigned to each process. For optimisation we
        // do not have to compute columns in A for which the corresponding row in C is 0, because during the 
        // matrix product with C those columns will always hit a 0. 
        size_t n_size = displacements.size();

        //std::vector<double> matrix(n_size * n_tensors, 0.0);

        //auto t1 = std::chrono::high_resolution_clock::now();

        std::fill(a_matrix.begin(), a_matrix.end(), 0.0);

        //auto t2 = std::chrono::high_resolution_clock::now();

        // Precalculate the rotated displacements. We are using the inverse rotation matrices here
        // because the summation in formula (17) in the CSLD paper is over the first index.
        rotate_displacements(displacements);

        //auto t3 = std::chrono::high_resolution_clock::now();

        for (const auto & orbit: orbits) {
            // compute -1 / \alpha!. Required only once per orbit.
            double prefactor = -1.0 / extensions::as<double>(orbit.prefactor); 
            for (const auto & cluster: orbit.clusters) {
                // the symmetry operation s for s\alpha.
                size_t symmetry_index = cluster.rotation_index;

                // Keep a reference for the atom indices in this cluster.
                auto & indices = cluster.indices;
                //auto & my_indices = cluster.my_indices;

                // Loop over all possible indices in a Tensor.
                for (size_t index = 0; index < orbit.dimension; index++) {
                    // The column in A corresponding to this tensor component. 
                    size_t column = orbit.tensor_index + index;

                    // Check whether the row in C corresponding to this column is 0.
                    if (_is_zero_at[column]) 
                        continue;

                    // convert the index in the flattened version of the tensor to the multi-index I.
		            auto multi_index = Tensor::index_to_subscript_asc(index, orbit.order);

                    // We are taking the derivative w.r.t. to the qth index.
                    //for (auto q: my_indices) {
                    for (size_t q = 0; q < indices.size(); q++) {

			            if (!_parallel_phi.interval.contains(indices[q])) {
                            // The derivatives w.r.t. this index do not contribute to
                            // the rows of A belonging to this process.
                            continue;
                        }

                        // Compute the row index in A which corresponds to the x component of the atom indices[q].
                        size_t row = indices[q] * 3 - my_start;
                        double product = prefactor;

                        // Compute the product of the rotated displacements skipping q.
                        for (size_t p = 0; p < indices.size(); p++) {
                            if (p == q) continue;
                            product *= rotated_displacements[symmetry_index * n_size + 3 * indices[p] + multi_index[p]];
                        }

                        // add the displacement product to the correctly (!) rotated tensor component. This rotation is the one which is 
                        // left over after the derivative and rotates the tensor directly. NOTE: you can probably safe 2/3 of the 
                        // computation time by realising that only one of the three columns of rotation is non zero.
                        for (size_t j = 0; j < 3; j++) {
                            a_matrix[(row + j) * n_tensors + column] += product * rotations[9 * symmetry_index + multi_index[q] * 3 + j];
                        }  
                    }
                }
            }
        }

        //auto t4 = std::chrono::high_resolution_clock::now();

        //if (_communicator.rank == 0) {
        //    std::cout << "    std::fill       " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << std::endl
	    //	            << "    Rotation        " << std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count() << std::endl
        //              << "    Calculation     " << std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count() << std::endl;
        //}
    }

    auto Forcefield::set_parameter_mask(const std::vector<size_t> & indices) -> void {
        // start by setting all columns to 0.
        _is_zero_at = std::vector<bool> (n_tensors, true);

        // Only allow the columns in indices. 
        for (auto const & index: indices) {
            // Skip those columns for which the rows in the C matrix are 0 anyway.
            if (_is_zero_at_cache[index] == true) {
                continue;
            }
            _is_zero_at[index] = false;
        }
    }

    auto Forcefield::rotate_displacements(const std::vector<double> & displacements) -> void {
        size_t n_size = displacements.size();
        size_t n_atoms = n_size / 3;

	    std::fill(rotated_displacements.begin(), rotated_displacements.end(), 0.0);

        for (size_t i_symmetry = 0; i_symmetry < n_symmetries; i_symmetry++) {
            for (size_t i_atom = 0; i_atom < n_atoms; i_atom++) {
                for (size_t i = 0; i < 3; i++) {
                    for (size_t j = 0; j < 3; j++) {
                        rotated_displacements[i_symmetry * n_size + i_atom * 3 + i] += rotations[9 * i_symmetry + 3 * i + j] * displacements[3 * i_atom + j];
                    }
                }
            }
        }
    }
}
