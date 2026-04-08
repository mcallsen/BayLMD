#include <string>
#include <vector>

#include "builder.h"
#include "forcefields.h"
#include "reader.h"

auto main(int argc, char* argv[]) -> int {
    // Parse the command line arguments if any and read the settings from a file.
    std::string file_name = "settings.dat";
    if (argc > 1)
        file_name = argv[1];
    auto settings = reader::read_input(file_name);


    //std::vector<double> cutoffs = {8.35, 6.25, 6.25, 3.5, 3.5};
    //settings.cutoffs = {5.7, 3.53, 3.53, 3.53, 3.53}; // SnSe
    //settings.cutoffs = {5.5, 3.41, 3.41, 3.41, 3.41}; // BAs 2nd NN
    settings.cutoffs = {5.5, 4.5, 4.5, 3.5}; // BAs 4th NN
    //settings.cutoffs = {7.5, 5.0, 5.0}; // Si 4th NN

    // Create the cluster space.
    auto cluster_space = Builder::build_cluster_space(settings);

    // Convert the cluster space into a forcefield and write it to a file.
    auto force_model = Forcefields::create_forcefield(cluster_space, settings);
    force_model.write(settings.forcefield_file);

    return 0;
}
