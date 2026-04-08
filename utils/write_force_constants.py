from forcefield_ml.settings import Settings
from forcefield_ml.forcefields import CsldModel
 
import numpy as np
 
settings = Settings(forcefield={"structure_file": "POSCAR.vasp", "forcefield_file": "ml_forcefield_csld.dat"})
settings_update = {
    'description': 'BAs 3x3x3 conventional 300K',
    'prim': 'POSCAR_primitive.vasp',
    'max_order': 4,
    'cluster_diameter': "7.1 4.0 4.0"
}

settings.training.csld_settings.update(settings_update)
 
model = CsldModel(settings)

parameters = np.loadtxt("ml_parameters.dat")
solution = np.zeros(model.n_parameters)

number = len(solution) - len(parameters)
solution[number:] = parameters

model.save_force_constants(solution, supercell_file="POSCAR.vasp", order=4)
