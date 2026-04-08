from forcefield_ml.settings import Settings
from forcefield_ml.forcefields import CsldModel

settings = Settings(forcefield={"forcefield_file": "ml_forcefield_csld.dat"})

# Customize the settings for the forcefield.
settings_update = {
    'description': 'BAs conventional 3x3x3', 
    'prim': 'POSCAR',
    'max_order': 4,
    'cluster_diameter': "5.5 3.5 3.5"
}
settings.training.csld_settings.update(settings_update)

model = CsldModel(settings)

model.initialize_supercell(settings.supercell.structure_file)
model.write(settings.forcefield.forcefield_file)
