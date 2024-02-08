import numpy as np
import unisim
import matplotlib.pyplot as plt

from asrm import ASRM, ANFIS
from constraints import DistanceConstraint, GridBoundsConstraint

# Load UNISIM dataset
dataset = unisim.load_unisim('UNISIM-I-D')
grid = dataset.grid()
properties = dataset.properties()

# Define constraints
constraints = [
  DistanceConstraint(existing_wells=dataset.well_locations),
  GridBoundsConstraint(grid)
]

# Initialize ASRM workflow
opt = ASRM(
  objective=dataset.simulate, # simulator interface
  grid=grid,
  properties=properties,
  constraints=constraints
)

# Perform optimization
solution = opt.run()
location, value = solution.location, solution.value

print(f'Optimal location: {location}')
print(f'NPV: {value:.2f} MM$')

# Visualize solution
dataset.render_well(location)
plt.title('Optimal Well Placement')
plt.show()