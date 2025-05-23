import numpy as np

# Import Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# Select the AerSimulator from the Aer provider
simulator = AerSimulator(method='matrix_product_state')

circ = QuantumCircuit(2)
circ.cx(0, 1)
print(circ)

# Define a snapshot that shows the current state vector
circ.save_statevector(label='my_sv')
print(circ)
tcirc = transpile(circ, simulator)
result = simulator.run(tcirc).result()


circ.save_matrix_product_state(label='my_mps')

# Execute and get saved data
result = simulator.run(tcirc).result()
data = result.data(0)

#print the result data
print(data['counts'])