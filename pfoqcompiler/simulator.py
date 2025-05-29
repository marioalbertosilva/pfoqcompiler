import numpy as np

# Import Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate, XGate
from qiskit.circuit import ControlledGate
from pfoqcompiler.compiler import PfoqCompiler


compiler = PfoqCompiler(filename="examples/search.pfoq",
                        nb_qubits=[6,1],
                        optimize_flag=True,
                        barriers=False,
                        old_optimize=False)
compiler.parse()
compiler.compile()
circ = compiler.compiled_circuit
print(circ)


# Select the AerSimulator from the Aer provider# circ = QuantumCircuit(2)
# circ.x(0)
# circ.cx(0, 1)
# print(circ)
simulator = AerSimulator(method="statevector")

# circuit.append(ControlledGate("cx",
#                                     1 + len(cs),
#                                     [],
#                                     num_ctrl_qubits=len(cs),
#                                     ctrl_state="".join(str(i) for _, i in reversed(sorted(cs.items()))),
#                                     base_gate=XGate()),[2, 1, 0])



#circ = circuit
# initial_state = QuantumCircuit(circ.num_qubits)
# initial_state.h(0)
# initial_state.cx(0,1)
# # circ = initial_state.compose(circ)
# print(circ)
def indexket(string):
    return int(string, 2)

# for instruction in circ:
#     print(instruction.operation.definition)
# print(circ[1])
# print(circ[1].operation.definition)
#initial_state = np.array([0 for i in range(int(2**circ.num_qubits()))])

initial_state = np.zeros(int(2**circ.num_qubits))
initial_state[indexket("0001110".ljust(circ.num_qubits,"0"))] = 1
#initial_state[indexket("0"*circ.num_ancillas+"1"*(circ.num_qubits-circ.num_ancillas))] = 1
state = Statevector(initial_state)

# for instruction in circ:
#    print(instruction)

print("INPUT:")
print(state.to_dict())

print("\nOUTPUT:")
out = state.evolve(circ,qargs=["q","r","|0\rangle"])

print(out.equiv(state)) #checks if two statevectors are equal up to a global phase
print({state:amplitude for state,amplitude in out.to_dict().items() if np.absolute(amplitude)>0.0001})

# instruction = circ[0]

# circ = QuantumCircuit(10)
# print(instruction.operation)
# print(instruction.operation.definition[2].operation.definition)
# circ.append(instruction.operation, [0,1,2])
# print(circ)
# print(state.evolve(circ).to_dict())



# for instruction in circ: print(instruction)

#print(circ.instructions)

# Define a snapshot that shows the current state vector
# circ.save_statevector(label='my_sv')
# # tcirc = transpile(circ, simulator)
# # result = simulator.run(tcirc).result()



# # Execute and get saved data
# result = simulator.run(circ).result()
# data = result.data(0)
# print(data["my_sv"].to_dict())
# #print(list(data["my_sv"].to_dict().keys())[0]=="0010000")
# print(simulator.configuration())
# #print the result data
# print(data)