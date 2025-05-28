
"""
Main class for PFOQ programs compilation
"""


import lark
import argparse
import matplotlib.pyplot as plt
import networkx as nx
from typing_extensions import Optional, Union
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import HGate, XGate, CCXGate, SwapGate, RYGate, CPhaseGate, PhaseGate, Barrier
from qiskit.circuit import Gate, ControlledGate, CircuitInstruction
from qiskit.visualization import circuit_drawer
#from qiskit_addon_cutting.utils.simulation import simulate_statevector_outcomes
import qiskit.qasm3
from math import ceil

import warnings
from pfoqcompiler.errors import WidthError, AncillaIndexError, NotCompiledError
from pfoqcompiler.parser import PfoqParser




DEBUG = False


class PfoqCompiler:
    """Compiler class for PFOQ programs.

    Instantiate a compiler to compile the provided PFOQ program.

    Parameters
    ----------
    program: str, optional
        PFOQ program to compile. Mutually exclusive with `filename`

    filename: str, optional
        Path linking to the file containing the PFOQ program to compile.

    nb_qubits: int
        Number of data qubits to consider, aka the size of the input.

    nb_ancillas: int
        Number of ancilla qubits available to work with.

    Examples
    --------
    >>> prg = "decl f(q){q[0]*=H;call f(q-[0]);}:: define q; :: call f(q);"
    >>> compiler = PfoqCompiler(program=prg)
    >>> compiler.compile()

    """

    def __init__(self,
                 program: Optional[str] = None,
                 filename: Optional[str] = None,
                 nb_qubits: list[int] = [8],
                 nb_ancillas: int = 1,
                 optimize_flag: bool = True,
                 old_optimize: bool = False,
                 barriers: bool = False):
        
        if len((tmp_set := {program, filename} - {None})) == 0:
            raise ValueError("Either program or filename argument must be provided.")

        if len(tmp_set) == 2:
            raise ValueError("At most one of program and filename must be provided.")

        self._filename = filename

        if filename is not None:
            with open(filename, 'r') as f:
                program = f.read()

        self._program = program
        self._parser = PfoqParser()
        self._nb_qubits = nb_qubits
        self._nb_ancillas = nb_ancillas
        #self._qr = [QuantumRegister(self._nb_qubits, name="q")]
        self._qr = []
        self._ar = AncillaRegister(self._nb_ancillas, name="|0\\rangle")
        self._functions = {}
        self._qubit_registers = []
        self._mutually_recursive_indices = {}
        self._max_used_ancilla = -1
        self._ast = None
        self._compiled_circuit = None
        self._optimize_flag = optimize_flag
        self._old_optimize = old_optimize
        self._enforce_order = barriers

    @property
    def compiled_circuit(self):
        if self._compiled_circuit is None:
            raise NotCompiledError("Circuit has not been compiled.")
        return self._compiled_circuit
        

    def parse(self):
        try:
            self._ast = self._parser.parse(self._program)
            
            if DEBUG:
                print(self._ast)
        except Exception as exception:
            if DEBUG:
                if self._filename is None:
                    print("Parsing of program failed due to:")
                else:
                    print(f"Parsing of file {self._filename} failed due to:")
            raise exception

        if DEBUG:
            print("Program parsed successfully!")

    def compile(self, remove_idle_wires: bool = True):
        """Compile the program.

        Compile the program according to some parametrizable rules.

        Parameters
        ----------
        optimize : bool
            Whether an optimiztion routine should be used.
        remove_idle_wires : bool
            Whether to unused ancilla qubits from the output.

        Raises
        ------
        exception

        Examples
        --------
        >>> prg = "decl f(q){q[0]*=H;call f(q-[0]);}::call f(q);"
        >>> compiler = PfoqCompiler(program=prg)
        >>> compiler.compile()

        """
        while True:
            try:
                self._compiled_circuit = self._compr_prg()
                if remove_idle_wires:
                    self.remove_idle_wires()
                break
            except AncillaIndexError:
                self._nb_ancillas *= 2
                if DEBUG:
                    print(f"Insufficient ancillas, doubling ({self._nb_ancillas})", flush=True)
                self._ar = AncillaRegister(self._nb_ancillas, name="|0\\rangle")
                self._max_used_ancilla = -1

            except Exception as exception:
                if DEBUG:
                    if self._filename is None:
                        print("Compilation of program failed due to:")
                    else:
                        print(f"Compilation of file {self._filename} failed due to:")
                raise exception

        if DEBUG:
            print(f"# ancillas = {self._nb_ancillas}.", flush=True)

    def save(self, filename: str):
        if self._compiled_circuit is None:
            raise NotCompiledError("The circuit hasn't been successfully compiled yet.")
        with open(f"{filename}.qasm", "w") as f:
            qiskit.qasm3.dump(self._compiled_circuit, f)

    def display(self):
        if self._compiled_circuit is None:
            raise NotCompiledError("The circuit hasn't been successfully compiled yet.")
                    
        try:
            
            if self._enforce_order:

                N = sum(self._nb_qubits) + self._nb_ancillas

                for i in range(len(self._compiled_circuit.data)):
                    self._compiled_circuit.data.insert(
                            2*i+1, CircuitInstruction(Barrier(N), range(N)))
                
            circuit_drawer(self._compiled_circuit, output="mpl", style="bw",
                            fold=-1, plot_barriers=False)
            plt.show()
        except Exception as exception:
            if DEBUG:
                print("Program has been successfully compiled, but could not be displayed due to:")
            raise exception

    def _compr_prg(self) -> QuantumCircuit:
        if DEBUG:
            print("let's go!")

        if len(self._ast.children) == 0:
            raise RuntimeWarning("Empty program")

        for child in self._ast.children[:-2]:
            if DEBUG:
                assert child.data == "decl"
            self._functions[child.children[0].value] = child

        graph = self._compute_call_graph()
        #nx.draw(graph, with_labels =True)
        #plt.show()
        components = list(nx.strongly_connected_components(graph))
        for index, value in enumerate(components):
            for name in value:
                self._mutually_recursive_indices[name] = index

        #this step stores a width parameter in the tree that controls the flow in optimize
        for function in self._functions:
           width = self._width_function(function)
           if self._optimize_flag and width > 1:
               print(f"Procedure {function} has width {width}. Turning off optimization.")
               self._optimize_flag = False

        # print(ast)

        # print(components)
        # nx.draw(graph, with_labels = True)
        # plt.show()


        program_statement = self._ast.children[-1]

        if DEBUG:
            assert (program_statement.data == "lstatement")

        register_definition = self._ast.children[-2]

        if DEBUG:
            assert (register_definition.data == "def")

        L = {reg.value : list(range(nb)) for reg, nb in zip(register_definition.children, self._nb_qubits)}
        #print(L)
        #L=list(range(self._nb_qubits))
        self._qubit_registers = register_definition.children


        if DEBUG:
            assert (len(self._nb_qubits) == len(self._qubit_registers) )


        self._qr = [QuantumRegister(nb, name=f"{reg}") for reg, nb in zip(register_definition.children, self._nb_qubits)]


        qc = QuantumCircuit(*self._qr, self._ar)

        qc = qc.compose(self._compr_lstatement(ast=program_statement, L=L, cs={}, variables={}))
        
        self._compiled_circuit = qc
        #is this return necessary?
        return self._compiled_circuit
    
    def _create_control_state(self, cs: dict):
        """" Create control state from dictionary cs.

        Parameters
        ----------
        cs : dict
            Dictionary of qubit addresses and respective state

        Examples
        --------
        >>> cs = {0:1, 2:1, 5:1}
        >>> compiler = PfoqCompiler(program=prg)
        >>> compiler._create_control_state(cs)    
        """
        return "".join(str(i) for _, i in reversed(sorted(cs.items())))

    def _compute_call_graph(self):
        initial_graph = nx.DiGraph()
        for f, g in self._functions.items():
            initial_graph.add_node(f)
            _create_call_graph(initial_graph, f, g)
        return initial_graph

    def _compr_lstatement(self, ast, L, cs, variables):
        qc = QuantumCircuit(*self._qr, self._ar)
        for child in ast.children:
            qc = qc.compose(self._compr_statement(child, L, cs, variables))
        return qc

    def _compr_statement(self, ast, L, cs, variables):
        if ast.data == "gate_application":
            return self._compr_gate_application(ast, L, cs, variables)
        elif ast.data == "cnot_gate":
            return self._compr_cnot_gate(ast, L, cs, variables)
        elif ast.data == "swap_gate":
            return self._compr_swap_gate(ast, L, cs, variables)
        elif ast.data == "toffoli_gate":
            return self._compr_toffoli_gate(ast, L, cs, variables)
        elif ast.data == "if_statement":
            if self._compr_boolean_expression(ast.children[0], L, cs, variables):
                return self._compr_lstatement(ast.children[1], L, cs, variables)
            elif len(ast.children) == 3:
                return self._compr_lstatement(ast.children[2], L, cs, variables)
            else:
                return QuantumCircuit(*self._qr, self._ar)
        elif ast.data == "qcase_statement":
            q = self._compr_qubit_expression(ast.children[0], L, cs, variables)
            if q in cs:
                raise IndexError(f"Already controlling on the state of qubit {q}.")
            circuit = QuantumCircuit(*self._qr, self._ar)
            cs[q] = 0
            circuit = circuit.compose(self._compr_lstatement(ast.children[1], L,
                                                             cs, variables))
            cs[q] = 1
            circuit = circuit.compose(self._compr_lstatement(ast.children[2], L,
                                                             cs, variables))
            del cs[q]
            return circuit

        elif ast.data == "procedure_call":
            return self._compr_procedure_call(ast, L, cs, variables)

        elif ast.data == "skip_statement":
            return QuantumCircuit(*self._qr, self._ar)

        else:
            raise NotImplementedError(f"Statement {ast.data} not yet handled.")

    def _compr_gate_application(self, ast, L, cs, variables):
        qubit = self._compr_qubit_expression(ast.children[0], L, cs, variables)
        if qubit in cs:
            raise IndexError(f"Cannot apply gate on qubit {qubit} that is controlled on its state.")
        
        if ast.children[1].data == "not_gate":
            qc = QuantumCircuit(*self._qr, self._ar)
            qc.mcx(list(sorted(cs)),qubit,ctrl_state = self._create_control_state(cs))
            return qc
        
        gate = self._compr_gate_expression(ast.children[1], L, cs, variables)
        qc = QuantumCircuit(*self._qr, self._ar)
        qc.append(gate, list(sorted(cs)) + [qubit])
        return qc

    def _compr_cnot_gate(self, ast, L, cs, variables):
        qubit1 = self._compr_qubit_expression(ast.children[0], L, cs, variables)
        if qubit1 in cs:
            raise IndexError(f"Multiple controls on same qubit {qubit1}.")
        cs[qubit1] = 1
        # gate = ControlledGate("CX",
        #                       1 + len(cs),
        #                       [],
        #                       num_ctrl_qubits=len(cs),
        #                       ctrl_state="".join(
        #                           str(i) for _, i in reversed(sorted(cs.items()))),
        #                       base_gate=XGate())
        qubit2 = self._compr_qubit_expression(ast.children[1], L, cs, variables)
        if qubit2 in cs:
            raise IndexError(
                f"Cannot apply gate on qubit {qubit2} that is controlled on its state.")

        qc = QuantumCircuit(*self._qr, self._ar)
        if cs:
            qc.mcx(list(sorted(cs)),qubit2, ctrl_state = self._create_control_state(cs))
            #qc.append(gate, list(sorted(cs)) + [qubit2])
            del cs[qubit1]
        return qc

    def _compr_swap_gate(self, ast, L, cs, variables):
        qubit1 = self._compr_qubit_expression(ast.children[0], L, cs, variables)
        if qubit1 in cs:
            raise IndexError(f"Cannot swap control qubit {qubit1}.")
        
        qubit2 = self._compr_qubit_expression(ast.children[1], L, cs, variables)
        
        if qubit2 in cs:
            raise IndexError(f"Cannot swap control qubit {qubit2}.")

        qc = QuantumCircuit(*self._qr, self._ar)

        if cs:
            gate = ControlledGate("CSWAP",
                                  2 + len(cs),
                                  [],
                                  num_ctrl_qubits=len(cs),
                                  ctrl_state="".join(
                                      str(i) for _, i in reversed(sorted(cs.items()))),
                                  base_gate=SwapGate())
        else:
            gate = SwapGate()

        qc.append(gate, list(sorted(cs)) + [qubit1, qubit2])
        return qc
    
    def _compr_toffoli_gate(self, ast, L, cs, variables):

        Q = [self._compr_qubit_expression(ast.children[i], L, cs, variables) for i in range(3)]

        for i in range(3):
            if Q[i] in cs:
                raise IndexError(f"Cannot apply Toffoli on control qubit {Q[i]}.")

        qc = QuantumCircuit(*self._qr, self._ar)

        if cs:
            gate = ControlledGate("cx",
                              3 + len(cs),
                              [],
                              num_ctrl_qubits=2+len(cs),
                              ctrl_state="11"+"".join(
                                  str(i) for _, i in reversed(sorted(cs.items()))),
                              base_gate=XGate())
            
        else:
            gate = ControlledGate("cx",
                              3,
                              [],
                              num_ctrl_qubits=2,
                              ctrl_state="11",
                              base_gate=XGate())

        qc.append(gate, list(sorted(cs)) + Q)

        return qc

    def _compr_procedure_call(self, ast, L, cs, variables):
        proc_identifier = ast.children[0].value
        if proc_identifier not in self._functions:
            raise NameError(f"Called function {proc_identifier} was not declared.")


        #update L: assumes very orderly inputs
        new_L = L.copy()

        for index in range(-len(L),0):
            new_L[self._qubit_registers[index]] = self._compr_register_expression(ast.children[index], L, cs, variables)

        if [] in new_L.values():
            return QuantumCircuit(*self._qr, self._ar)


        function = self._functions[proc_identifier]
        
        function_parameter = function.children[1] if len(function.children) > 2 + len(L) else None

        int_parameter = None

        if len(ast.children) > 1 + len(L):
            int_parameter = self._compr_int_expression(ast.children[1], L, cs, variables)

        if (function_parameter is None) ^ (int_parameter is None):
            raise (ValueError(
                f"Incorrect number of parameters passed to function {proc_identifier}"))

        if int_parameter is not None:
            old_value = variables[function_parameter] if function_parameter in variables else None
            variables[function_parameter] = int_parameter

        if DEBUG:
            print(f"in compr: calling {proc_identifier} on input {new_L}")

        if new_L:
            if self._optimize_flag and self._width_function(proc_identifier) == 1:
                l_CST = [(cs, self._functions[proc_identifier].children[-1], new_L, variables)]
                return self._optimize(l_CST)

            circ = self._compr_lstatement(
                self._functions[proc_identifier].children[-1], new_L, cs, variables)
            if int_parameter is not None:
                if old_value is None:
                    del variables[function_parameter]
                else:
                    variables[function_parameter] = old_value
            return circ
        return QuantumCircuit(*self._qr, self._ar)

    # EXPRESSIONS (Registers, Booleans, Integers, Gates)
    def _compr_qubit_expression(self, ast, L, cs, variables):

        #determine qubit_register_name
        
        qubit_register_name = ast.children[0]


        #determine qubit list

        if ast.data == "qubit_expression_identifier":
            qubit_list = self._compr_register_identifier(ast, L, cs, variables)

        elif ast.data == "qubit_expression_parenthesed":
            qubit_list = self._compr_parenthesed_register_expression(
                ast.children[0], L, cs, variables)
        else:
            raise NotImplementedError(
                f"Qubit expression {ast.data} not yet handled.")
        

        integer_value = self._compr_int_expression(ast.children[1], L, cs, variables)


        try:
            qubit = qubit_list[integer_value] #address within register

            for i in range(len(self._qubit_registers)): #find global address
                if self._qubit_registers[i] != qubit_register_name:
                    qubit += self._nb_qubits[i]
                else:
                    break

        except IndexError:
            raise IndexError(
                f"Qubit index {integer_value} cannot reference a qubit in a register of size {len(qubit_list)}.")

        return qubit

    def _compr_register_identifier(self, ast, L, cs, variables):
        return L[ast.children[0].value]

    def _compr_parenthesed_register_expression(self, ast, L, cs, variables):
        return self._compr_register_expression(ast.children[0], L, cs, variables)

    def _compr_parenthesed_register_expression_first_half(self, ast, L, cs, variables):
        qubit_list = self._compr_parenthesed_register_expression(ast.children[0], L, cs, variables)
        if len(qubit_list) <= 1:
            return []
        else:
            m = ceil(len(qubit_list)/2)
            return qubit_list[:m]
        
    def _compr_parenthesed_register_expression_second_half(self, ast, L, cs, variables):
        qubit_list = self._compr_parenthesed_register_expression(ast.children[0], L, cs, variables)
        if len(qubit_list) <= 1:
            return []
        else:
            m = ceil(len(qubit_list)/2)
            return qubit_list[m:]
        
        
    def _compr_register_identifier_first_half(self,ast,L,cs,variables):
        qubit_list = L[ast.children[0].value]
        if len(qubit_list) <= 1:
            return []
        else:
            m = ceil(len(qubit_list)/2)
            return qubit_list[:m]



    def _compr_register_identifier_second_half(self,ast,L,cs,variables):
        qubit_list = L[ast.children[0].value]
        if len(qubit_list) <= 1:
            return []
        else:
            m = ceil(len(qubit_list)/2)
            return qubit_list[m:]

    
    


    def _compr_register_expression_minus(self, ast, L, cs, variables):
        qubit_list = self._compr_register_expression(ast.children[0], L, cs, variables)
        indices = [self._compr_int_expression(ast.children[i], L, cs, variables) for i in range(1,len(ast.children))]

        N = len(qubit_list)
        if min(indices) < -N or max(indices) > N - 1:
            raise IndexError("list index out of range")        
        
        nonnegative_indices = [ i % N for i in indices]
        #indices written as nonnegative values in decreasing order

        return [qubit for index,qubit in enumerate(qubit_list) if index not in nonnegative_indices]
        
        i = self._compr_int_expression(ast.children[1], L, cs, variables)
        if i < 0:
            i = len(qubit_list) + i
        if i < 0 or i >= len(qubit_list):
            return []
        return qubit_list[:i] + qubit_list[i + 1:]

    def _compr_register_expression(self, ast, L, cs, variables):
        if ast.data == "register_expression_identifier":
            return self._compr_register_identifier(ast, L, cs, variables)
        elif ast.data == "parenthesed_register_expression":
            return self._compr_parenthesed_register_expression(ast, L, cs, variables)
        elif ast.data == "register_expression_minus":
            return self._compr_register_expression_minus(ast, L, cs, variables)
        elif ast.data == "register_expression_parenthesed_first_half":
            return self._compr_parenthesed_register_expression_first_half(ast, L, cs, variables)
        elif ast.data == "register_expression_parenthesed_second_half":
            return self._compr_parenthesed_register_expression_second_half(ast, L, cs, variables)
        elif ast.data == "register_identifier_first_half":
            return self._compr_register_identifier_first_half(ast, L, cs, variables)
        elif ast.data == "register_identifier_second_half":
            return self._compr_register_identifier_second_half(ast, L, cs, variables)
        else:
            raise NotImplementedError(
                f"Register expression {ast.data} not yet handled:")

    def _compr_boolean_expression(self, ast, L, cs, variables):
        if ast.data == "bool_literal":
            return ast.children[0].value == "true"
        elif ast.data == "bool_greater_than":
            return self._compr_int_expression(ast.children[0], L, cs, variables) > self._compr_int_expression(ast.children[1], L, cs, variables)
        else:
            raise NotImplementedError(f"Boolean expression {ast.data} not yet handled.")

    def _compr_int_expression(self, ast, L, cs, variables):

        if ast.data == "int_expression_literal":
            return int(ast.children[0].value)
        elif ast.data == "binary_op":
            if ast.children[1].value == "+":
                return int(self._compr_int_expression(ast.children[0], L, cs, variables) + self._compr_int_expression(ast.children[2], L, cs, variables))
            elif ast.children[1].value == "-":
                return int(self._compr_int_expression(ast.children[0], L, cs, variables) - self._compr_int_expression(ast.children[2], L, cs, variables))
        elif ast.data == "size_of_register":
            return len(self._compr_register_expression(ast.children[0], L, cs, variables))
        elif ast.data == "int_expression_identifier":
            variable_name = ast.children[0].value
            if variable_name not in variables:
                raise ValueError(f"Variable {variable_name} not defined.")
            return variables[variable_name]
        elif ast.data == "parenthesed_int_expression":
            return self._compr_int_expression(ast.children[0], L, cs, variables)
        elif ast.data == "parenthesed_int_expression_half":
            return int(self._compr_int_expression(ast.children[0], L, cs, variables)/2)
        elif ast.data == "int_expression_half_size":
            return int(len(self._compr_register_expression(ast.children[0], L, cs, variables))/2)
        else:
            raise NotImplementedError(f"Integer expression {ast.data} not yet handled.")

    def _compr_gate_expression(self, ast, L, cs, variables):
        # reversed(sorted(cs.items()))): Because qiskit indexes its qubits in the wrong order
        if ast.data == "hadamard_gate":
            if cs:
                return ControlledGate("CH",
                                      1 + len(cs),
                                      [],
                                      num_ctrl_qubits=len(cs),
                                      ctrl_state="".join(
                                          str(i) for _, i in reversed(sorted(cs.items()))),
                                      base_gate=HGate())
            return HGate()
        
        elif ast.data == "rotation_gate":
            theta = self._compr_int_expression(ast.children[0],L,cs,variables)
            if cs:
                return ControlledGate("CRot",
                                      1 + len(cs),
                                      [],
                                      num_ctrl_qubits=len(cs),
                                      ctrl_state="".join(
                                          str(i) for _, i in reversed(sorted(cs.items()))),
                                      base_gate=RYGate(theta=theta, label=f"R{theta}"))
            return RYGate(theta=theta,label=f"R{theta}")
        
        elif ast.data == "phase_shift_gate":
            theta = self._compr_int_expression(ast.children[0],L,cs,variables)
            if cs:
                return CPhaseGate(theta=theta,
                                  ctrl_state="".join(
                                           str(i) for _, i in reversed(sorted(cs.items())))
                                  )
            return PhaseGate(theta=theta,label=f"Ph({theta})")
        
        elif ast.data == "toffoli_gate":

            q1 = self._compr_int_expression(ast.children[0],L,cs,variables)
            q2 = self._compr_int_expression(ast.children[1],L,cs,variables)
            q3 = self._compr_int_expression(ast.children[2],L,cs,variables)

            if cs:
                return ControlledGate("mcx",
                                      3,
                                      [],
                                      num_ctrl_qubits = 2,
                                      ctrl_state="".join(
                                          str(i) for _, i in reversed(sorted(cs.items()))),
                                      base_gate=XGate())
            return XGate()
        
        elif ast.data == "not_gate":
            if cs:
                return ControlledGate("mcx",
                                      1 + len(cs),
                                      [],
                                      num_ctrl_qubits=len(cs),
                                      ctrl_state="".join(
                                          str(i) for _, i in reversed(sorted(cs.items()))),
                                      base_gate=XGate())
            return XGate()
        
        elif ast.data == "other_gates":
            if cs:
                return ControlledGate("C" + ast.children[0].value[1:-1],
                                      1 + len(cs),
                                      [],
                                      num_ctrl_qubits=len(cs),
                                      ctrl_state="".join(
                                          str(i) for _, i in reversed(sorted(cs.items()))),
                                      base_gate=Gate(ast.children[0].value[1:-1].format(**variables), 1, []))
            return Gate(ast.children[0].value[1:-1], 1, [])
        else:
            raise ValueError(f"Unexpected gate value {ast.data}.")

    #PARTIAL ORDERING ON INPUTS
    def _input_ordering(self,x):
        return -max(len(x[2][reg]) for reg in self._qubit_registers)

    def _optimize(self, l_CST):

        Ancillas = {}
        C_L, C_R = QuantumCircuit(*self._qr, self._ar), QuantumCircuit(*self._qr, self._ar)
        l_M = []

        while l_CST:
            cs, ast, L, variables = l_CST.pop(0)

            if ast.data == "lstatement":
                before = True
                for child in ast.children:
                    if child.width == 0:
                        if before:
                            C_L = C_L.compose(self._compr_statement(child, L, cs, variables))
                        else:
                            C_R = self._compr_statement(child, L, cs, variables).compose(C_R)
                    else:
                        before = False
                        l_CST.append((cs, child, L, variables))

            elif ast.data == "if_statement":
                guard = self._compr_boolean_expression(ast.children[0], L, cs, variables)

                if guard:

                    if ast.children[1].width:
                        l_CST.append((cs, ast.children[1], L, variables))

                    elif self._old_optimize:
                        C_R = self._compr_lstatement(ast.children[1], L, cs, variables).compose(C_R)

                    else:                            
                        l_M.append((cs, ast.children[1], L, variables))

                elif len(ast.children) == 3:

                    if ast.children[2].width:
                        l_CST.append((cs, ast.children[2], L, variables))

                    elif self._old_optimize:
                        C_R = self._compr_lstatement(ast.children[2], L, cs, variables).compose(C_R)      

                    else:  
                        l_M.append((cs, ast.children[2], L, variables))

            elif ast.data == "qcase_statement":
                q = self._compr_qubit_expression(ast.children[0], L, cs, variables)
                if q in cs:
                    raise IndexError(
                        f"Already controlling on the state of qubit {q}.")

                cs_0, cs_1 = cs.copy(), cs.copy()
                cs_0[q] = 0
                cs_1[q] = 1

                if ast.children[1].width:
                    l_CST.append((cs_0, ast.children[1], L, variables))

                elif self._old_optimize:
                    C_R = self._compr_lstatement(ast.children[1], L, cs_0, variables).compose(C_R)      
                
                else:
                    l_M.append((cs_0, ast.children[1], L, variables))



                if ast.children[2].width:
                    l_CST.append((cs_1, ast.children[2], L, variables))

                elif self._old_optimize:
                    C_R = self._compr_lstatement(ast.children[2], L, cs_1, variables).compose(C_R) 

                else:           
                    l_M.append((cs_1, ast.children[2], L, variables))

            elif ast.data == "procedure_call":
                

                proc_identifier = ast.children[0].value
                if proc_identifier not in self._functions:
                    raise NameError(
                        f"Called function {proc_identifier} was not declared.")


                new_L = L.copy()


                for index in range(-len(L),0):
                    new_L[self._qubit_registers[index]] = self._compr_register_expression(ast.children[index], L, cs, variables)

                if [] in new_L.values():
                    continue
                
                # if not new_L:
                #     continue

                function = self._functions[proc_identifier]
                function_parameter = function.children[1] if len(function.children) > 2 + len(L) else None
                int_parameter = None

                if len(ast.children) > 1 + len(L):
                    int_parameter = self._compr_int_expression(
                        ast.children[1], L, cs, variables)

                if (function_parameter is None) ^ (int_parameter is None):
                    raise (ValueError(
                        f"Incorrect number of parameters passed to function {proc_identifier}"))

                if int_parameter is not None:
                    variables[function_parameter] = int_parameter

                reg_sizes = tuple([len(new_L[reg]) for reg in self._qubit_registers])
                if (proc_identifier,reg_sizes, int_parameter) in Ancillas:
                    ancilla, anchored_L = Ancillas[(proc_identifier, reg_sizes, int_parameter)]
                    # gate = ControlledGate("CX",
                    #                       1 + len(cs),
                    #                       [],
                    #                       num_ctrl_qubits=len(cs),
                    #                       ctrl_state="".join(str(i)
                    #                                          for _, i in reversed(sorted(cs.items()))),
                    #                       base_gate=XGate())


                    C_L.mcx(list(sorted(cs)),ancilla, ctrl_state = self._create_control_state(cs))
                    #C_L.append(gate, list(sorted(cs)) + [ancilla])

                    circ = QuantumCircuit(*self._qr, self._ar)
                    circ.mcx(list(sorted(cs)),ancilla, ctrl_state = self._create_control_state(cs))
                    C_R = circ.compose(C_R)


                    anchored_register, merging_register = [], []

                    for index,reg in enumerate(self._qubit_registers):
                        anchored_register += [qubit + sum(self._nb_qubits[:index]) for qubit in anchored_L[reg]]
                        merging_register += [qubit + sum(self._nb_qubits[:index]) for qubit in new_L[reg]]


                    if merging_register!= anchored_register:
                        if DEBUG:
                            print("with controlled_swaps")

                        transposition_list = _merging_transpositions(merging_register, anchored_register)
                        largest_size = max([len(i) for i in transposition_list])

                        self._max_used_ancilla += 1
                        starting_ancilla = sum(self._nb_qubits) + self._max_used_ancilla

                        if self._max_used_ancilla >= self._nb_ancillas:
                            raise AncillaIndexError("Not enough ancillas")

                        C_L.mcx(list(sorted(cs)),starting_ancilla, ctrl_state = self._create_control_state(cs))
                        #C_L.append(gate, list(sorted(cs)) + [starting_ancilla])
                        circ = QuantumCircuit(*self._qr, self._ar)

                        circ.mcx(list(sorted(cs)),starting_ancilla, ctrl_state = self._create_control_state(cs))
                        #circ.append(gate, list(sorted(cs)) + [starting_ancilla])
                        C_R = circ.compose(C_R)

                        swap_ancillas = 1

                        cnot = ControlledGate("cx",
                                              2,
                                              [],
                                              num_ctrl_qubits=1,
                                              ctrl_state="1",
                                              base_gate=XGate())

                        # log-depth ancilla preparation
                        while swap_ancillas < largest_size:
                            for sa in range(swap_ancillas):

                                source = starting_ancilla + sa  # actual address

                                self._max_used_ancilla += 1
                                target = sum(self._nb_qubits) + self._max_used_ancilla

                                if self._max_used_ancilla >= self._nb_ancillas:
                                    raise AncillaIndexError("Not enough ancillas")

                                C_L.append(cnot, [source, target])
                                circ = QuantumCircuit(*self._qr, self._ar)
                                circ.append(cnot, [source, target])
                                C_R = circ.compose(C_R)

                                swap_ancillas += 1

                                if swap_ancillas >= largest_size:
                                    break

                            else:
                                continue  # only executed if the inner loop did NOT break
                            break

                        # performing controlled-swaps

                        cswap = ControlledGate("CSWAP",
                                               3,
                                               [],
                                               num_ctrl_qubits=1,
                                               ctrl_state="1",
                                               base_gate=SwapGate())


                        for step in transposition_list:
                            if step:
                                i = 0
                                for qubit_pair in step:
                                    [q1, q2] = qubit_pair

                                    source = starting_ancilla + i
                                    C_L.append(cswap, [source, q1, q2])
                                    circ = QuantumCircuit(*self._qr, self._ar)
                                    circ.append(cswap, [source, q1, q2])
                                    C_R = circ.compose(C_R)

                                    i += 1

                else:
                    if DEBUG:
                        print(f"in optimize: calling {proc_identifier} on input {new_L}")
                    if len(cs) > 0:
                        # ANCHORING
                        self._max_used_ancilla += 1
                        ancilla = sum(self._nb_qubits) + self._max_used_ancilla
                        if self._max_used_ancilla >= self._nb_ancillas:
                            raise AncillaIndexError("Not enough ancillas")
                        Ancillas[(proc_identifier, tuple([len(new_L[reg]) for reg in self._qubit_registers]), int_parameter)] = [ancilla, new_L]
                        #print("anc", Ancillas)
                        # gate = ControlledGate("CX",
                        #                       1 + len(cs),
                        #                       [],
                        #                       num_ctrl_qubits=len(cs),
                        #                       ctrl_state="".join(str(i) for _, i in reversed(sorted(cs.items()))),
                        #                       base_gate=XGate())
                        
                        C_L.mcx(list(sorted(cs)),ancilla, ctrl_state = self._create_control_state(cs))

                        #C_L.append(gate, list(sorted(cs)) + [ancilla])

                        circ = QuantumCircuit(*self._qr, self._ar)
                        circ.mcx(list(sorted(cs)),ancilla, ctrl_state = self._create_control_state(cs))
                        #circ.append(gate, list(sorted(cs)) + [ancilla])
                        C_R = circ.compose(C_R)

                        l_CST.append(
                            ({ancilla: 1}, self._functions[proc_identifier].children[-1], new_L, variables))
                    else:
                        l_CST.append(
                            ({}, self._functions[proc_identifier].children[-1], new_L, variables))

            else:
                raise NotImplementedError(
                    f"Statement {ast.data} not yet handled in optimize.")

            l_CST.sort(key=lambda x: self._input_ordering(x))





        if self._old_optimize:

            # for (cs,ast,L,variables) in l_M:
                
            #     c = self._compr_lstatement(ast,L,cs,variables)
            #     print(ast.data,cs, c.size())

            #     C_M = C_M.compose(c)
            #     print("size C_M", C_M.size())

            return C_L.compose(C_R)


        else:
            # SEQUENTIAL SPLIT

            l_M_split = []

            C_M = QuantumCircuit(*self._qr, self._ar)


            for (cs, ast, L, variables) in l_M:

                for index, value in enumerate(self._sequential_split(cs, ast, L, variables)):
                    
                    try:
                        l_M_split[index].append(value)
                    except IndexError:
                        l_M_split.append([value])

            # print(len(l_M_split))
            # print("l_M split", l_M_split)

            for l_t in l_M_split:
                rec_split = self._recursive_split(l_t)
                for index, value in enumerate(rec_split):
                    # non-recursive
                    if index == 0:
                        for (cs, ast, L, variables) in value:
                            C_M = C_M.compose(self._compr_statement(ast, L, cs, variables))
                    else:
                        C_M = C_M.compose(self._optimize(value))


            return C_L.compose(C_M).compose(C_R)
        
        return C_L.compose(C_R)





    # CONTEXTUAL LIST
    def _sequential_split(self, cs, ast, L, variables):
        if ast.data == "lstatement":
            seq = []
            for child in ast.children:
                sub_statements = self._sequential_split(cs, child, L, variables)
                if sub_statements:
                    seq.extend(self._sequential_split(cs, child, L, variables))
            return seq

        if ast.data == "skip_statement":
            return []

        elif ast.data in ["gate_application", "cnot_gate", "swap_gate", "toffoli_gate"]:
            return [(cs, ast, L, variables)]

        elif ast.data == "if_statement":
            if self._compr_boolean_expression(ast.children[0], L, cs, variables):
                return self._sequential_split(cs, ast.children[1], L, variables)
            elif len(ast.children) == 3:
                return self._sequential_split(cs, ast.children[2], L,variables)
            else:
                return []

        elif ast.data == "qcase_statement":
            q = self._compr_qubit_expression(ast.children[0], L, cs, variables)
            if q in cs:
                raise IndexError(f"Already controlling on the state of qubit {q}.")
            
            cs0, cs1 = cs.copy(), cs.copy()
            cs0[q] = 0
            cs1[q] = 1

            return self._sequential_split(cs0, ast.children[1], L, variables) + self._sequential_split(cs1, ast.children[2], L, variables)

        elif ast.data == "procedure_call":
            return [(cs, ast, L, variables)]

        else:
            raise ValueError(f"Statement {ast.data} not treated in sequential_split")

    def _recursive_split(self, L):
        m = max([v for k, v in self._mutually_recursive_indices.items()]) + 2
        split = [[] for i in range(m)]
        for (cs, ast, L, variables) in L:
            if ast.data == "procedure_call":
                proc_identifier = ast.children[0].value
                if self._width_function(proc_identifier) != 0:
                    split[self._mutually_recursive_indices[proc_identifier] + 1].append((cs, ast, L, variables))
                else:
                    split[0].append((cs, ast, L, variables))
            else:
                split[0].append((cs, ast, L, variables))
        return split

    def _width_function(self, function_name):
        return self._width_lstatement(self._functions[function_name].children[-1], function_name)

    def _width_lstatement(self, ast, function_name):
        #print(ast,"\n")
        width = 0
        for child in ast.children:
            width_child = self._width_statement(child, function_name)
            child.width = width_child
            width += width_child
            # if self._optimize_flag and width >= 2:
            #     print(f"Procedure {function_name} has width >1 and will be compiled without optimization.")
            #     self._optimize_flag = False
        ast.width = width
        return width

    def _width_statement(self, ast, function_name):
        if ast.data == "gate_application":
            return 0
        elif ast.data == "cnot_gate":
            return 0
        elif ast.data == "swap_gate":
            return 0
        elif ast.data == "toffoli_gate":
            return 0
        elif ast.data == "if_statement":
            if len(ast.children) == 3:
                return max(self._width_lstatement(ast.children[1], function_name),
                           self._width_lstatement(ast.children[2], function_name))
            return self._width_lstatement(ast.children[1], function_name)

        elif ast.data == "qcase_statement":
            return max(self._width_lstatement(ast.children[1], function_name),
                       self._width_lstatement(ast.children[2], function_name))
        elif ast.data == "procedure_call":
            return self._mutually_recursive_indices[function_name] == self._mutually_recursive_indices[ast.children[0].value]

        elif ast.data == "skip_statement":
            return 0

        else:
            raise NotImplementedError(f"Statement {ast.data} not yet handled.")

    # POST-TREATMENT: REMOVE IDLE ANCILLAS
    def remove_idle_wires(self):
        qc_out = self._compiled_circuit.copy()
        gate_count = count_gates(qc_out)
        for qubit, count in gate_count.items():
            if count == 0 and type(qubit) != qiskit.circuit.Qubit:
                qc_out.qubits.remove(qubit)
                self._nb_ancillas -= 1
        self._compiled_circuit = qc_out
        return self._compiled_circuit





def count_gates(qc: QuantumCircuit) -> dict[int, int]:
    gate_count = {qubit: 0 for qubit in qc.qubits}
    for gate in qc.data:
        for qubit in gate.qubits:
            gate_count[qubit] += 1
    return gate_count


def _create_call_graph(call_graph: nx.DiGraph, f: str, g: Union[lark.Tree, lark.Token]):
    if isinstance(g, lark.Token):
        if g.type == "PROC_IDENTIFIER":
            call_graph.add_edge(f, g.value)
        return
    for child in g.children:
        _create_call_graph(call_graph, f, child)
    return


def _merging_transpositions(first_reg, second_reg) -> list[list[list[int]]]:
    # quick check if no permutation is needed:
    # if first_reg == second_reg: return QuantumCircuit(qr,ar)

    # elif len(first_reg)!= len(second_reg):
    #     raise ValueError(f"Attempt to merge two procedure calls with different input size: {len(first_reg)} and {len(second_reg)}.")

    # Step 0: write out the permutation
    # domain of values: {0,...,max(first_reg,second_reg)}
    domain = max(max(first_reg), max(second_reg))
    not_in_second_reg = [i for i in range(domain + 1) if i not in second_reg]
    D = {k: v for (v, k) in enumerate(first_reg)}

    permutation = []
    for x in range(domain + 1):
        if x in first_reg:
            permutation += [second_reg[D[x]]]
        else:
            permutation += [not_in_second_reg[0]]
            not_in_second_reg = not_in_second_reg[1:]

    to_do = permutation[:]  # numbers still to be processed
    cycles = []      # will hold the final list of cycles
    while len(to_do) > 0:  # not done yet...
        cycle = []
        image = to_do[0]  # pick a starting point for a new cycle
        while not (image in cycle):
            cycle.append(image)  # add element to current cycle
            to_do.remove(image)  # ... and remove it from to-do list
            image = permutation[image]   # find next cycle entry

        cycles.append(cycle)  # store complete cycle
    # return cycles             # return result

    transpositions = [[], []]

    for cyc in cycles:
        # base cases: (0,1) or (0,1,2)
        if len(cyc) == 1:
            continue
        if len(cyc) == 2:  # cicle is already transposition
            if not transpositions[0]:
                transpositions[0].append(cyc)
            else:
                transpositions[0] += [cyc]

            continue
        else:
            transpositions[0].append([cyc[0], cyc[2]])
            transpositions[1].append([cyc[1], cyc[2]])
            if len(cyc) == 3:
                continue
            else:
                transpositions[1].append([cyc[0], cyc[3]])
                n = len(cyc) - 1
                i, j = 0, 3
                round = 0
                while n - i > j:
                    transpositions[round].append([cyc[n - i], cyc[j]])
                    if not round:
                        j += 1
                    else:
                        i += 1
                    round = 1 - round

    return transpositions


if __name__ == "__main__":
    parser = argparse.ArgumentParser( 
        description='Compile some PFOQ programs to quantum circuits.')
    
    parser.add_argument('-f', '--filename', type=str, nargs="+",
                        help='Some input PFOQ programs to compile.')
    
    parser.add_argument('-i', '--inputsizes', type=int, nargs="+",
                        help='Qubit input sizes.')
    
    parser.add_argument('-d', '--display', type=bool,
                        help='Indicates if the circuit should be displayed with Matplotlib.',
                        default=True)
    
    parser.add_argument('-s', '--save', type=bool,
                        help='Indicates if the circuit should be saved to a file.',
                        default=False)
    
    parser.add_argument('--optimize', action=argparse.BooleanOptionalAction,
                        help='Indicates if procedure calls should be merged. Defaults to \'True\'.',
                        default=True)
    
    parser.add_argument('--barriers', action=argparse.BooleanOptionalAction,
                        help='Determines whether or not the circuit is displayed ' \
                        'sequentially according to the pfoq-compiler, or if parallel ' \
                        'gates are performed concurrently. Defaults to \'True\', where ' \
                        'sequential order is imposed for displaying purposes.',
                        default=True)
    
    parser.add_argument('--old-optimize', action='store_true',
                        help='Determines whether or not to use compile or compileplus.' \
                        'Defaults to \'False\', where compileplus is used.',
                        default=False)

    args = parser.parse_args()

    

    FILENAMES = ["examples/cat_state_parallel.pfoq"]
    filenames = FILENAMES if args.filename is None else [filename for filename in args.filename]

    for filename in filenames:

        compiler = PfoqCompiler(filename=filename,
                                nb_qubits=args.inputsizes,
                                optimize_flag=args.optimize,
                                barriers=args.barriers,
                                old_optimize=args.old_optimize)
        compiler.parse()
        compiler.compile()

        #print(simulate_statevector_outcomes(compiler._compiled_circuit))

        if args.save:
            compiler.save()

        if args.display:
            compiler.display()
