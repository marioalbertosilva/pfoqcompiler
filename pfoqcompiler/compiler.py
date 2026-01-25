
"""
Main class for PFOQ programs compilation
"""


import lark
from lark import Tree, Token
import argparse
import matplotlib.pyplot as plt
import networkx as nx
from typing_extensions import Optional, Union, Sequence
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit import Qubit
from qiskit.circuit.library import HGate, XGate, CCXGate, SwapGate, RYGate, CPhaseGate, PhaseGate, Barrier
from qiskit.circuit import Gate, ControlledGate, CircuitInstruction
from qiskit.visualization import circuit_drawer
import qiskit.qasm3
from math import ceil, pi

from pfoqcompiler.errors import WellFoundedError, WidthError, AncillaIndexError, NotCompiledError
from pfoqcompiler.parser import PfoqParser


_DEBUG = False


class PfoqCompiler:
    """Compiler class for PFOQ programs.

    Instantiate a compiler to compile the provided PFOQ program.

    Parameters
    ----------
    program: str, optional
        PFOQ program to compile. Mutually exclusive with `filename`

    filename: str, optional
        Path linking to the file containing the PFOQ program to compile.

    nb_qubits: Sequence[int]
        Number of data qubits to consider, aka the size of the input.

    nb_ancillas: int
        Number of ancilla qubits available to work with.

    optimize_flag: bool
        Whether a compilation technique optimizing the circuit is run, default is True.

    old_optimize: bool
        Toggle either the old optimization procedure or the new one, default is False.

    barriers: bool
        Whether some invisible barriers are added to the circuit to clarify the display, default is False.

    _no_ast: bool
        Internal option to skip the program acquisition.
        The user is expected to modify the internal attribute _ast themselves before any compilation.

    Examples
    --------
    >>> prg = "decl f(q){q[0]*=H;call f(q-[0]);}:: define q; :: call f(q);"
    >>> compiler = PfoqCompiler(program=prg)
    >>> compiler.parse()
    >>> compiler.compile()

    """

    def __init__(self,
                 program: Optional[str] = None,
                 filename: Optional[str] = None,
                 nb_qubits: Sequence[int] = [8],
                 nb_ancillas: int = 1,
                 optimize_flag: bool = True,
                 old_optimize: bool = False,
                 debug_flag : bool = False,
                 verbose_flag : bool = False,
                 barriers: bool = False,
                 _no_ast: bool = False):

        if _no_ast:
            pass

        elif len((tmp_set := {program, filename} - {None})) == 0:
            raise ValueError("Either program or filename argument must be provided.")

        elif len(tmp_set) == 2:
            raise ValueError("At most one of program and filename must be provided.")

        self._filename = filename

        if filename is not None:
            with open(filename, 'r') as f:
                program = f.read()

        self._program = program
        self._parser = PfoqParser()
        self._nb_qubits = nb_qubits
        self._nb_ancillas = nb_ancillas
        self._nb_total_wires: int = sum(self._nb_qubits) + self._nb_ancillas
        self._qr = []
        self._ar = AncillaRegister(self._nb_ancillas, name="|0\\rangle")
        self._functions = {} # information about procedure statements
        self._qubit_registers = []
        self._mutually_recursive_indices = {}
        self._max_used_ancilla = -1
        self._ast = None
        self._compiled_circuit = None
        self._optimize_flag = optimize_flag
        self._debug_flag = debug_flag
        self._verbose_flag = verbose_flag
        self._old_optimize = old_optimize
        self._enforce_order = barriers

    @property
    def compiled_circuit(self):
        return self._compiled_circuit

    @property
    def ast(self):
        return self._ast

    def parse(self):
        try:
            self._ast = self._parser.parse(self._program)

            if self._debug_flag:
                print(self._ast)

        except Exception as exception:
            if self._debug_flag:
                if self._filename is None:
                    print("Parsing of program failed due to:")
                else:
                    print(f"Parsing of file {self._filename} failed due to:")
            raise exception

        if self._verbose_flag:
            print(f"File \'{self._filename}\' parsed successfully.")

    

    def verify(self):
        """Statically check the following properties:
        - Well-foundedness: all cycles in the call graph reduce the number of accessible qubits
        - Halving (subsumes well-foundedness): all cycles in the call graph reduce by half some input qubit list
        - Bounded-width: recursive procedure calls occur on orthogonal computation branches   
        """

        assert self._ast is not None, "No AST is available."

        if self._verbose_flag:
            print("\nProperties:")

        if len(self._ast.children) == 0:
            raise RuntimeWarning("Empty program")

        for child in self._ast.children[:-2]:
            assert isinstance(child, Tree)
            if _DEBUG:
                assert _get_data(child) == "decl"

            function_name = _get_data(child.children[0])
            
            self._functions[function_name] = child

        graph = self._compute_call_graph()

        if self._debug_flag:
            nx.draw(graph)

        components = list(nx.strongly_connected_components(graph))
        for index, value in enumerate(components):
            for name in value:
                self._mutually_recursive_indices[name] = index


        # Well foundedness check

        zero_weight_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("weight", None) == 0]

        zero_weight_subgraph = nx.DiGraph()
        zero_weight_subgraph.add_nodes_from(graph.nodes())
        zero_weight_subgraph.add_edges_from(zero_weight_edges)

        well_founded = True

        try:
            zero_cycle = nx.find_cycle(zero_weight_subgraph)
            well_founded = False
            raise WellFoundedError("Program is not well-founded. The following call cycle does not remove any qubits:",
                                   " -> ".join(zero_cycle))

        except nx.NetworkXNoCycle:
            if self._verbose_flag:
                print("- Well-founded")

        # HALVING check

        #create call subgraph without -2 edges
        minus_two_excluded_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("weight", -42) >= -1]

        minus_two_excluded_subgraph = nx.DiGraph()
        minus_two_excluded_subgraph.add_nodes_from(graph.nodes())
        minus_two_excluded_subgraph.add_edges_from(minus_two_excluded_edges)


        halving = True

        try:
            minus_two_avoiding_cycle = list(nx.find_cycle(minus_two_excluded_subgraph)[0])
            halving = False
            if self._verbose_flag:
                print("- NOT halving. An example call cycle that does not reduce qubits by half:",
                      " -> ".join(minus_two_avoiding_cycle))

        except nx.NetworkXNoCycle:
            if self._verbose_flag:
                print("- Halving")

        # WIDTH <= 1 check
        # this step stores a width parameter in the tree that controls the flow in optimize
        max_width = 0

        for function in self._functions:
            width = self._width_function(function)
            max_width = max(width,max_width)
                

            if self._optimize_flag and width > 1:
                if self._verbose_flag:
                    print(f"Procedure {function} has width {width}. Turning off optimization.")
                self._optimize_flag = False


        if max_width <= 1 and self._verbose_flag:
            print(f"- Bounded width: {max_width}")

    def compile(self, remove_idle_wires: bool = True):
        """Compile the program.

        Compile the program according to some parametrizable rules.

        Parameters
        ----------
        remove_idle_wires : bool
            Whether to unused ancilla qubits from the output.

        Raises
        ------
        exception

        Examples
        --------
        >>> prg = "decl f(q){q[0]*=H;call f(q-[0]);}::define q;::call f(q);"
        >>> compiler = PfoqCompiler(program=prg)
        >>> compiler.parse()
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
                if self._debug_flag:
                    print(f"Insufficient ancillas, doubling number to {self._nb_ancillas}", flush=True)
                self._ar = AncillaRegister(self._nb_ancillas, name="|0\\rangle")
                self._max_used_ancilla = -1

            except Exception as exception:
                if self._debug_flag:
                    if self._filename is None:
                        print("Compilation of program failed due to:")
                    else:
                        print(f"Compilation of file {self._filename} failed due to:")
                raise exception

        if self._verbose_flag:
            print(f"\nCompiled circuit using {self._nb_ancillas} anchoring and merging ancillas.", flush=True)
            print("Compiled circuit in file \"circuits/"
                  + self._filename.split(".")[0] + "_"
                  + "_".join([str(i) for i in self._nb_qubits]) + ".pdf\"", flush=True)

    def save(self, filename: str) -> None:
        """
        Saves the compiled circuit in OpenQASM 3 format.
        """

        if self._compiled_circuit is None:
            raise NotCompiledError("The circuit hasn't been successfully compiled.")
        
        with open(f"{filename}.qasm", "w") as f:
            qiskit.qasm3.dump(self._compiled_circuit, f)

    def display(self) -> None:
        """
        Generates a circuit representation in matplotlib.
        """

        if self._compiled_circuit is None:
            raise NotCompiledError("The circuit hasn't been successfully compiled.")

        try:
            if self._enforce_order:

                N = sum(self._nb_qubits) + self._nb_ancillas

                for i in range(len(self._compiled_circuit.data)):
                    self._compiled_circuit.data.insert(
                            2*i+1, CircuitInstruction(Barrier(N), range(N)))
                
            circuit_drawer(self._compiled_circuit, output="mpl", style="bw",
                            fold=-1, plot_barriers=False)
            #plt.show()

            plt.savefig("circuits/" + self._filename.split(".")[0] + "_" + "_".join([str(i) for i in self._nb_qubits]), format="pdf")


        except Exception as exception:
            if _DEBUG:
                print("Program has been successfully compiled, but could not be displayed due to:")
            raise exception
        

    def _compr_prg(self) -> QuantumCircuit:
        """
        Compiles program statement of the input program.
        """
        assert self._ast is not None, "No ast is available"
        program_statement = self._ast.children[-1]


        if _DEBUG:
            assert (_get_data(program_statement) == "lstatement")

        register_definition = self._ast.children[-2]
        assert isinstance(register_definition, Tree)

        if _DEBUG:
            assert (_get_data(register_definition) == "def")

        # relative qubit addresses
        L = {}
        nb_qubits_before = 0
        for reg, nb in zip(register_definition.children, self._nb_qubits):
            L[_get_data(reg)] = list(range(nb_qubits_before, nb+nb_qubits_before))
            nb_qubits_before += nb

        self._qubit_registers = register_definition.children

        if _DEBUG:
            assert (len(self._nb_qubits) == len(self._qubit_registers) )

        self._qr = [QuantumRegister(nb, name=f"{reg}") for reg, nb in zip(register_definition.children, self._nb_qubits)]

        qc = QuantumCircuit(*self._qr, self._ar)
        qc.compose(self._compr_lstatement(ast=program_statement, L=L, cs={}, variables={}, cqubits = {}), inplace=True)
        
        return qc


    def _compute_call_graph(self) -> nx.DiGraph:
        """
        Generates the program call graph.
        """
        initial_graph = nx.DiGraph()
        for f, g in self._functions.items():
            initial_graph.add_node(f)
            _create_call_graph(initial_graph, f, g)
        return initial_graph

    def _compr_lstatement(self, ast, L, cs, variables, cqubits) -> QuantumCircuit:
        qc = QuantumCircuit(*self._qr, self._ar)
        for child in ast.children:
            qc.compose(self._compr_statement(child, L, cs, variables, cqubits), inplace=True)
        return qc


    def _compr_statement(self, ast, L, cs, variables, cqubits) -> QuantumCircuit:
        """
        Manages compilation of a PFOQ statement by calling different subfunctions.
        """

        match ast.data:

            case "gate_application":
                return self._compr_gate_application(ast, L, cs, variables, cqubits)
            
            case "cnot_gate":
                return self._compr_cnot_gate(ast, L, cs, variables, cqubits)
            
            case "swap_gate":
                return self._compr_swap_gate(ast, L, cs, variables, cqubits)
            
            case "toffoli_gate":
                return self._compr_toffoli_gate(ast, L, cs, variables, cqubits)
            
            case "if_statement":
                if self._compr_disjunction(ast.children[0], L, cs, variables, cqubits):
                    return self._compr_lstatement(ast.children[1], L, cs, variables, cqubits)
                elif len(ast.children) == 3:
                    return self._compr_lstatement(ast.children[2], L, cs, variables, cqubits)
                else:
                    return QuantumCircuit(*self._qr, self._ar)

            case "qcase_statement":

                q = self._compr_qubit_expression(ast.children[0], L, cs, variables, cqubits)

                if q in cqubits:
                    raise IndexError(f"Already controlling on the state of qubit {q}.")
                circuit = QuantumCircuit(*self._qr, self._ar)

                cs[q] = 0
                cqubits[q] = 0

                circuit.compose(self._compr_lstatement(ast.children[1], L,
                                                       cs, variables, cqubits), inplace=True)
                cs[q] = 1
                cqubits[q] = 1

                circuit.compose(self._compr_lstatement(ast.children[2], L,
                                                       cs, variables, cqubits), inplace=True)
                del cs[q]
                del cqubits[q]

                return circuit
            
            case "qcase_statement_two_qubits":

                q1 = self._compr_qubit_expression(ast.children[0], L, cs, variables, cqubits)
                q2 = self._compr_qubit_expression(ast.children[1], L, cs, variables, cqubits)

                for q in [q1,q2]:
                    if q in cqubits:
                        raise IndexError(f"Already controlling on the state of qubit {q}.")
                

                circuit = QuantumCircuit(*self._qr, self._ar)

                # case 00

                cs[q1] = 0
                cs[q2] = 0
                cqubits[q1] = 0
                cqubits[q2] = 0

                circuit.compose(self._compr_lstatement(ast.children[2], L, cs, variables, cqubits), inplace=True)

                # case 01

                cs[q2] = 1
                cqubits[q2] = 1

                circuit.compose(self._compr_lstatement(ast.children[3], L, cs, variables, cqubits), inplace=True)

                # case 10
                
                cs[q1] = 1
                cs[q2] = 0
                cqubits[q1] = 1
                cqubits[q2] = 0

                circuit.compose(self._compr_lstatement(ast.children[4], L, cs, variables, cqubits), inplace=True)

                # case 11
                
                cs[q2] = 1
                cqubits[q2] = 1

                circuit.compose(self._compr_lstatement(ast.children[5], L, cs, variables, cqubits), inplace=True)

                del cs[q1]
                del cqubits[q1]
                del cs[q2]
                del cqubits[q2]

                return circuit

            case "procedure_call":
                return self._compr_procedure_call(ast, L, cs, variables, cqubits)

            case "skip_statement":
                return QuantumCircuit(*self._qr, self._ar)

            case _:
                raise NotImplementedError(f"Statement {ast.data} not handled.")

    def _compr_gate_application(self,
                                ast: lark.Tree,
                                L: dict[str,list[int]],
                                cs: dict[int,int],
                                variables: dict[lark.Token,int],
                                cqubits: dict[int,int]) -> QuantumCircuit:

        qubit = self._compr_qubit_expression(ast.children[0], L, cs, variables, cqubits)

        if qubit in cqubits:
            raise IndexError(f"Cannot apply gate on qubit {qubit} that is controlled on its state.")

        qc = QuantumCircuit(*self._qr, self._ar)

        gate_name = ast.children[1].data

        match gate_name:

            case "not_gate":
                if cs: qc.mcx(list(sorted(cs)),qubit,ctrl_state = _create_control_state(cs))
                else: qc.x(qubit)

            case "hadamard_gate":
                if cs:
                    cH = HGate().control(num_ctrl_qubits=len(cs),
                                         label="H",
                                        ctrl_state=_create_control_state(cs))
                    
                    qc.append(cH,list(sorted(cs)) + [qubit])
                else:
                    qc.h(qubit)
            
            case "rotation_gate":

                theta = self._compr_int_expression(ast.children[1].children[-1],L,cs,variables) #integer input given to gate

                if len(ast.children[1].children) == 2: 

                    func = _get_data(ast.children[1].children[0]) #function parameter given as string
                    theta = eval(func)(theta)

                ry = RYGate(theta)

                if cs:
                    cRY = ry.control(num_ctrl_qubits=len(cs),
                                        label=f"Ry({theta})",
                                        ctrl_state=_create_control_state(cs))
                    
                    qc.append(cRY,list(sorted(cs)) + [qubit])
                else:
                    qc.ry(theta, qubit)
            
            case "phase_shift_gate":

                theta = self._compr_int_expression(ast.children[1].children[-1],L,cs,variables) #integer input given to gate

                if len(ast.children[1].children) == 2: 

                    func = _get_data(ast.children[1].children[0]) #function parameter given as string
                    theta = eval(func)(theta)

                if cs:
                    qc.mcp(theta, list(sorted(cs)),qubit,ctrl_state = _create_control_state(cs))
                else:
                    qc.p(theta, qubit)

            case "toffoli_gate":
                
                qubits = [self._compr_qubit_expression(ast.children[i], L, cs, variables, cqubits) for i in range(3)]

                for i in range(3):
                    if qubits[i] in cs:
                        raise IndexError(f"Cannot apply Toffoli on control qubit {qubits[i]}.")

                qc = QuantumCircuit(*self._qr, self._ar)

                qc.mcx(list(sorted(cs)) + qubits[:1], qubits[2], ctrl_state = "11" + _create_control_state(cs))

            case "cnot_gate":

                qubit1 = self._compr_qubit_expression(ast.children[0], L, cs, variables,cqubits)
                if qubit1 in cs:
                    raise IndexError(f"Multiple controls on same qubit {qubit1}.")
                
                cs[qubit1] = 1

                qubit2 = self._compr_qubit_expression(ast.children[1], L, cs, variables,cqubits)

                if qubit2 in cs:
                    raise IndexError(
                        f"Cannot apply gate on qubit {qubit2} that is controlled on its state.")
                
                qc.mcx(list(sorted(cs)),qubit2, ctrl_state = _create_control_state(cs))
                
                del cs[qubit1]

                return qc

            case "other_gates":
                gate_ast = ast.children[1]

                if cs:
                    gate = HGate().control(num_ctrl_qubits=len(cs),
                                        label="C" + _get_data(gate_ast.children[0])[1:-1],
                                        ctrl_state=_create_control_state(cs))
                    
                else: 
                    gate = Gate(_get_data(gate_ast.children[0])[1:-1], 1, [])

                qc.append(gate,list(sorted(cs)) + [qubit])

            case _:
                raise ValueError(f"Unexpected gate value {gate_name}.")


        return qc


    def _compr_cnot_gate(self, ast, L, cs, variables, cqubits) -> QuantumCircuit:

        qubit1 = self._compr_qubit_expression(ast.children[0], L, cs, variables, cqubits)
        if qubit1 in cqubits:
            raise IndexError(f"Multiple controls on same qubit {qubit1}.")
        
        cs[qubit1] = 1
        cqubits[qubit1] = 1

        qubit2 = self._compr_qubit_expression(ast.children[1], L, cs, variables, cqubits)

        if qubit2 in cqubits:
            raise IndexError(
                f"Cannot apply gate on qubit {qubit2} that is controlled on its state.")

        qc = QuantumCircuit(*self._qr, self._ar)


        qc.mcx(list(sorted(cs)),qubit2, ctrl_state = _create_control_state(cs))
        
        del cs[qubit1]
        del cqubits[qubit1]

        return qc
    


    def _compr_swap_gate(self, ast, L, cs, variables, cqubits) -> QuantumCircuit:

        qubit1 = self._compr_qubit_expression(ast.children[0], L, cs, variables, cqubits)
        if qubit1 in cqubits:
            raise IndexError(f"Cannot swap control qubit {qubit1}.")
        
        qubit2 = self._compr_qubit_expression(ast.children[1], L, cs, variables, cqubits)
        
        if qubit2 in cqubits:
            raise IndexError(f"Cannot swap control qubit {qubit2}.")

        qc = QuantumCircuit(*self._qr, self._ar)

        swap = SwapGate()

        if cs:
            gate = swap.control(num_ctrl_qubits=len(cs),
                                label=None,
                                ctrl_state=_create_control_state(cs))
        else:
            gate = SwapGate()

        qc.append(gate, list(sorted(cs)) + [qubit1, qubit2])
        return qc
    
    def _compr_toffoli_gate(self, ast, L, cs, variables, cqubits) -> QuantumCircuit:

        Q = [self._compr_qubit_expression(ast.children[i], L, cs, variables, cqubits) for i in range(3)]

        for i in range(3):
            if Q[i] in cqubits:
                raise IndexError(f"Cannot apply Toffoli on control qubit {Q[i]}.")

        qc = QuantumCircuit(*self._qr, self._ar)


        qc.mcx(list(sorted(cs)) + Q[:2], Q[2],ctrl_state="11"+_create_control_state(cs))


        return qc

    def _compr_procedure_call(self, ast, L, cs, variables, cqubits) -> QuantumCircuit:
        

        proc_identifier = ast.children[0].value

        qregs = []
        int_variables = []


        for child in self._functions[proc_identifier].children:

            if isinstance(child, lark.Token):
            
                match child.type:

                    case "REGISTER_VARIABLE":
                        qregs += [child.value]

                    case "INT_IDENTIFIER":
                        int_variables += [child.value]


        if proc_identifier not in self._functions:
            raise NameError(f"Called function {proc_identifier} was not declared.")
        
        

        new_L = L.copy()


        for index in range(-len(qregs),0):
            new_L[qregs[index]] = self._compr_register_expression(ast.children[index], L, cs, variables)


        if [] in new_L.values():
            return QuantumCircuit(*self._qr, self._ar)


        function = self._functions[proc_identifier]
        


        num_int_variables = len(int_variables)

        function_parameters = function.children[1:num_int_variables+1]

        if len(function_parameters) != num_int_variables:
            raise (ValueError(
                f"Incorrect number of parameters passed to function {proc_identifier}"))

        int_parameters = {}

        for i in range(num_int_variables):
            int_parameters[int_variables[i]] = self._compr_int_expression(ast.children[1+i], L, cs, variables)

        old_values = {}

        for param in function_parameters:
            
            if variables.get(param):
                old_values[param] = variables[param] # save previous value if it exsists
            
            variables[param] = int_parameters[param]


        if _DEBUG:
            print(f"in compr: calling {proc_identifier} on input {new_L}")

        if new_L:
            if self._optimize_flag and self._width_function(proc_identifier) == 1:
                l_CST = [(cs, self._functions[proc_identifier].children[-1], new_L, variables, cqubits)]
                return self._optimize(l_CST)

            circ = self._compr_lstatement(
                self._functions[proc_identifier].children[-1], new_L, cs, variables, cqubits)
            
            if int_parameters:
                for param in function_parameters:
                    if not old_values.get(param):
                        del variables[param]
                    else:
                        variables[param] = old_values[param]

            return circ
        
        return QuantumCircuit(*self._qr, self._ar)



    def _compr_qubit_expression(self, ast, L, cs, variables, cqubits) -> int:

        match ast.data:

            case "qubit_expression_identifier":
                qubit_list = self._compr_register_identifier(ast, L, cs, variables)

            case "qubit_expression_variable":
                qubit_list = self._compr_register_variable(ast, L, cs, variables)

            case "qubit_expression_parenthesed":
                qubit_list = self._compr_parenthesed_register_expression(ast.children[0], L, cs, variables)

            case _:
                raise NotImplementedError(f"Qubit expression {ast.data} not yet handled.")


        integer_value = self._compr_int_expression(ast.children[1], L, cs, variables)

        try:
            qubit = qubit_list[integer_value]

        except IndexError:
            raise IndexError(
                f"Qubit index {integer_value} cannot reference a qubit in a register of size {len(qubit_list)}.")

        return qubit


    def _compr_register_identifier(self, ast, L, cs, variables) -> list[int]:
        return L[ast.children[0].value]
    
    def _compr_register_variable(self, ast, L, cs, variables) -> list[int]:
        return L[ast.children[0].value]

    def _compr_parenthesed_register_expression(self, ast, L, cs, variables) -> list[int]:
        return self._compr_register_expression(ast.children[0], L, cs, variables)

    def _compr_parenthesed_register_expression_first_half(self, ast, L, cs, variables) -> list[int]:
        qubit_list = self._compr_parenthesed_register_expression(ast.children[0], L, cs, variables)
        if len(qubit_list) <= 1:
            return []
        else:
            m = ceil(len(qubit_list)/2)
            return qubit_list[:m]
        
    def _compr_parenthesed_register_expression_second_half(self, ast, L, cs, variables) -> list[int]:
        qubit_list = self._compr_parenthesed_register_expression(ast.children[0], L, cs, variables)
        if len(qubit_list) <= 1:
            return []
        else:
            m = ceil(len(qubit_list)/2)
            return qubit_list[m:]
        
        
    def _compr_register_identifier_first_half(self,ast,L,cs,variables) -> list[int]:
        qubit_list = L[ast.children[0].value]
        if len(qubit_list) <= 1:
            return []
        else:
            m = ceil(len(qubit_list)/2)
            return qubit_list[:m]

    def _compr_register_identifier_second_half(self,ast,L,cs,variables) -> list[int]:

        qubit_list = L[ast.children[0].value]
        if len(qubit_list) <= 1:
            return []
        else:
            m = ceil(len(qubit_list)/2)
            return qubit_list[m:]

    def _compr_register_expression_minus(self, ast, L, cs, variables) -> list[int]:

        qubit_list = self._compr_register_expression(ast.children[0], L, cs, variables)
        indices = [self._compr_int_expression(ast.children[i], L, cs, variables) for i in range(1,len(ast.children))]

        N = len(qubit_list)
        if min(indices) < -N or max(indices) > N - 1:
            raise IndexError("list index out of range")        
        
        nonnegative_indices = [ i % N for i in indices]
        #indices written as nonnegative values in decreasing order

        return [qubit for index,qubit in enumerate(qubit_list) if index not in nonnegative_indices]

    def _compr_register_expression(self, ast, L, cs, variables) -> list[int]:

        match ast.data:

            case "register_expression_identifier":
                return self._compr_register_identifier(ast, L, cs, variables)
            
            case "register_variable":
                return self._compr_register_variable(ast, L, cs, variables)
            
            case "register_expression_parenthesed" | "parenthesed_register_expression":
                return self._compr_parenthesed_register_expression(ast, L, cs, variables)
            
            case "register_expression_minus":
                return self._compr_register_expression_minus(ast, L, cs, variables)
            
            case "register_expression_parenthesed_first_half":
                return self._compr_parenthesed_register_expression_first_half(ast, L, cs, variables)
            
            case "register_expression_parenthesed_second_half":
                return self._compr_parenthesed_register_expression_second_half(ast, L, cs, variables)

            case "register_identifier_first_half":
                return self._compr_register_identifier_first_half(ast, L, cs, variables)
            
            case "register_identifier_second_half":
                return self._compr_register_identifier_second_half(ast, L, cs, variables)
            
            case _:
                raise NotImplementedError(
                        f"Register expression {ast.data} not handled.")

    def _compr_disjunction(self, ast, L, cs, variables, cqubits) -> bool:
        match ast.data:
            case "multiple_conjs": # Lazy evaluation
                return any(self._compr_conjunction(child, L, cs, variables, cqubits) for child in ast.children)
            case "conjunction":
                return self._compr_conjunction(ast.children[0], L, cs, variables, cqubits)
            case _:
                raise NotImplementedError(f"Missing disjunction handling for {ast.data}.")

    def _compr_conjunction(self, ast, L, cs, variables, cqubits) -> bool:
        match ast.data:
            case "multiple_disjs": # Lazy evaluation
                return all(self._compr_invert(child, L, cs, variables, cqubits) for child in ast.children)
            case "disjonction":
                return self._compr_invert(ast.children[0], L, cs, variables, cqubits)
            case _:
                raise NotImplementedError(f"Missing conjunction handling for {ast.data}.")
            
    def _compr_invert(self, ast, L, cs, variables, cqubits) -> bool:
        match ast.data:
            case "inversion":
                return not self._compr_disjunction(ast.children[0], L, cs, variables, cqubits)
            case "boolean_expression":
                return self._compr_boolean_expression(ast.children[0], L, cs, variables, cqubits)
            case _:
                raise NotImplementedError(f"Missing inversion handling for {ast.data}.")

    def _compr_boolean_expression(self, ast, L, cs, variables, cqubits) -> bool:

        match ast.data:
            case "bool_literal":
                return ast.children[0].value == "true"
            case "bool_greater_than":
                return self._compr_int_expression(ast.children[0], L, cs, variables) > self._compr_int_expression(ast.children[1], L, cs, variables)
            case "bool_greatereq_than":
                return self._compr_int_expression(ast.children[0], L, cs, variables) >= self._compr_int_expression(ast.children[1], L, cs, variables)            
            case "bool_smaller_than":
                return self._compr_int_expression(ast.children[0], L, cs, variables) < self._compr_int_expression(ast.children[1], L, cs, variables)
            case "bool_smallereq_than":
                return self._compr_int_expression(ast.children[0], L, cs, variables) <= self._compr_int_expression(ast.children[1], L, cs, variables)
            case "bool_equals":
                return self._compr_int_expression(ast.children[0], L, cs, variables) == self._compr_int_expression(ast.children[1], L, cs, variables)
            case "bool_different":
                return self._compr_int_expression(ast.children[0], L, cs, variables) != self._compr_int_expression(ast.children[1], L, cs, variables)
            case "par_disj":
                return self._compr_disjunction(ast.children[0], L, cs, variables, cqubits)
            case _:
                raise NotImplementedError(f"Boolean expression {ast.data} not handled.")


    def _compr_int_expression(self, ast, L, cs, variables) -> int:

        match ast.data:

            case "int_expression_literal":
                return int(ast.children[0].value)
            
            case "binary_op":

                if ast.children[1].value == "+":
                    return int(self._compr_int_expression(ast.children[0], L, cs, variables)
                               + self._compr_int_expression(ast.children[2], L, cs, variables))
                elif ast.children[1].value == "-":
                    return int(self._compr_int_expression(ast.children[0], L, cs, variables)
                               - self._compr_int_expression(ast.children[2], L, cs, variables))
                else:
                    raise NotImplementedError(f"Only valid binary operations are + and -, not {_get_data(ast.children[1])}.")

            case "size_of_register":
                return len(self._compr_register_expression(ast.children[0], L, cs, variables))
            
            case "int_expression_identifier":

                variable_name = ast.children[0].value

                if variable_name not in variables:
                    raise ValueError(f"Variable {variable_name} not defined.")
                
                return variables[variable_name]

            case "parenthesed_int_expression":
                return self._compr_int_expression(ast.children[0], L, cs, variables)

            case "parenthesed_int_expression_half":
                return int(self._compr_int_expression(ast.children[0], L, cs, variables)/2)
            
            case "int_expression_half_size":
                return int(len(self._compr_register_expression(ast.children[0], L, cs, variables))/2)
            
            case _:
                raise NotImplementedError(
                    f"Integer expression {ast.data} not yet handled.")
            

    #PARTIAL ORDERING ON INPUTS
    def _input_ordering(self, x):

        return -max(len(x[2][reg]) for reg in self._qubit_registers)

    def _optimize(self, l_CST):

        Ancillas = {}
        C_L, C_R = QuantumCircuit(*self._qr, self._ar), QuantumCircuit(*self._qr, self._ar)
        l_M = []

        while l_CST:

            cs, ast, L, variables, cqubits = l_CST.pop(0)

            match ast.data:

                case "lstatement":

                    before = True
                    for child in ast.children:
                        if child.width == 0:
                            if before:
                                C_L.compose(self._compr_statement(child, L, cs, variables, cqubits), inplace=True)
                            else:
                                C_R.compose(self._compr_statement(child, L, cs, variables, cqubits), front=True, inplace=True)
                        else:
                            before = False
                            l_CST.append((cs, child, L, variables, cqubits))
                

                case "if_statement":

                    guard = self._compr_disjunction(ast.children[0], L, cs, variables, cqubits)

                    if guard:

                        if ast.children[1].width:
                            l_CST.append((cs, ast.children[1], L, variables, cqubits))

                        elif self._old_optimize:
                            C_R.compose(self._compr_lstatement(ast.children[1], L, cs, variables, cqubits), front=True, inplace=True)

                        else:                            
                            l_M.append((cs, ast.children[1], L, variables, cqubits))

                    elif len(ast.children) == 3:

                        if ast.children[2].width:
                            l_CST.append((cs, ast.children[2], L, variables, cqubits))

                        elif self._old_optimize:
                            C_R.compose(self._compr_lstatement(ast.children[2], L, cs, variables, cqubits), front=True, inplace=True)

                        else:  
                            l_M.append((cs, ast.children[2], L, variables, cqubits))


                case "qcase_statement":

                    q = self._compr_qubit_expression(ast.children[0], L, cs, variables, cqubits)
            
                    if q in cqubits:
                        raise IndexError(
                            f"Already controlling on the state of qubit {q}.")

                    cs_0, cs_1 = cs.copy(), cs.copy()
                    cs_0[q] = 0
                    cs_1[q] = 1

                    cqubits_0, cqubits_1 = cqubits.copy(), cqubits.copy()
                    cqubits_0[q] = 0
                    cqubits_1[q] = 1


                    if ast.children[1].width:
                        l_CST.append((cs_0, ast.children[1], L, variables,cqubits_0))

                    elif self._old_optimize:
                        C_R.compose(self._compr_lstatement(ast.children[1], L, cs_0, variables, cqubits, cqubits_1), front=True, inplace=True)      
                    
                    else:
                        l_M.append((cs_0, ast.children[1], L, variables, cqubits))



                    if ast.children[2].width:
                        l_CST.append((cs_1, ast.children[2], L, variables, cqubits))

                    elif self._old_optimize:
                        C_R.compose(self._compr_lstatement(ast.children[2], L, cs_1, variables, cqubits), front=True, inplace=True) 

                    else:           
                        l_M.append((cs_1, ast.children[2], L, variables, cqubits))


                case "qcase_statement_two_qubits":

                    q1 = self._compr_qubit_expression(ast.children[0], L, cs, variables, cqubits)
                    q2 = self._compr_qubit_expression(ast.children[1], L, cs, variables, cqubits)
            
                    for q in [q1, q2]:
                        if q in cqubits:
                            raise IndexError(
                                f"Already controlling on the state of qubit {q}.")
                        

                    cs_00, cs_01, cs_10, cs_11 = cs.copy(), cs.copy(), cs.copy(), cs.copy()
                    cs_00[q1] = 0
                    cs_00[q2] = 0
                    cs_01[q1] = 0
                    cs_01[q2] = 1
                    cs_10[q1] = 1
                    cs_10[q2] = 0
                    cs_11[q1] = 1
                    cs_11[q2] = 1

                    all_cs = [cs_00, cs_01, cs_10, cs_11]



                    cqubits_00, cqubits_01, cqubits_10, cqubits_11 = cqubits.copy(), cqubits.copy(), cqubits.copy(), cqubits.copy()
                    cqubits_00[q1] = 0
                    cqubits_00[q2] = 0
                    cqubits_01[q1] = 0
                    cqubits_01[q2] = 1
                    cqubits_10[q1] = 1
                    cqubits_10[q2] = 0
                    cqubits_11[q1] = 1
                    cqubits_11[q2] = 1


                    all_cqubits = [cqubits_00, cqubits_01, cqubits_10, cqubits_11]


                    for i in range(4):
                        child_index = i + 2

                        if ast.children[child_index].width:
                            l_CST.append((all_cs[i], ast.children[child_index], L, variables, all_cqubits[i]))

                        elif self._old_optimize:
                            C_R.compose(self._compr_lstatement(ast.children[child_index], L, all_cs[i], variables, all_cqubits[i]), front=True, inplace=True)  

                        else:
                            l_M.append((all_cs[i], ast.children[child_index], L, variables, all_cqubits[i]))



                case "procedure_call":

                    proc_identifier = ast.children[0].value

                    qregs = []

                    int_variables = []

                    for child in self._functions[proc_identifier].children:
                       
                        if isinstance(child, lark.Token):
                        
                            match child.type:

                                case "REGISTER_VARIABLE":
                                    qregs += [child.value]

                                case "INT_IDENTIFIER":
                                    int_variables += [child.value]


                    if proc_identifier not in self._functions:
                        raise NameError(
                            f"Called function {proc_identifier} was not declared.")


                    
                    new_L = L.copy()

                    for index in range(-len(qregs),0):
                        new_L[qregs[index]] = self._compr_register_expression(ast.children[index], L, cs, variables)

                    if [] in new_L.values():
                        continue

                    

                    function = self._functions[proc_identifier]
                    


                    num_int_variables = len(int_variables)

                    function_parameters = function.children[1:num_int_variables+1]

                    if len(function_parameters) != num_int_variables:
                        raise (ValueError(
                            f"Incorrect number of parameters passed to function {proc_identifier}"))

                    int_parameters = {}

                    for i in range(num_int_variables):
                        int_parameters[int_variables[i]] = self._compr_int_expression(ast.children[1+i], L, cs, variables)

                    old_values = {}

                    for param in function_parameters:
                        
                        if variables.get(param):
                            old_values[param] = variables[param] # save previous value if it exsists
                        
                        variables[param] = int_parameters[param]


                    reg_sizes = tuple([len(value) for key, value in sorted(new_L.items())])

                    int_values = tuple([value for key,value in sorted(variables.items())])

                    if (proc_identifier,reg_sizes, int_values) in Ancillas:
                        # merging

                        ancilla, anchored_L = Ancillas[(proc_identifier, reg_sizes, int_values)]

                        C_L.mcx(list(sorted(cs)),ancilla, ctrl_state = _create_control_state(cs))

                        circ = QuantumCircuit(*self._qr, self._ar)
                        circ.mcx(list(sorted(cs)),ancilla, ctrl_state = _create_control_state(cs))

                        C_R.compose(circ, front=True, inplace=True)


                        anchored_register, merging_register = [], []

                        for reg in sorted(qregs):
                            anchored_register += anchored_L[reg]
                            merging_register += new_L[reg]


                        if merging_register!= anchored_register:
                            
                            transposition_list = _merging_transpositions(merging_register, anchored_register)
                            largest_size = max([len(i) for i in transposition_list])

                            self._max_used_ancilla += 1
                            starting_ancilla = sum(self._nb_qubits) + self._max_used_ancilla

                            if self._max_used_ancilla >= self._nb_ancillas:
                                raise AncillaIndexError("Not enough ancillas")

                            C_L.mcx(list(sorted(cs)),starting_ancilla, ctrl_state = _create_control_state(cs))
                            
                            circ = QuantumCircuit(*self._qr, self._ar)

                            circ.mcx(list(sorted(cs)),starting_ancilla, ctrl_state = _create_control_state(cs))
                            
                            C_R.compose(circ, front=True, inplace=True)

                            swap_ancillas = 1

                            # log-depth ancilla preparation
                            while swap_ancillas < largest_size:
                                for sa in range(swap_ancillas):

                                    source = starting_ancilla + sa  # actual address

                                    self._max_used_ancilla += 1
                                    target = sum(self._nb_qubits) + self._max_used_ancilla

                                    if self._max_used_ancilla >= self._nb_ancillas:
                                        raise AncillaIndexError("Not enough ancillas")

                                    C_L.cx(source,target)
                                    
                                    circ = QuantumCircuit(*self._qr, self._ar)
                                    circ.cx(source,target)
                                    
                                    C_R.compose(circ, front=True, inplace=True)

                                    swap_ancillas += 1

                                    if swap_ancillas >= largest_size:
                                        break

                                else:
                                    continue
                                break


                            for step in transposition_list:
                                if step:
                                    i = 0
                                    for qubit_pair in step:
                                        [q1, q2] = qubit_pair

                                        source = starting_ancilla + i
                                        C_L.cswap(source,q1,q2)
                                        
                                        circ = QuantumCircuit(*self._qr, self._ar)
                                        
                                        circ.cswap(source,q1,q2)
                                        C_R.compose(circ, front=True, inplace=True)

                                        i += 1

                    else:
                        if _DEBUG:
                            print(f"in optimize: calling {proc_identifier} on input {new_L}")

                        if len(cs) > 0:
                            # ANCHORING
                            self._max_used_ancilla += 1
                            ancilla = sum(self._nb_qubits) + self._max_used_ancilla
                            if self._max_used_ancilla >= self._nb_ancillas:
                                raise AncillaIndexError("Not enough ancillas")
                            
                            reg_sizes = tuple([len(value) for key, value in sorted(new_L.items())])
                            int_values = tuple([value for key,value in sorted(variables.items())])

                            Ancillas[(proc_identifier, reg_sizes, int_values)] = [ancilla, new_L]
                            
                            C_L.mcx(list(sorted(cs)),ancilla, ctrl_state = _create_control_state(cs))

                            circ = QuantumCircuit(*self._qr, self._ar)
                            circ.mcx(list(sorted(cs)),ancilla, ctrl_state =_create_control_state(cs))
                            
                            C_R.compose(circ, front=True, inplace=True)

                            l_CST.append(
                                ({ancilla: 1}, self._functions[proc_identifier].children[-1], new_L, variables, cqubits))
                        else:
                            l_CST.append(
                                ({}, self._functions[proc_identifier].children[-1], new_L, variables, cqubits))

                case _:
                    raise NotImplementedError(
                        f"Statement {ast.data} not handled in optimize.")

            l_CST.sort(key=lambda x: self._input_ordering(x))



        if self._old_optimize:
            C_L.compose(C_R, inplace=True)
            return C_L


        else:

            l_M_split = []

            C_M = QuantumCircuit(*self._qr, self._ar)


            for (cs, ast, L, variables,cqubits) in l_M:

                for index, value in enumerate(self._sequential_split(cs, ast, L, variables, cqubits)):
                    
                    try:
                        l_M_split[index].append(value)
                    except IndexError:
                        l_M_split.append([value])

            for l_t in l_M_split:
                rec_split = self._recursive_split(l_t)
                for index, value in enumerate(rec_split):
                    # non-recursive
                    if index == 0:
                        for (cs, ast, L, variables, cqubits) in value:
                            C_M.compose(self._compr_statement(ast, L, cs, variables, cqubits), inplace=True)
                    else:
                        C_M.compose(self._optimize(value), inplace=True)

            C_M.compose(C_R, inplace=True)
            C_M.compose(C_L, front=True, inplace=True)
            return C_M


    # CONTEXTUAL LIST
    def _sequential_split(self, cs, ast, L, variables, cqubits):

        match ast.data:

            case "lstatement":
                seq = []
                for child in ast.children:
                    sub_statements = self._sequential_split(cs, child, L, variables, cqubits)
                    if sub_statements:
                        seq.extend(self._sequential_split(cs, child, L, variables, cqubits))
                return seq
            
            case "skip_statement":
                return []
            
            case "gate_application" | "cnot_gate" | "swap_gate" | "toffoli_gate":
                return [(cs, ast, L, variables, cqubits)]
            
            case "if_statement":

                if self._compr_disjunction(ast.children[0], L, cs, variables, cqubits):
                    return self._sequential_split(cs, ast.children[1], L, variables, cqubits)
                
                elif len(ast.children) == 3:
                    return self._sequential_split(cs, ast.children[2], L,variables, cqubits)
                
                return []
            
            case "qcase_statement":

                q = self._compr_qubit_expression(ast.children[0], L, cs, variables, cqubits)

                if q in cqubits:
                    raise IndexError(
                        f"Already controlling on the state of qubit {q}.")
                
                cs0, cs1 = cs.copy(), cs.copy()
                cs0[q] = 0
                cs1[q] = 1

                cqubits0, cqubits1 = cs.copy(), cs.copy()
                cqubits0[q] = 0
                cqubits1[q] = 1

                return self._sequential_split(cs0,
                                            ast.children[1],
                                            L,
                                            variables,
                                            cqubits0) + self._sequential_split(cs1, ast.children[2], L, variables, cqubits1)
            
            case "qcase_statement_two_qubits":


                    q1 = self._compr_qubit_expression(ast.children[0], L, cs, variables, cqubits)
                    q2 = self._compr_qubit_expression(ast.children[1], L, cs, variables, cqubits)
            
                    for q in [q1, q2]:
                        if q in cqubits:
                            raise IndexError(
                                f"Already controlling on the state of qubit {q}.")
                        

                    cs_00, cs_01, cs_10, cs_11 = cs.copy(), cs.copy(), cs.copy(), cs.copy()
                    cs_00[q1] = 0
                    cs_00[q2] = 0
                    cs_01[q1] = 0
                    cs_01[q2] = 1
                    cs_10[q1] = 1
                    cs_10[q2] = 0
                    cs_11[q1] = 1
                    cs_11[q2] = 1

                    all_cs = [cs_00, cs_01, cs_10, cs_11]



                    cqubits_00, cqubits_01, cqubits_10, cqubits_11 = cqubits.copy(), cqubits.copy(), cqubits.copy(), cqubits.copy()
                    cqubits_00[q1] = 0
                    cqubits_00[q2] = 0
                    cqubits_01[q1] = 0
                    cqubits_01[q2] = 1
                    cqubits_10[q1] = 1
                    cqubits_10[q2] = 0
                    cqubits_11[q1] = 1
                    cqubits_11[q2] = 1


                    all_cqubits = [cqubits_00, cqubits_01, cqubits_10, cqubits_11]

                    out = []

                    for i in range(4):

                        out += self._sequential_split(all_cs[i], ast.children[2+i], L, variables, all_cqubits[i]) 
                    
                    return out

                    

            
            case "procedure_call":
                return [(cs, ast, L, variables, cqubits)]
            
            case _:
                raise ValueError(
                    f"Statement {ast.data} not treated in sequential_split")
    

    def _recursive_split(self, L):
        m = max([v for k, v in self._mutually_recursive_indices.items()]) + 2
        split = [[] for i in range(m)]
        for (cs, ast, L, variables, cqubits) in L:
            if ast.data == "procedure_call":
                proc_identifier = ast.children[0].value
                if self._width_function(proc_identifier) != 0:
                    split[self._mutually_recursive_indices[proc_identifier] + 1].append((cs, ast, L, variables, cqubits))
                else:
                    split[0].append((cs, ast, L, variables, cqubits))
            else:
                split[0].append((cs, ast, L, variables, cqubits))
        return split

    def _width_function(self, function_name):
        return self._width_lstatement(self._functions[function_name].children[-1], function_name)

    def _width_lstatement(self, ast, function_name):
    
        width = 0

        for child in ast.children:

            width_child = self._width_statement(child, function_name)
            child.width = width_child
            width += width_child

        ast.width = width

        return width

    def _width_statement(self, ast, function_name):

        match ast.data:

            case "skip_statement" | "gate_application" | "cnot_gate" | "swap_gate" | "toffoli_gate":
                return 0
            
            case "if_statement":
                if len(ast.children) == 3:
                    return max(
                            self._width_lstatement(ast.children[1], function_name),
                            self._width_lstatement(ast.children[2], function_name))
                
                return self._width_lstatement(ast.children[1], function_name)

            case "qcase_statement":
                return max(
                        self._width_lstatement(ast.children[1], function_name),
                        self._width_lstatement(ast.children[2], function_name))
            
            case "qcase_statement_two_qubits":
                return max(
                        self._width_lstatement(ast.children[2], function_name),
                        self._width_lstatement(ast.children[3], function_name),
                        self._width_lstatement(ast.children[4], function_name),
                        self._width_lstatement(ast.children[5], function_name))
            
            case "procedure_call":
                return self._mutually_recursive_indices[function_name] == self._mutually_recursive_indices[ast.children[0].value]

            case _:
                raise NotImplementedError(
                    f"Statement {ast.data} not handled.")
            


    # POST-TREATMENT: REMOVE IDLE ANCILLAS
    def remove_idle_wires(self):
        assert self._compiled_circuit is not None, "No compiled circuit."
        qc_out = self._compiled_circuit.copy()
        gate_count = count_gates(qc_out)
        for qubit, count in gate_count.items():
            if count == 0 and type(qubit) != Qubit:
                qc_out.qubits.remove(qubit)
                self._nb_ancillas -= 1
                self._nb_total_wires -= 1
        self._compiled_circuit = qc_out
        return self._compiled_circuit


def _create_control_state(cs: dict[int, int]) -> str:
    """ Create control state from dictionary cs.

    Parameters
    ----------
    cs : dict
        Dictionary of qubit addresses and respective state

    Examples
    --------
    >>> cs = {0:1, 2:1, 5:0}
    >>> _create_control_state(cs)
    '011'
    """
    return "".join(str(i) for _, i in reversed(sorted(cs.items())))


def count_gates(qc: QuantumCircuit) -> dict[Qubit, int]:
    """ Count the number of gates applied to each qubit

    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit to analyze

    Examples
    --------
    >>> qc = QuantumCircuit(2)
    >>> qc.h(0)
    <...>
    >>> qc.cx(0, 1)
    <...>
    >>> count = count_gates(qc)
    >>> count[qc.qubits[0]]
    2
    >>> count[qc.qubits[1]]
    1

    """
    gate_count = {qubit: 0 for qubit in qc.qubits}
    for gate in qc.data:
        for qubit in gate.qubits:
            gate_count[qubit] += 1
    return gate_count


def _create_call_graph(call_graph: nx.DiGraph, f: str, g: Union[lark.Tree, lark.Token]) -> None:

    if isinstance(g, lark.Tree):

        if g.data == "procedure_call":

            proc_identifier = _get_data(g.children[0])

            qubit_difference = 0

            input_registers = []

            for input in g.children[1::]:

                if "register_expression" in input.data:

                    qubit_difference = min(qubit_difference,_determine_qubit_difference(input))
                    
                    register_variable = _find_register_variable(input)

                    if register_variable in input_registers:

                        raise ValueError(
                            f"Repeated input {register_variable} on call to {proc_identifier}")
                    
                    else: input_registers += [register_variable]

            

            call_graph.add_edge(f, proc_identifier, weight = qubit_difference)
    
        for child in g.children:
            _create_call_graph(call_graph, f, child)
            
    return


def _find_register_variable(g: Union[lark.Tree, lark.Token]) -> str:

    if isinstance(g, lark.Tree):
        # DFS on first node:

        for child in g.children:
            out = _find_register_variable(child)
            if out: 
                return out
        raise ValueError("No register variables found.")
    
    elif isinstance(g, lark.Token):

        if g.type != "REGISTER_VARIABLE":
            raise ValueError(
                f"Procedure calls on qubit inputs not allowed. Found example for qubit list '{g}'."
            )

        return g.value
    
    else:
        raise ValueError(f"Received unexpected value {g}.")


def _get_data(larkobject: Union[Tree, Token]) -> str:
    if isinstance(larkobject, Tree):
        return larkobject.data
    elif isinstance(larkobject, Token):
        return larkobject.value
    else:
        raise TypeError(f"Only a Tree or a Token can have their data extracted, not {type(larkobject).__name__}.")


def _determine_qubit_difference(g: Union[lark.Tree, lark.Token]) -> int:
    
    if isinstance(g, lark.Tree):

        match g.data:
        
            case "register_expression_parenthesed_first_half" | "register_expression_parenthesed_second_half":

                return -2 # halving qubit list
            
            case "register_expression_minus":

                return min(-1, min(_determine_qubit_difference(child) for child in g.children)) # at least reducing qubit list
            
            case _:

                return min(_determine_qubit_difference(child) for child in g.children)

    return 0






def _merging_transpositions(first_reg: list[int], second_reg: list[int]) -> list[list[list[int]]]:
    """
    Given any two registers first_reg and second_reg of equal length,
    outputs a list describing two sets of disjoint transpositions (i.e. length 2 cycles)
    that transform first_reg into second_reg.

    Examples
    --------
    >>> first_reg, second_reg = [1,2,3], [2,4,5]
    >>> _merging_transpositions(first_reg, second_reg)
    [[[2, 1], [5, 3]], [[4, 1]]]
    >>> second_reg = [2,4,5,1]
    >>> _merging_transpositions(first_reg, second_reg)
    ValueError: Registers [1, 2, 3] and [2, 4, 5, 1] have different lengths.
    """

    if len(first_reg) != len(second_reg): raise ValueError(
        f"Registers {first_reg} and {second_reg} have different lengths.")
    
    domain: list[int] = list(set(first_reg).union(second_reg))

    first_not_in_second: list[int] = [qubit for qubit in first_reg if qubit not in second_reg]
    second_not_in_first: list[int] = [qubit for qubit in second_reg if qubit not in first_reg]

    # start by determining permutation

    permutation: dict = {} # to be defined over the domain

    for i, qubit in enumerate(first_reg):
        permutation[qubit] = second_reg[i]

    for i, qubit in enumerate(second_not_in_first):
        permutation[qubit] = first_not_in_second[i]

    # define permutation as sequence of disjoint cycles

    cycle_list: list[list[int]] = []

    qubit_list = domain.copy()

    while qubit_list:

        current = qubit_list[0]

        qubit_list.remove(current)

        cycle = [current]

        next = permutation[current]

        while next != cycle[0]:
            
            qubit_list.remove(next)

            cycle += [next]

            next = permutation[next]

        if len(cycle) > 1:

            cycle_list += [cycle]
    
    # transform cycles into two sets of transpositions

    transpositions = [[], []]

    for cyc in cycle_list:
        # base cases, e.g. (0,1) or (0,1,2)
        if len(cyc) == 1:
            continue
        if len(cyc) == 2:
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
    
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction,
                        help='Output debugging information.' \
                        'Defaults to \'False\'.',
                        default=False)
    
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction,
                        help='Output more informtion about the compilation,' \
                        'including statically verified properties.' \
                        'Defaults to \'False\'.',
                        default=False)

    args = parser.parse_args()

    

    FILENAMES = ["examples/cat_state_parallel.pfoq"]
    filenames = FILENAMES if args.filename is None else [filename for filename in args.filename]

    for filename in filenames:

        compiler = PfoqCompiler(filename=filename,
                                nb_qubits=args.inputsizes,
                                optimize_flag=args.optimize,
                                barriers=args.barriers,
                                old_optimize=args.old_optimize,
                                debug_flag = args.debug,
                                verbose_flag = args.verbose)
        

        compiler.parse()

        compiler.verify()

        compiler.compile()

        if args.save:
            compiler.save()

        if args.display:
            compiler.display()