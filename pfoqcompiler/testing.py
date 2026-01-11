import unittest
from pfoqcompiler.compiler import PfoqCompiler
from pfoqcompiler.errors import WidthError, AncillaIndexError, NotCompiledError
from qiskit.quantum_info import Statevector
from typing_extensions import Optional, Union
import numpy as np


class ProgramTester():
    """Class for testing FOQ programs.

    Automatize the testing of the parsing, compilation and execution of a FOQ program.

    Parameters
    ----------
    program: str
        PFOQ program to test.

    inout: dict[tuple[int],list[tuple[Union[Statevector,str]]]]
        Inputs to be tested and corresponding outputs. Dictionary of register sizes mapped to input-output pairs.

    flags: dict, optional

        classical: bool
            whether to run a classical simulation (for circuits consiting of classical gates)

    Examples
    --------
    >>> prg = "decl pairs(q){if(|q|>1)then{qcase(q[0]) of {0->qcase(q[1]) of{0->call pairs(q-[0,1]);,1->skip;},1->qcase(q[1]) of{0->skip;,1->call pairs(q-[0,1]);}}}else{q[0]*=NOT;}}::define q;::call pairs(q);"
    >>> tester = ProgramTester(program=prg, inout={(3,): [("000", "100"), ("101", "101")]})
    >>> tester.run()
    >>> tester = ProgramTester(program=prg+"err", inout={(3,): [("000", "100"), ("101", "101")]})
    >>> tester.run()  # Expected parsing error that skips following tests

    """

    def __init__(self,
                 program: str,
                 inout: dict[tuple[int], list[tuple[Union[Statevector, str]]]],
                 *args,
                 **flags):

        self.classical_simulation = flags.get("classical", False)
        self.program = program
        self.inout = inout
        self.tests = unittest.TestSuite()

        if len(inout) == 0:
            return

        dummy_compiler = PfoqCompiler(program=program,
                                      nb_qubits=next(iter(self.inout)),
                                      optimize_flag=True,
                                      barriers=False,
                                      old_optimize=False)

        self.tests.addTest(TestPFOQParsing(dummy_compiler))
        for input_sizes, inout_list in self.inout.items():
            compiler = PfoqCompiler(_ast=True,
                                    nb_qubits=input_sizes,
                                    optimize_flag=True,
                                    barriers=False,
                                    old_optimize=False)

            self.tests.addTest(TestPFOQCompilation(compiler, dummy_compiler))

            for input, output in inout_list:
                self.tests.addTest(TestPFOQExecution(compiler, input, output))

    def run(self):
        """
        Run the planned tests

        """
        runner = unittest.TextTestRunner()
        runner.run(self.tests)


class TestPFOQParsing(unittest.TestCase):
    """TestCase of the parsing of a configured compiler.

    compiler: PfoqCompiler
        Compiler with program whose parsing is to be tested.

    """

    def __init__(self, compiler, methodName="test_parse"):
        self.compiler = compiler
        super().__init__(methodName)

    def test_parse(self):
        self.compiler.parse()


class TestPFOQCompilation(unittest.TestCase):
    """TestCase of the compilation of a configured compiler.

    compiler: PfoqCompiler
        Compiler with program whose compilation is to be tested.

    """

    def __init__(self, compiler, dummy_compiler=None, methodName="test_compile"):
        self.compiler = compiler
        self.dummy_compiler = dummy_compiler
        super().__init__(methodName)

    def setUp(self):
        if self.dummy_compiler is not None:
            self.compiler._ast = self.dummy_compiler.ast
        if self.compiler.ast is None:
            self.skipTest("Unsuccessful parsing")
            return

    def test_compile(self):
        self.assertIsNotNone(self.compiler.ast)
        self.compiler.verify()
        self.compiler.compile()


class TestPFOQExecution(unittest.TestCase):
    """TestCase of the execution of a compiled FOQ program.

    compiler: PfoqCompiler
        Compiler with program whose compilation is to be tested.

    """

    def __init__(self, compiler: PfoqCompiler, input, output, methodName="test_exec"):
        self.compiler = compiler
        self.input = input
        self.output = output
        super().__init__(methodName)

    def setUp(self):
        if self.compiler.compiled_circuit is None:
            self.skipTest("Unsuccessful compilation")
            return
        self.input = manageinout(self.input, self.compiler)
        self.output = manageinout(self.output, self.compiler)

    def test_exec(self):

        self.assertIsNotNone(self.compiler.compiled_circuit)
        self.assertTrue((compiled := self.input.evolve(self.compiler.compiled_circuit)).equiv(self.output),
                        f"Obtained output {sv_to_dict(compiled)} on input {sv_to_dict(self.input)}, expected {sv_to_dict(self.output)}")


def manageinout(inout, compiler):
    state = np.zeros(int(2**compiler.compiled_circuit.num_qubits))
    state[indexket(inout.zfill(compiler._nb_total_wires))] = 1

    return Statevector(state)


def sv_to_dict(sv: Statevector, precision=4):
    return {str(state): round(float(amplitude.real), precision)+1j*round(float(amplitude.imag), precision) for state, amplitude in sv.to_dict().items() if np.absolute(amplitude) > 10**(-precision)}


def indexket(string):
    return int(string, 2)


if __name__ == '__main__':
    from pfoqcompiler import __examples_directory__
    import os
    os.chdir(__examples_directory__)

    program_tester = ProgramTester(program=open("qcase_SWAP.pfoq", "r").read(),
                                   inout={(4,): [("0000", "0000"),   ("1000", "0001"),   ("0101", "1100"),  ("0101","1100"),  ("0010","1000")],
                                          (5,): [("00000", "00000"), ("10000", "00001"), ("00101", "01100"),("00101","01100"),("00010","10000")]})

    program_tester2 = ProgramTester(program=open("qcase_CNOT.pfoq", "r").read(),
                                   inout={(3,): [("000", "011"), ("001", "000"), ("010", "001"), ("011", "010"), ("100", "111"), ("101", "100"), ("110", "101"), ("111", "110")]} )

    program_tester3 = ProgramTester(program=open("pairs.pfoq", "r").read(),
                                   inout={(5,): [("00000","10000"), ("10000","00000"),
                                                 ("00001","00001"), ("10001","10001"),
                                                 ("00010","00010"), ("10010","10010"),
                                                 ("00011","10011"), ("10011","00011")] })
    
    program_tester4 = ProgramTester(program=open("boolean_semantics.pfoq", "r").read(),
                                   inout={(14,): [("00000000000000","11111111111111")] })


    print("Testing qcase_SWAP")
    program_tester.run()

    print("Testing qcase_CNOT")
    program_tester2.run()

    print("Testing pairs")
    program_tester3.run()

    print("Testing boolean semantics")
    program_tester4.run()

