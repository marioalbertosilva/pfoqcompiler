import unittest
from pfoqcompiler.compiler import PfoqCompiler
from pfoqcompiler.errors import WidthError, AncillaIndexError, NotCompiledError
from qiskit.quantum_info import Statevector
from typing_extensions import Optional, Union
from qiskit_aer import AerSimulator
import numpy as np




def indexket(string):
    return int(string, 2)


class ProgramTester():
    """Class for testing FOQ programs.

    Parameters
    ----------
    program: str
        PFOQ program to test.

    inout: dict[tuple[int],list[tuple[Union[Statevector,str]]]]
        Inputs to be tested and corresponding outputs. Dictionary of register sizes mapped to input-output pairs.

    flags: dict, optional

        classical: bool
            whether to run a classical simulation (for circuits consiting of classical gates)

    """

    def __init__(self,
                 program: str,
                 inout: dict[tuple[int],list[tuple[Union[Statevector,str]]]],
                 *args,
                 **flags):
        

        self.classical_simulation = flags.get("classical", False)
        self.program = program
        self.inout = inout

        # self.simulator = AerSimulator(method="statevector")
        self.tests = unittest.TestSuite()


        for input_sizes, inout_list in self.inout.items():
            compiler = PfoqCompiler(program=program,
                nb_qubits=input_sizes,
                optimize_flag=True,
                barriers=False,
                old_optimize=False)


            self.tests.addTest(TestPFOQParsing(compiler))

            self.tests.addTest(TestPFOQCompilation(compiler))

            for input, output in inout_list:
                self.tests.addTest(TestPFOQExecution(compiler, input, output))


class TestPFOQParsing(unittest.TestCase):

    def __init__(self, compiler, methodName="test_parse"):
        self.compiler = compiler
        super().__init__(methodName)

    def test_parse(self):
        self.compiler.parse()


class TestPFOQCompilation(unittest.TestCase):

    def __init__(self, compiler, methodName="test_compile"):
        self.compiler = compiler
        super().__init__(methodName)

    def setUp(self):
        if self.compiler.ast is None:
            self.skipTest("Unsuccessful parsing")
            return

    def test_compile(self):
        self.assertIsNotNone(self.compiler.ast)
        self.compiler.compile()

    # def run(self, result=None):
    #     if self.compiler.ast is None:
    #         result.stop()
    #         return
    #     super().run(result=result)

class TestPFOQExecution(unittest.TestCase):

    def __init__(self, compiler: PfoqCompiler, input, output, methodName="test_exec"):
        self.compiler = compiler
        self.input = input
        self.output = output
        super().__init__(methodName)

    def setUp(self):
        if self.compiler.compiled_circuit is None:
            self.skipTest("Unsuccessful compilation")
            return
        self.input = manageinout(self.input, self.compiler.compiled_circuit)
        self.output = manageinout(self.output, self.compiler.compiled_circuit)

    def test_exec(self):

        self.assertIsNotNone(self.compiler.compiled_circuit)
        self.assertTrue((compiled := self.input.evolve(self.compiler.compiled_circuit)).equiv(self.output),
                        f"Obtained output {sv_to_dict(compiled)} on input {sv_to_dict(self.input)}, expected {sv_to_dict(self.output)}")
        
    # def run(self, result=None):
    #     if self.compiler.compiled_circuit is None:
    #         print("iStop")
    #         result.stop()
    #         return
    #     super().run(result=result)



def manageinout(inout,circ):
    state = np.zeros(int(2**circ.num_qubits))
    state[indexket(inout.zfill(circ.num_qubits))] = 1

    return Statevector(state)

def sv_to_dict(sv: Statevector, precision = 4):
    return {str(state): round(float(amplitude.real), precision)+1j*round(float(amplitude.imag), precision) for state, amplitude in sv.to_dict().items() if np.absolute(amplitude) > 10**(-precision)}




if __name__ == '__main__':

    program_tester = ProgramTester(program=open("examples/pairs.pfoq","r").read(),
                                   inout={(7,):[("0"*7,"1"+"0"*6),
                                                ("1111101","0"+"1"*6)],
                                            (9,):[("0"*9,"1"+"0"*8),
                                                  ("1"*9,"0"+"1"*8)]})
    
    program_tester2= ProgramTester(program=open("examples/pairs_error.pfoq","r").read(),
                                   inout={(7,):[("0"*7,"1"+"0"*6),
                                                ("1111101","0"+"1"*6)],
                                            (9,):[("0"*9,"1"+"0"*8),
                                                  ("1"*9,"0"+"1"*8)]})




    runner = unittest.TextTestRunner()
    runner.run(program_tester.tests)
    runner.run(program_tester2.tests)