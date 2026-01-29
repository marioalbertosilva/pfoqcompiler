import unittest
from pfoqcompiler.compiler import PfoqCompiler
from pfoqcompiler.errors import WidthError, AncillaIndexError, NotCompiledError
from qiskit.quantum_info import Statevector
from typing_extensions import Optional, Union, Literal, Any, Type
import numpy as np


class ProgramTester():
    """Class for testing FOQ programs.

    Automatize the testing of the parsing, compilation and execution of a FOQ program.

    Parameters
    ----------
    program: str
        PFOQ program to test.

    inout: dict[tuple[int, ...], list[tuple[Union[Statevector, str], ...]]]
        Inputs to be tested and corresponding outputs. Dictionary of register sizes mapped to input-output pairs.

    expected_error: type[Exception], optional
        Type of the expected raised exception, if any

    expected_error_stage: ("parsing" | "compilation" | "runtime"), optional
        If an error is expected, the stage at which it should occur must me specified.
        It can be raised either during parsing, compilation or at runtime execution.

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
                 *,
                 program: str,
                 inout: dict[tuple[int, ...], list[tuple[Union[Statevector, str], ...]]],
                 expected_error: Optional[Type[Exception]] = None,
                 expected_error_stage: Optional[Literal["parsing", "compilation", "runtime"]] = None,
                 **flags: dict[str, Any]):

        self.classical_simulation = flags.get("classical", False)
        self.program = program
        self.inout = inout
        self.tests = unittest.TestSuite()

        if expected_error_stage not in (None, "parsing", "compilation", "runtime"):
            raise ValueError(f"Provided expected error stage should be in (None, 'parsing', 'compilation', 'runtime'), not {expected_error_stage}.")

        if len(inout) == 0:
            return

        dummy_compiler = PfoqCompiler(program=program,
                                      nb_qubits=next(iter(self.inout)),
                                      optimize_flag=True,
                                      barriers=False,
                                      old_optimize=False)

        if expected_error_stage == "parsing":
            self.tests.addTest(TestPFOQParsing(dummy_compiler, expected_error=expected_error))
        else:
            self.tests.addTest(TestPFOQParsing(dummy_compiler))

        for input_sizes, inout_list in self.inout.items():
            compiler = PfoqCompiler(nb_qubits=input_sizes,
                                    optimize_flag=True,
                                    barriers=False,
                                    old_optimize=False,
                                    _no_ast=True)

            if expected_error_stage == "compilation":
                self.tests.addTest(TestPFOQCompilation(compiler, dummy_compiler, expected_error=expected_error))
            else:
                self.tests.addTest(TestPFOQCompilation(compiler, dummy_compiler))

            for input, output in inout_list:
                if expected_error_stage == "runtime":
                    self.tests.addTest(TestPFOQExecution(compiler, input, output, expected_error=expected_error))
                else:
                    self.tests.addTest(TestPFOQExecution(compiler, input, output, expected_error=expected_error))

    def run(self):
        """
        Run the planned tests.

        """
        runner = unittest.TextTestRunner()
        runner.run(self.tests)


class TestPFOQParsing(unittest.TestCase):
    """TestCase of the parsing of a configured compiler.

    Parameters
    ----------
    compiler: PfoqCompiler
        Compiler with program whose parsing is to be tested.
    methodName: str
        Name of the method to be tested.
    expected_error: type[Exception], optional
        Type of the expected raised exception, if any

    """

    def __init__(self,
                 compiler: PfoqCompiler,
                 methodName: str = "test_parse",
                 expected_error: Optional[Type[Exception]] = None):
        self.compiler = compiler
        self.expected_error = expected_error
        super().__init__(methodName)

    def test_parse(self):
        if self.expected_error is not None:
            with self.assertRaises(self.expected_error):
                self.compiler.parse()
        else:
            self.compiler.parse()
                


class TestPFOQCompilation(unittest.TestCase):
    """TestCase of the compilation of a configured compiler.

    Parameters
    ----------
    compiler: PfoqCompiler
        Compiler with program whose compilation is to be tested.
    dummy_compiler: PfoqCompiler, optional
        Optional compiled program from which the ast can be reused.
    methodName: str
        Name of the method to be tested.
    expected_error: type[Exception], optional
        Type of the expected raised exception, if any

    """

    def __init__(self,
                 compiler: PfoqCompiler,
                 dummy_compiler: Optional[PfoqCompiler] = None,
                 methodName: str = "test_compile",
                 expected_error: Optional[Type[Exception]] = None):
        self.compiler = compiler
        self.dummy_compiler = dummy_compiler
        self.expected_error = expected_error
        super().__init__(methodName)

    def setUp(self):
        if self.dummy_compiler is not None:
            self.compiler._ast = self.dummy_compiler.ast
            self.compiler._functions = self.dummy_compiler._functions

        if self.compiler.ast is None:
            self.skipTest("Unsuccessful parsing")
            return

    def test_compile(self):
        self.assertIsNotNone(self.compiler.ast)
        self.compiler.verify()

        if self.expected_error is not None:
            with self.assertRaises(self.expected_error):
                self.compiler.compile()
        else:
            self.compiler.compile()


class TestPFOQExecution(unittest.TestCase):
    """TestCase of the execution of a compiled FOQ program.

    Parameters
    ----------
    compiler: PfoqCompiler
        Compiler with program whose compilation is to be tested.
    input: Union[Statevector, str]
        Input for the TestCase.
    output: Union[Statevector, str]
        Expected output of the TestCase
    methodName: str
        Name of the method to be tested.
    expected_error: type[Exception], optional
        Type of the expected raised exception, if any

    """

    def __init__(self,
                 compiler: PfoqCompiler,
                 input: Union[Statevector, str],
                 output: Union[Statevector, str],
                 methodName: str = "test_exec",
                 expected_error: Optional[Type[Exception]] = None):
        self.compiler = compiler
        self.input = input
        self.output = output
        self.expected_error = expected_error
        self.failed_setup: Optional[Exception] = None
        super().__init__(methodName)

    def setUp(self):
        if self.compiler.compiled_circuit is None:
            self.skipTest("Unsuccessful compilation")
            return
        
        try:
            self.input_statevector = manageinout(self.input, self.compiler)
            self.output_statevector = manageinout(self.output, self.compiler)
        except Exception as e:
            self.failed_setup = e

    def test_exec(self):
        self.assertIsNotNone(self.compiler.compiled_circuit)
        assert(self.compiler.compiled_circuit is not None)

        if self.expected_error is not None:
            with self.assertRaises(self.expected_error):
                if self.failed_setup is not None:
                    raise self.failed_setup
                self.assertIsNotNone(self.input_statevector)
                assert(self.input_statevector is not None)
                self.assertTrue((compiled := self.input_statevector.evolve(self.compiler.compiled_circuit)).equiv(self.output_statevector),
                                f"Obtained output {sv_to_dict(compiled)} on input {sv_to_dict(self.input_statevector)}, expected {sv_to_dict(self.output_statevector)}")
        else:
            if self.failed_setup is not None:
                    raise self.failed_setup
            self.assertTrue((compiled := self.input_statevector.evolve(self.compiler.compiled_circuit)).equiv(self.output_statevector),
                                f"Obtained output {sv_to_dict(compiled)} on input {sv_to_dict(self.input_statevector)}, expected {sv_to_dict(self.output_statevector)}")


def manageinout(inout: Union[Statevector, str],
                compiler: PfoqCompiler) -> Statevector:
    """Statevector building helper

    Builds the statevector corresponding to the given string sequence of bits.
    
    Parameters
    ----------
    inout: str
        Binary sequence representing a basis statevector

    compiler: PfoqCompiler
        Compiled instance of a PFOQ program the statevector will be used as in/output for.

    Returns
    -------
    Statevector
        Built basis statevector

    """
    if isinstance(inout, str):
        assert compiler.compiled_circuit is not None
        state = np.zeros(int(2**compiler.compiled_circuit.num_qubits))
        state[indexket(inout.zfill(compiler._nb_total_wires))] = 1

        return Statevector(state)
    
    elif isinstance(inout, Statevector):
        raise NotImplementedError
    
    else:
        raise TypeError(f"Given input {inout} of type {type(inout).__name__} is neither a str or a Statevector.")


def sv_to_dict(sv: Statevector,
               precision: int = 4) -> dict[str, complex]:
    """Convert a statevector to a dict

    The output dictionary represents the complex amplitudes of the given statevector
    in the computational basis.
    
    Parameters
    ----------
    sv: Statevector
        Statevector to convert.
    precision: int
        Order of precision according to which the amplitude are rounded.

    Returns
    -------
    dict[str, complex]
        Decomposition of the statevector in the computational basis.

    """
    return {str(state): round(float(amplitude.real), precision)+1j*round(float(amplitude.imag), precision) for state, amplitude in sv.to_dict().items() if np.absolute(amplitude) > 10**(-precision)}


def indexket(string: str) -> int:
    """Base2 to base10 conversion

    Gives the base10 interpretation of given binary sequence.
    
    Parameters
    ----------
    string: str
        Binary sequence to convert

    Returns
    -------
    int
        Interpreted base 10 value

    """
    return int(string, 2)


if __name__ == '__main__':
    from pfoqcompiler import __examples_directory__
    import os
    os.chdir(__examples_directory__)

    program_tester = ProgramTester(program=open("qcase_SWAP.foq", "r").read(),
                                   inout={(4,): [("0000", "0000"),   ("1000", "0001"),   ("0101", "1100"),  ("0101","1100"),  ("0010","1000")],
                                          (5,): [("00000", "00000"), ("10000", "00001"), ("00101", "01100"),("00101","01100"),("00010","10000")]})

    program_tester2 = ProgramTester(program=open("qcase_CNOT.foq", "r").read(),
                                   inout={(3,): [("000", "011"), ("001", "000"), ("010", "001"), ("011", "010"), ("100", "111"), ("101", "100"), ("110", "101"), ("111", "110")]} )

    program_tester3 = ProgramTester(program=open("pairs.foq", "r").read(),
                                   inout={(5,): [("00000","10000"), ("10000","00000"),
                                                 ("00001","00001"), ("10001","10001"),
                                                 ("00010","00010"), ("10010","10010"),
                                                 ("00011","10011"), ("10011","00011")] })
    
    program_tester4 = ProgramTester(program=open("boolean_semantics.foq", "r").read(),
                                   inout={(14,): [("00000000000000","11111111111111")] })


    print("Testing qcase_SWAP")
    program_tester.run()

    print("Testing qcase_CNOT")
    program_tester2.run()

    print("Testing pairs")
    program_tester3.run()

    print("Testing boolean semantics")
    program_tester4.run()

