from pfoqcompiler.testing import ProgramTester
from lark.exceptions import UnexpectedCharacters
from pfoqcompiler import __examples_directory__
from numpy._core._exceptions import _ArrayMemoryError
import os

os.chdir(__examples_directory__)

print("Testing " + (namefile := "basic.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(10,): []}).run()

print("Testing " + (namefile := "Bell_CNOT.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(2,): []}).run()

print("Testing " + (namefile := "Bell_qcase.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(2,): []}).run()

print("Testing " + (namefile := "boolean_expression.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(2,): []}).run()

print("Testing " + (namefile := "boolean_semantics.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(14,): [("00000000000000", "11111111111111")] }).run()

print("Testing " + (namefile := "cat_state_parallel.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (15,): [], (16,): []}).run()

print("Testing " + (namefile := "cat_state.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [],
                                                         (2,): [],
                                                         (15,): [], (16,): []}).run()

print("Testing " + (namefile := "CNOT1.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(2,): [("00", "00"), ("01", "11"), ("01", "11"), ("10", "10")]}).run()

print("Testing " + (namefile := "CNOT2.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(2,): [("00", "00"), ("01", "11"), ("01", "11"), ("10", "10")]}).run()

print("Testing " + (namefile := "chained-substring.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [],
                                                         (4,): [("0000", "0000"), ("0001", "0001"), ("0100", "1100")],
                                                         (5,): [("00000", "00000"), ("00001", "00001"), ("01000", "11000")]}).run()

print("Testing " + (namefile := "chained-substring2.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (15,): [], (16,): []}).run()

print("Testing " + (namefile := "count.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1, 1): [], (2, 2): [], (15, 4): [], (16, 5): []}).run()

print("Testing " + (namefile := "CSWAP.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(3,): [("000", "000"), ("010", "010"), ("100", "100"), ("110", "110"),
                                                                ("001", "001"), ("011", "101"), ("101", "011"), ("111", "111")]}).run()

print("Testing " + (namefile := "empty.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [('0', '0'), ('1', '1')]}).run()
ProgramTester(program=open(namefile, "r").read(), inout={(64,): [('0'*64, '0'*64)]}, expected_error=ValueError, expected_error_stage="runtime").run()

print("Testing " + (namefile := "error_example.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(4,): []}, expected_error=IndexError, expected_error_stage="compilation").run()

print("Testing " + (namefile := "example_recursion.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(4,): [], (1,): [], (15,): []}).run()

print("Testing " + (namefile := "example_register.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(5,): [('00000', '00000')]}).run()

print("Testing " + (namefile := "half_register_test.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(6,): [('000000', '001000'), ('001111', '000111')]}).run()

print("Testing " + (namefile := "if_example.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(4,): [("0000","1110")] }).run()

print("Testing " + (namefile := "integer_input.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(4,): [("0000","0001")]}).run()
ProgramTester(program=open(namefile, "r").read(), inout={(3,): []}, expected_error=IndexError, expected_error_stage="compilation").run()

print("Testing " + (namefile := "integer.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(5,): [("00000","00001")], (6,): [("000000","000010")]}).run()

print("Testing " + (namefile := "integer2.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [("0","1")], (6,): [("000000","100000")]}).run()

print("Testing " + (namefile := "merging_with_swaps.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (15,): [], (16,): []}).run()

print("Testing " + (namefile := "multiple_classical_inputs.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(2,): [], (14,): []}).run()

print("Testing " + (namefile := "multiple_inputs.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1, 0): [], (2, 0): [], (15, 0): [], (16, 0): []}).run()

print("Testing " + (namefile := "nested_qcase.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(4,): [] }).run()

print("Testing " + (namefile := "not_and_H.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(4, 0): [] }).run()

print("Testing " + (namefile := "not_layer.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(3,): [("000","111")], (5,): [("00000","11111")], (6,): [("000000","111111")]}).run()

print("Testing " + (namefile := "other_gate.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(2,): [] }).run()

print("Testing " + (namefile := "pairs_error.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(5,): []}, expected_error=IndexError, expected_error_stage="compilation").run()

print("Testing " + (namefile := "pairs.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(5,): [("00110","00111")], (5,): [("011000","011000")]}).run()

print("Testing " + (namefile := "palindrome.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(4,1): [("00110","10110"), ("00010","00010")],
                                                         (5,1): [("011011","111011"),],
                                                         (1,1): [("00", "10")]}).run()

print("Testing " + (namefile := "parsing_error.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): []}, expected_error=UnexpectedCharacters, expected_error_stage="parsing").run()

print("Testing " + (namefile := "procedure_mutual_recursive.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (15,): [], (16,): []}).run()

print("Testing " + (namefile := "procedure_recursive.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (15,): [], (16,): []}).run()

print("Testing " + (namefile := "procedure.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (15,): [], (16,): []}).run()

print("Testing " + (namefile := "qcase_CNOT.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(3,): [("000", "011"), ("010", "001"), ("100", "111"), ("110", "101"),
                                                                ("001", "000"), ("011", "010"), ("101", "100"), ("111", "110")]}).run()

print("Testing " + (namefile := "qcase_SWAP.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(4,): [("1010", "1001"), ("1110", "0111")] }).run()

print("Testing " + (namefile := "qft_basic.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (15,): [], (16,): []}).run()

print("Testing " + (namefile := "qft.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (15,): [], (16,): []}).run()

print("Testing " + (namefile := "qrca.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,1,2): [], (2,2,3): [], (15,15,16): [], (16,16,17): []}).run()

print("Testing " + (namefile := "rec.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (15,): [], (16,): []}).run()

print("Testing " + (namefile := "search.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(4,1): [("01100","11100"), ("00000","00000")]}).run()

print("Testing " + (namefile := "skip.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [('0', '0'), ('1', '1')]}).run()

print("Testing " + (namefile := "sum_three.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (15,): [], (16,): []}).run()

print("Testing " + (namefile := "width_two.foq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (14,): []}).run()