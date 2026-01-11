from pfoqcompiler.testing import ProgramTester
from pfoqcompiler import __examples_directory__
import os

os.chdir(__examples_directory__)

print("Testing " + (namefile := "basic.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(10,): []}).run()

print("Testing " + (namefile := "Bell_CNOT.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(2,): []}).run()

print("Testing " + (namefile := "Bell_qcase.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(2,): []}).run()

print("Testing " + (namefile := "boolean_semantics.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(14,): [("00000000000000","11111111111111")] }).run()

print("Testing " + (namefile := "cat_state_parallel.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (15,): [], (16,): []}).run()

print("Testing " + (namefile := "cat_state.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (15,): [], (16,): []}).run()

print("Testing " + (namefile := "CNOT1.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(2,): [("00", "00"), ("01", "11"), ("01", "11"), ("10", "10")]}).run()

print("Testing " + (namefile := "CNOT2.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(2,): [("00", "00"), ("01", "11"), ("01", "11"), ("10", "10")]}).run()

print("Testing " + (namefile := "chained-substring.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (15,): [], (16,): []}).run()

print("Testing " + (namefile := "chained-substring2.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (15,): [], (16,): []}).run()

print("Testing " + (namefile := "count.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1, 1): [], (2, 2): [], (15, 4): [], (16, 5): []}).run()

print("Testing " + (namefile := "CSWAP.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(3,): [("000", "000"), ("010", "010"), ("100", "100"), ("110", "110"),
                                                                ("001", "001"), ("011", "101"), ("101", "011"), ("111", "111")]}).run()

print("Testing " + (namefile := "empty.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [('0', '0'), ('1', '1')]}).run()

print("Testing " + (namefile := "error_example.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(4,): []}).run()

print("Testing " + (namefile := "example_recursion.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(4,): [], (1,): [], (15,): []}).run()

print("Testing " + (namefile := "example_register.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(5,): [('00000', '10000')]}).run()

print("Testing " + (namefile := "full_adder-basic.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(6,): [('001010', '011010')]}).run()

print("Testing " + (namefile := "half_register_test.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(2,): [('00', '01'), ('11', '10')]}).run()

print("Testing " + (namefile := "if_example.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(4,): [("0000","1110")] }).run()

print("Testing " + (namefile := "integer_input.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(3,): [("000","000")], (5,): [("00000","10000")], (6,): [("000000","010000")]}).run()

print("Testing " + (namefile := "integer.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(5,): [("00000","00001")], (6,): [("000000","000010")]}).run()

print("Testing " + (namefile := "integer2.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [("0","1")], (6,): [("000000","100000")]}).run()

print("Testing " + (namefile := "merging_with_swaps.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (15,): [], (16,): []}).run()

print("Testing " + (namefile := "multiple_inputs.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1, 0): [], (2, 0): [], (15, 0): [], (16, 0): []}).run()

print("Testing " + (namefile := "nested_qcase.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(4,): [] }).run()

print("Testing " + (namefile := "not_and_H.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(4, 0): [] }).run()

print("Testing " + (namefile := "not_layer.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(3,): [("000","111")], (5,): [("00000","11111")], (6,): [("000000","111111")]}).run()

print("Testing " + (namefile := "other_gate.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(2,): [] }).run()

print("Testing " + (namefile := "pairs_error.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(5,): [("00110","00111")], (5,): [("011000","01100")]}).run()

print("Testing " + (namefile := "pairs.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(5,): [("00110","00111")], (5,): [("011000","01100")]}).run()

print("Testing " + (namefile := "palindrome.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(4,1): [("00110","10110"), ("00010","00010")], (5,1): [("011011","111011"),],
                                                         (0,1): [    ("0",    "1")], (1,1): [("00", "10")]}).run()

print("Testing " + (namefile := "procedure_mutual_recursive.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (15,): [], (16,): []}).run()

print("Testing " + (namefile := "procedure_recursive.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (15,): [], (16,): []}).run()

print("Testing " + (namefile := "procedure.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (15,): [], (16,): []}).run()

print("Testing " + (namefile := "qcase_CNOT.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(3,): [("000", "011"), ("010", "001"), ("100", "111"), ("110", "101"),
                                                                ("001", "000"), ("011", "010"), ("101", "100"), ("111", "110")]}).run()

print("Testing " + (namefile := "qcase_SWAP.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(4,): [("1010", "1001"), ("1110", "0111")] }).run()

print("Testing " + (namefile := "QFT-basic.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (15,): [], (16,): []}).run()

print("Testing " + (namefile := "QFT.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (15,): [], (16,): []}).run()

print("Testing " + (namefile := "QRCA.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,1,2): [], (2,2,3): [], (15,15,16): [], (16,16,17): []}).run()

print("Testing " + (namefile := "rec.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (15,): [], (16,): []}).run()

print("Testing " + (namefile := "search.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(4,1): [("01100","11100"), ("00000","00000")]}).run()

print("Testing " + (namefile := "skip.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [('0', '0'), ('1', '1')]}).run()

print("Testing " + (namefile := "sqlog.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1, 1): [], (2, 2): [], (15, 4): [], (16, 5): []}).run()

print("Testing " + (namefile := "sum_three.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (15,): [], (16,): []}).run()

print("Testing " + (namefile := "width_two.pfoq"))
ProgramTester(program=open(namefile, "r").read(), inout={(1,): [], (2,): [], (15,): [], (16,): []}).run()