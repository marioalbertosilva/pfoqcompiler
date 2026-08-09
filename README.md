## Installation

### Installing Python 3.10 on a Ubuntu machine with Python < 3.10:

```console
machine@user:~$ sudo apt install software-properties-common -y
machine@user:~$ sudo add-apt-repository ppa:deadsnakes/ppa
machine@user:~$ sudo apt install python3.10
machine@user:~$ curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
```
If you needed these steps to upgrade to Python 3.10, do **NOT** try under any circumstances to play with the different Python versions now installed on your machine, you **will** break it.
The previously installed Python 2 and Python 3 should still be the default and everything should still work. 

The newly installed Python can be called and accessed via the new `python3.10` command.

### Creating a new Python 3.10 environment:

```console
machine@user:~$ python3.10 -m virtualenv venv
```
`venv` can be replaced with any name you like. An eponymous directory has been created which will handle the environment.

The environment must be activated every time you use the package:

```console
machine@user:~$ source venv/bin/activate
(venv) machine@user:~$ python
Python 3.10.16 (main, Dec  4 2024, 08:53:37) [GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```

### Installing the package:

First, you must clone the Github repository (remove the `-b Zenodo` if you widh to install the latest version):

```console
machine@user:~$ git clone -b Zenodo https://github.com/marioalbertosilva/pfoqcompiler.git
```

Then, install the `pfoqcompiler` package as an editable package. Make sure you activated the virtual environment beforehand:

```console
(venv) machine@user:~$ pip install -e .
```

This should install the `pfoqcompiler` package, along with all the package listed as dependencies in the `pyproject.toml` file and their respective dependencies.

Any use of the library, being for its development or intended use, must now be imported with:

```python
from pfoqcompiler.compiler import *
```

And not:

```python
from compiler import *
```


## Examples

### Basic programs

We provide here instructions to compile different basic programs as examples. The compiler can be called either from command line or a Python program.

The program PAIRS can be compiled for input size 11 with the following command
```console
machine@user:~/pfoqcompiler$ python pfoqcompiler/compiler.py -f pfoqcompiler/examples/pairs.foq -i 11
```
and should output the circuit:
![Program PAIRS on input size 11](/doc/source/frontpage/pairs_11.png "Program PAIRS on input size 11")

The same circuit is compiled with the following Python statement:

```{python}
compiler = PfoqCompiler(filename="pfoqcompiler/examples/pairs.foq",  # Instantiate compilation task
                        nb_qubits=[11])
compiler.compile() # Parse and compile the program
circuit = compiler.compiled_circuit # Output can be accessed as a qiskit.QuantumCircuit
compiler.display()  # Output can be displayed
compiler.save("pairs.pdf") # Or saved to a file
```

The program QFT on 4 qubits can be obtained with the command

```console
machine@user:~/pfoqcompiler$ python pfoqcompiler/compiler.py -f pfoqcompiler/examples/qft_unif.foq -i 4
```
with the circuit
![Program QFT on input size 4](/doc/source/frontpage/QFT_4.png "Program QFT on input size 4")

The Full Adder program can be compiled for the case of 4-qubit input size with the command

```console
machine@user:~/pfoqcompiler$ python pfoqcompiler/compiler.py -f pfoqcompiler/examples/qrca.foq -i 4 4 5
```
and results in the circuit
![Program Full Adder on input size 13](/doc/source/frontpage/full-adder_4.png "Program Full Adder on input size 13")

The Chained Substring example for $k=2$ can be compiled on input size 10 with the command

```console
machine@user:~/pfoqcompiler$ python pfoqcompiler/compiler.py -f pfoqcompiler/examples/chained-substring.foq -i 10
```
with the circuit
![Program for chained substring (k=2) on input size 10](/doc/source/frontpage/chained-substring_10.png "Program for chained substring (k=2) on input size 10")

A more readable pdf version can be found ![here](/doc/source/frontpage/chained-substring.pdf "here").

The Sum(r) example for r=3 compiled for 6 input qubits is done with the command

```console
machine@user:~/pfoqcompiler$ python pfoqcompiler/compiler.py -f pfoqcompiler/examples/sum_three.foq -i 6
```
with the circuit
![Program Sum(r=3) on input size 6](/doc/source/frontpage/sum_three_6.png "Program Sum(r=3) on input size 6")

#### Running without optimization

The examples can also be easily run without any optimization (i.e. we ignore the width condition and always perform the first case of the procedure compilation rule). This is triggered with the option ```--no-optimize```. For instance, program PAIRS on input size 11 with no optimization results in the following circuit.

```console
machine@user:~/pfoqcompiler$ python pfoqcompiler/compiler.py -f pfoqcompiler/examples/pairs.foq -i 11 --no-optimize
```
![Program PAIRS on input size 11 (no optimization)](/doc/source/frontpage/pairs_no-optimize.png "Program PAIRS on input size 11 (no optimization)")


## Testings
Our tests can be run by running the `run_tests.sh` script. These unit tests checks the correct parsing and compiling of various FOQ programs.

As part of these tests, a very basic program testing class has been created to ensure compiled programs behave correctly. Programs mapping computational basis states to computational basis states can be tested like this:

```{python}
ProgramTester(program=open("pfoqcompiler/examples/qrca", "r").read(),
              inout={(1,1,2): [],
                     (2,2,3): [("0001010", "0101010"), ("0000110", "1100110"), ("0000101", "0010101")],
                     (15,15,16): [],
                     (16,16,17): []}).run()
```

The `inout` parameter is a dictionary mapping a sequence of register sizes to a possibly empty list of test cases. The program will be compiled for each given sequence of register size, even when no test cases are given.

Each test case is a pair of computational basis states written as binary strings, the left one being the input and the right one the expected output of the program. If you which to test the compiled circuit for more general input/output, you will have to simulate the statevector yourself with the usual qiskit approach by using the `compiled_circuit` property of a compiled `PfoqCompiler`.

Note that the encoding used in the inout parameter is the qiskit encoding: the rightmost bit corresponds to the most-significant qubit of the first register and the leftmost bit corresponds to the least-significant qubit of the last register. Here, we are testing if the `"qrca.foq"` program correclty sums two-qubit registers by testing that the output for input 1 $(01_{2})$ and 1 $(01_{2})$ is indeed 2 $(010_{2})$, that the output for input 1 $(01_{2})$ and 2 $(10_{2})$ is indeed 3 $(011_{2})$ and that the output for input 2 $(10_{2})$ and 2 $(10_{2})$ is indeed 4 $(100_{2})$.

## Documentation
The documentation can be automatically generated with the following command:

```{console}
machine@user:~$ cd pfoqcompiler/doc
machine@user:~$ make html
machine@user:~$ firefox build/html/index.html
```


