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

First, you must clone the Github repository (remove the `-b Zenodo` if you wish to install the latest version):

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


## Examples (Section 3)

### Bell-state creation with qcase

```console
machine@user:~/pfoqcompiler$ python pfoqcompiler/compiler.py -f "pfoqcompiler/examples/Bell_qcase.foq" -i 2
```
![Program Bell_qcase.foq on input size 12](/doc/source/frontpage/Bell_qcase_2.pdf "Program Bell_qcase on input size 2")

### Bell-state creation with CNOT

```console
machine@user:~/pfoqcompiler$ python pfoqcompiler/compiler.py -f "pfoqcompiler/examples/Bell_qcase.foq" -i 2
```
(results in the same circuit)

### GHZ state creation (linear)

```console
machine@user:~/pfoqcompiler$ python pfoqcompiler/compiler.py -f "pfoqcompiler/examples/ghz.foq" -i 6
```
![Program ghz.foq on input size 6](/doc/source/frontpage/ghz.pdf "Program GHZ on input size 6")

### GHZ state creation (parallel)

```console
machine@user:~/pfoqcompiler$ python pfoqcompiler/compiler.py -f "pfoqcompiler/examples/ghz_par.foq" -i 8
```
![Program ghz_par.foq on input size 8](/doc/source/frontpage/ghz_par.pdf "Program GHZ_PAR on input size 8")

### Shifts

```console
machine@user:~/pfoqcompiler$ python pfoqcompiler/compiler.py -f "pfoqcompiler/examples/shifts.foq" -i 5 1
```
![Program shifts.foq on input size (5,1)](/doc/source/frontpage/shifts_5_1.pdf "Program SHIFTS on input size (5,1)")

### Quantum Fourier transform

```console
machine@user:~/pfoqcompiler$ python pfoqcompiler/compiler.py -f "pfoqcompiler/examples/qft.foq" -i 6
```
![Program qft.foq on input size 6](/doc/source/frontpage/qft_6.pdf "Program QFT on input size 6")

### Quantum ripple-carry adder

```console
machine@user:~/pfoqcompiler$ python pfoqcompiler/compiler.py -f "pfoqcompiler/examples/qrca.foq" -i 4 4 5
```
![Program qrca.foq on input size (4,4,5)](/doc/source/frontpage/qrca_4_4_5.pdf "Program QRCA on input size (4,4,5)")

### Pairs

```console
machine@user:~/pfoqcompiler$ python pfoqcompiler/compiler.py -f "pfoqcompiler/examples/pairs.foq" -i 11
```
![Program pairs.foq on input size 11](/doc/source/frontpage/pairs_11.pdf "Program PAIRS on input size 11")


### Binary search

```console
machine@user:~/pfoqcompiler$ python pfoqcompiler/compiler.py -f "pfoqcompiler/examples/search.foq" -i 6 1
```
![Program search.foq on input size (6,1)](/doc/source/frontpage/search_6_1.pdf "Program SEARCH on input size (6,1)")


#### Running without optimization

The `pfoqcompiler` also has a setting for compiling without anchoring-and-merging, essentially applying the sequential method of Figure 2. This is triggered with the option ```--no-optimize```. For instance, program PAIRS on input size 11 with no optimization results in the following circuit.

```console
machine@user:~/pfoqcompiler$ python pfoqcompiler/compiler.py -f pfoqcompiler/examples/pairs.foq -i 11 --no-optimize
```
![Program PAIRS on input size 11 with sequential method](/doc/source/frontpage/pairs_no-optimize.pdf "Program PAIRS on input size 11 with sequential method")


## Testing
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


