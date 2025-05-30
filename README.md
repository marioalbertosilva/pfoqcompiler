[Jump to FSCD2025 Examples section](#examples)

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

### Installing the package as an editable package:

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

### BASIC programs (FSCD2025 submission)

We provide here instructions to compile the different BASIC programs in the FSCD2025 submission, with the outcomes also given in the file *FSCD_2025*.

The compiler ```compiler.py``` can be found in the ```pfoqcompiler``` file.

The program PAIRS (Figure 1) can be compiled for input size 11 with the following command
```console
machine@user:~$ python compiler.py -f pairs.pfoq -i 11
```
and should output the circuit
![Program PAIRS on input size 11](/FSCD_circuits/pairs_11.png "Program PAIRS on input size 11")

The program QFT (Figure 5) on 4 qubits can be obtained with the command

```console
machine@user:~$ python compiler.py -f QFT-basic.pfoq -i 4
```
with the circuit
![Program QFT on input size 4](/FSCD_circuits/QFT_4.png "Program QFT on input size 4")

The Full Adder program (Example 14) can be compiled for the case of 13 qubits with the command

```console
machine@user:~$ python compiler.py -f full_adder-basic.pfoq -i 13
```
and results in the circuit
![Program Full Adder on input size 13](/FSCD_circuits/full-adder_13.png "Program Full Adder on input size 13")

The Chained Substring example (Example 15) for k=2 can be compiled on input size 10 with the command

```console
machine@user:~$ python compiler.py -f chained-substring.pfoq -i 10
```
with the circuit
![Program for chained substring (k=2) on input size 10](/FSCD_circuits/chained-substring_10.png "Program for chained substring (k=2) on input size 10")

A more readable pdf version can be found ![here](/FSCD_circuits/chained-substring_10.pdf "here").

The Sum(r) example (Example 16) for r=3 compiled for 6 input qubits is done with the command

```console
machine@user:~$ python compiler.py -f sum_three.pfoq -i 6
```
with the circuit
![Program Sum(r=3) on input size 6](/FSCD_circuits/sum_three_6.png "Program Sum(r=3) on input size 6")

#### Running without optimization

The examples can also be easily run without any optimization (basically as in the rules of **compile** (Figure 7) where we ignore the width condition and always perform the first case of the procedure compilation rule). This is triggered with the option ```--no-optimize```. For instance, program PAIRS on input size 11 with no optimization results in the following circuit.

```console
machine@user:~$ python compiler.py -f pairs.pfoq -i 11 --no-optimize
```
![Program PAIRS on input size 11 (no optimization)](/FSCD_circuits/pairs_no-optimize.png "Program PAIRS on input size 11 (no optimization)")




