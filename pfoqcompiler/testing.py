from pfoqcompiler.compiler import PfoqCompiler

import matplotlib.pyplot as plt
from math import ceil, log2

import argparse

from qiskit.compiler import transpile
#size = number of qubits but what about constant time ?


import sys
# import time

# for i in range(10):
#     sys.stdout.write("\r{0}>".format("="*i))
#     sys.stdout.flush()
#     time.sleep(0.5)



def start_progress(title):
    global progress_x
    sys.stdout.write(title + ": [" + "-"*40 + "]" + chr(8)*41)
    sys.stdout.flush()
    progress_x = 0

def progress(x):
    global progress_x
    x = int(x * 40 // 100)
    sys.stdout.write("#" * (x - progress_x))
    sys.stdout.flush()
    progress_x = x

def end_progress():
    sys.stdout.write("#" * (40 - progress_x) + "]\n")
    sys.stdout.flush()


input_sizes = range(5, 100, 5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser( 
        description='Estimate size and depth of circuits obtained from PFOQ programs.')
    
    parser.add_argument('-f', '--filename', type=str, nargs="+",
                        help='PFOQ programs to run.')
    
    parser.add_argument('-r', '--range', type=int, nargs="+",
                        help='Range of input sizes: start end step. Default ')
    
    parser.add_argument('-d', '--display', type=bool,
                        help='Indicates if the circuit should be displayed with Matplotlib.',
                        default=True)
    
    parser.add_argument('--optimize', action=argparse.BooleanOptionalAction,
                        help='Indicates if procedure calls should be merged. Defaults to \'True\'.',
                        default=True)
    
    parser.add_argument('--write', action=argparse.BooleanOptionalAction,
                        help='Indicates if procedure calls should be merged. Defaults to \'True\'.',
                        default=True)
    
    # parser.add_argument('--barriers', action=argparse.BooleanOptionalAction,
    #                     help='Determines whether or not the circuit is displayed ' \
    #                     'sequentially according to the pfoq-compiler, or if parallel ' \
    #                     'gates are performed concurrently. Defaults to \'True\', where ' \
    #                     'sequential order is imposed for displaying purposes.',
    #                     default=True)

    args = parser.parse_args()

    if args.range:
        if len(args.range)>2:
            input_sizes = range(args.range[0], args.range[1], args.range[2])
        elif len(args.range)==2:
            input_sizes = range(args.range[0], args.range[1], 5)
        else:
            input_sizes = range(4, 100, 4)



FILENAMES = ["search.pfoq"]

if args.filename:
    filename = args.filename[0]
else:
    filename = FILENAMES[0]

N = len(input_sizes)


if args.write:
    print("computing circuits...")
    start_progress("progress")

    with open("resource_estimation/old_"+filename, "a") as f:

        for i in range(N):

            input = input_sizes[i]

            compiler = PfoqCompiler(filename="examples/"+filename,
                                    nb_qubits=[input],
                                    nb_ancillas = 3*input,
                                    old_optimize = False,
                                    optimize_flag=True,
                                    barriers=False)

            compiler.parse()    
            compiler.compile()
            
            qc = compiler._compiled_circuit

            f.write(f"{input} {qc.size()} {qc.depth()}\n")
            progress(int(i/N*100))
    
    end_progress()
    f.close()


with open("resource_estimation/old_"+filename, "r") as f:
    lines = [line.rstrip() for line in f]
f.close()

L = len(lines)


in_size, circuit_sizes, circuit_depths = [0]*L, [0]*L, [0]*L

for i in range(L):
    values = lines[i].split(" ")
    in_size[i], circuit_sizes[i], circuit_depths[i] = int(values[0]),  int(values[1]),  int(values[2])


#print("size:", circuit_sizes)
#print("depth:", circuit_depths)



plt.xticks(range(0,max(in_size)+100,300))
plt.xlabel("Input size")
plt.plot(in_size,circuit_sizes,'.')
plt.show()



plt.title("Circuit depth ")
plt.plot(in_size,circuit_depths,'.')
plt.xlabel("Input size")

plt.show()
