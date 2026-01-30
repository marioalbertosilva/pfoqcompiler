from pfoqcompiler.compiler import PfoqCompiler
import sys



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

circuit_sizes, circuit_depths = {}, {}

max_input = 0

for suffix in [""]:

    with open("resource_estimation/sum_three" + suffix + ".pfoq", "r") as f:
        lines = [line.rstrip() for line in f]
    f.close()


    L = len(lines)

    for i in range(L):

        values = lines[i].split(" ")

        input_size = int(values[0])

        if input_size > max_input:
            max_input = input_size

        circuit_sizes[input_size] = int(values[1])

        circuit_depths[input_size] = int(values[2]) 

print("max input:", max_input)

start_progress("progress")


with open("resource_estimation/sum_three_compileplus.dat", "w") as f:
    for i in range(max_input+1):
        if i in circuit_sizes:
            f.write(f"{i} {circuit_sizes[i]} {circuit_depths[i]}\n")

        else:
            compiler = PfoqCompiler(filename="examples/sum_three.pfoq",
                                    nb_qubits=[i],
                                    nb_ancillas = 3*i,
                                    optimize_flag=True,
                                    barriers=False)
            compiler.parse()    
            compiler.compile()
            qc = compiler._compiled_circuit

            f.write(f"{i} {qc.size()} {qc.depth()}\n")

        progress(int(i/max_input*100))
    
end_progress()
f.close()