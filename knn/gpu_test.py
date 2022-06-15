from qiskit.providers.aer import *
from qiskit import *
from qiskit.extensions import Initialize, UnitaryGate
from qiskit.circuit.library import *

N = 8
M = 2**N
D = []

#Binary Length
bin_str_pattern = '{:0%sb}' % N

#Filling the dataset
for i in range(M):
    binary_string = bin_str_pattern.format(i) #return the binary repr of i. 
    D.append(binary_string) 


x = QuantumRegister(N, name="x")
g = QuantumRegister(N-1, name="g") 
c = QuantumRegister(2, name="c")

c_bits = ClassicalRegister(N, name="classical")
qc = QuantumCircuit(x, g, c, c_bits)

def c0not(qc, control, target):
    qc.x(control)
    qc.cx(control, target)
    qc.x(control)
    
def custom_toffoli(qc, operand_0, operand_1, input_0, input_1, output):
    if not operand_0: #if operand_1 is 0, then "SELECT" input_0 / input_0 is 0
        qc.x(input_0)
    if not operand_1: #if operand_2 is 0, then "SELECT" input_1 / input_1 is 0
        qc.x(input_1)
        
    qc.ccx(input_0, input_1, output)
    
    if not operand_0: #reset
        qc.x(input_0)
    if not operand_1: #reset
        qc.x(input_1)
    
def mu_rotation(qc, mu):
    #Not really the same
    rotation_matrix = [ [((mu-1)/mu)**(1/2), 1/(mu**(1/2))],
                       [-1/(mu**(1/2)), ((mu-1)/mu)**(1/2)] ]
    u = UnitaryGate(rotation_matrix, label="rotation")
    qc1 = QuantumCircuit(1)
    qc1.append(u, [0])
    custom = qc1.to_gate().control(1)
    qc.append(custom, [c[0], c[1]])


for m in range(M): #iter over inputs
    for n in range(N):
        bit = int(D[m][n])
        if bit: #If equal 1, then C_0Not(c_1, n)
            c0not(qc, c[1], n)
            

    c0not(qc, c[1], c[0]) #Flip c_0 if c_1 == 0
    mu = M + 1 - (m + 1)
    mu_rotation(qc, mu)
    qc.barrier()
    
    custom_toffoli(qc, int(D[m][0]), int(D[m][1]), x[0], x[1], g[0])   #x[2], g[0], g[1]
    for n in range(2, N):
        qc.barrier()
        custom_toffoli(qc, int(D[m][n]), 1, x[n], g[n-2], g[n-1])
    
    qc.barrier()
    qc.cx(g[n-1], c[0])
    qc.barrier()

    for n in range(N-1, 1, -1):
        custom_toffoli(qc, int(D[m][n]), 1, x[n], g[n-2], g[n-1])
        qc.barrier()
    custom_toffoli(qc, int(D[m][0]), int(D[m][1]), x[0], x[1], g[0])   

    
    qc.barrier()
    for n in range(N):
        bit = int(D[m][n])
        if bit:
            c0not(qc, c[1], n)
    
    qc.barrier()
    
for n in range(N):
    qc.measure(n, n)
#
#simulator = AerSimulator(method='statevector', device='GPU')
#results = execute(qc,simulator, cuStateVec_enable=True).result()
simulator = AerSimulator(method='statevector', shots=8192, device='GPU', cuStateVec_enable=True)
#print(simulatok.available_devices())

# Execute and get counts
#result = simulator.run(qc,simulator,shots=8192,seed_simulator=12345).result()
result = execute(qc, simulator).result()
counts = result.get_counts(qc)
#print(counts)
print("End")

