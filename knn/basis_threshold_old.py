from qiskit.providers.aer import AerSimulator, AerError
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.result import Counts
from qiskit.visualization import plot_histogram

from qiskit.circuit.library import MCXGate

import sys
sys.path.append('..')

import math
from utility.quantum_encoding.basis_encoding import *
import matplotlib
matplotlib.use('tkagg') 
import matplotlib.pyplot as plt




def _registers_switcher(circuit, value, qubit_index): #TODO: duplicare register
    bin_str_pattern = '{:0%sb}' % len(qubit_index)
    value = bin_str_pattern.format(value)[::-1]  
    for idx, bit in enumerate(value):
        if int(bit): #TODO: qui avevo un "if not int(..)" perchè quel not?"
            circuit.x(qubit_index[idx])

def _get_binary_value(value, encoding_length): #refactor uniformare con il _registers_switcher
    bin_str_pattern = '{:0%sb}' % encoding_length
    value = bin_str_pattern.format(value)
    return value

# Return the gate which compute the Hamming Distance        
def get_flipped_hamming_distance_gate(N, name="hamming_distance"): #TODO: già definito -> refactor
    v = QuantumRegister(N, name='v')
    x = QuantumRegister(N, name='x')
    qc = QuantumCircuit(v, x, name=name)
    
    for v_i, x_i in zip(v, x): 
        qc.cx(v_i, x_i)  #XOR(V,X)   
    
    qc.x(x)

    return qc.to_gate()


def inCk(N, name='inCK'):
    d_i = QuantumRegister(1, 'd_i')
    a = QuantumRegister(N, 'a')
    one = QuantumRegister(1, 'one')
    qc = QuantumCircuit(a,one, name=name)


    for i in range(len(a)-1):
        qc.cx(one, a[i]) 

        for p in reversed(range(i)):
            qc.x(a[p])

        qc.append(MCXGate(num_ctrl_qubits=len(a[:i])+1), a[:i+1]+one[0:])

        for p in reversed(range(i)):
            qc.x(a[p])

    qc.cx(one, a[-1])
    qc.x(one)

    for p in range(len(a)-1):
        qc.x(a[p])

    qc.append(MCXGate(num_ctrl_qubits=len(a)-1), a[:-1]+one[0:])

    for p in range(len(a)-1):
        qc.x(a[p])
    
    #print(qc.draw())

    return qc.to_gate().control(1)



#TODO: refactor also in the other files
#Returns the gate encoding a binary_value
def _get_binary_value_gate(binary_value, N):
    if len(binary_value) != N:
        raise Exception("len bin(test) {}, while N is {}".format(len(binary_value), N)) 

    b = QuantumRegister(N, name='b')
    qc = QuantumCircuit(b, name=str(int(binary_value,2)))

    for i in range(len(binary_value)):
        if binary_value[i] == '1':
            qc.x(b[len(b)-1-i])

    return qc.to_gate()


def or_gate(n):
    a = QuantumRegister(n, 'a')
    v = QuantumRegister(n-1, 'v')
    qc = QuantumCircuit(a,v, name='OR')

    qc.x(a)
    qc.ccx(a[0], a[1], v[0])
    for i in range(2,len(a)):
        qc.ccx(a[i],v[i-2],v[i-1])

    qc.x(v[-1])

    return qc.to_gate()



class BasisRuan:

    def __init__(self, precision=3, threshold=3): #TODO Non mi piace che gli setto io la precision
        self.N = precision #binary value lenght
        self.n = int(math.ceil(math.log2(self.N+1))+1) #minimum number of quibits to represent the sum of Hamming Distance

        if threshold > math.pow(2, self.N)-1: # maximum threshold (otherwise wasting of qubits)
            self.t = math.pow(2, self.N)-1 
        else: 
            self.t = threshold

        self.l = int(math.pow(2, math.ceil(math.log2(self.N))) - self.N)  #2^k - N. where 2^(k-1) <= N <= 2^k. More details in the paper

        self.one = None
        self.v = None
        self.c = None 
        self.bcl = None # binary class lenght
        self.x = None #qubits for test

        self.a = None 
        
    def _init_circuit(self):

        self.v = QuantumRegister(self.N, name='v')  #encode training vectors
        self.c = QuantumRegister(self.bcl, name='c') #TODO: occhio remapping delle classi   #store class of training
        self.x = QuantumRegister(self.N, name='x') #encode test vector

        self.one = QuantumRegister(1, name='one')

        #Basis Encoding
        self.u = QuantumRegister(2, 'u')

        self.a = QuantumRegister(self.N, name='a') 
        
        self.cj = ClassicalRegister(1, name='cj')
        self.cv = ClassicalRegister(self.N, name='cv')
        self.cc = ClassicalRegister(self.bcl, name='cc') #TODO: set size class
        self.cx = ClassicalRegister(self.N, name='cx')

        try:
            self.simulator = AerSimulator(method='statevector', shots=8192, device='GPU')
        except AerErrorr as e:
            raise Exception('Simulator'+str(e))

        self.circuit = QuantumCircuit(self.v, self.c, self.u, self.x, self.a, self.one, self.cx, self.cv, self.cc, self.cj)


    def fit(self, X, y):
        self.circuit = None
        self._init_circuit()
        if len(X) != len(y):
            raise Exception("size of X different from size of y")

        #Append binary encoding of y in X
        Xy = []
        for x_i, y_i in zip(X,y):
            print(y_i+x_i)
            print(y_i, x_i)
            Xy.append(y_i+x_i)
        
        self.circuit.append(basis_encode_dataset(Xy, self.N+self.bcl),  self.v[0:]+self.c[0:]+self.u[0:]) #TODO: non mi piace self.N+self.bcl



    def predict(self, test):
        if self.circuit == None:
            raise Exception("Circuit not initialized")
        self.circuit.append(_get_binary_value_gate(test, self.N), self.x[0:])
        self.circuit.append(get_flipped_hamming_distance_gate(self.N), self.v[0:]+self.x[0:]) #TODO: rename


        bin_l_r = _get_binary_value(self.l+self.t, self.N)

        self.circuit.barrier()
        self.circuit.append(_get_binary_value_gate(bin_l_r, self.N), self.a[0:])
        self.circuit.barrier()

        self.circuit.x(self.one) 

        for i in range(len(self.x)):
            self.circuit.append(inCk(self.N), [self.x[i]]+self.a[0:]+self.one[0:])
            self.circuit.barrier()

        msq = int(math.ceil(math.log2(self.t)))
        print('msg '+str(msq))
        j = QuantumRegister(msq-1, 'j') #rename dentro lo chiamo v
        self.circuit.add_register(j)
        self.circuit.append(or_gate(msq), self.a[-msq:]+j[0:]) #TODO: uncomment

        self.circuit.measure(j, self.cj)
        self.circuit.measure(self.c, self.cc)
        #self.circuit.measure(self.v, self.cv)
        #self.circuit.measure(self.x, self.cx)

        result = execute(self.circuit, self.simulator).result()
        counts = result.get_counts(self.circuit)

        print(self.circuit.draw())
        #postselection = dict(post_select(counts))
        #plot_histogram(counts)
        print(counts)
        post_select = lambda counts: [(state, occurences) for state, occurences in counts.items() if state[0] == '1']
        postselection = Counts(dict(post_select(counts)))
        print(postselection)
        print(postselection.most_frequent()[2]) #most frequent class



    


br = BasisRuan(precision=4, threshold=3) #error if thrshold = 1 #TODO: può essere la threshold maggiore della precisione? Non ha molto senso
#evitare di scrivere la precision

X = ['1011','1101','1010'] #non salta #TODO: come le encoda nel basis encoding? nota che ho cambiato il register switcher if NOT è andato via
y = ['0', '1', '0']
test = '0110'

br.fit(X, y)
br.predict(test)

#TODO: dividere per step come nell'articolo
#TODO: lui riusa i qubit su x
