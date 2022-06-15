from qiskit import *
from qiskit.providers.aer import *
from qiskit.circuit.library import *
from qiskit.extensions import Initialize, UnitaryGate
from qiskit.result import Counts
from qiskit.visualization import plot_histogram


import sys
sys.path.append('..')

import math
from utility.quantum_encoding.basis_encoding import *
import matplotlib

import random
#matplotlib.use('tkagg') 
#import matplotlib.pyplot as plt



def same_lenght(items):
    #Get maximum lenght of binary encodings in X
    binary_lenght_items = len(max(items, key=len))
    are_all_same_lengths_items = all(len(binary_value) == binary_lenght_items for binary_value in items)
    if not are_all_same_lengths_items:
        raise Exception('Binary values not of the same lenght')
    return binary_lenght_items

def registers_switcher(circuit, value, qubit_index): 
    bin_str_pattern = '{:0%sb}' % len(qubit_index)
    value = bin_str_pattern.format(value)[::-1]  
    for idx, bit in enumerate(value):
        if int(bit): #TODO: qui avevo un "if not int(..)" perchè quel not?"
            circuit.x(qubit_index[idx])

def get_binary_value(value, encoding_length): #refactor uniformare con il _registers_switcher
    bin_str_pattern = '{:0%sb}' % encoding_length
    value = bin_str_pattern.format(value)
    return value

# Return the gate which compute the Hamming Distance        
def get_hamming_distance_gate(binary_lenght, name="Hamming Distance"): #TODO: già definito -> refactor
    v = QuantumRegister(binary_lenght, name='v')
    x = QuantumRegister(binary_lenght, name='x')
    qc = QuantumCircuit(v, x, name=name)
    
    for v_i, x_i in zip(v, x): 
        qc.cx(v_i, x_i)  #XOR(V,X)   

    return qc.to_gate()


#TODO: refactor also in the other files
#Returns the gate encoding a binary_value
def get_binary_value_gate(binary_value, binary_lenght, name=''):
    if len(binary_value) != binary_lenght:
        raise Exception("len bin(test) {}, while binary_lenght is {}".format(len(binary_value), binary_lenght)) 

    b = QuantumRegister(binary_lenght, name='b')
    qc = QuantumCircuit(b, name=name+' '+str(int(binary_value, 2)))

    for i in range(len(binary_value)):
        if binary_value[i] == '1':
            qc.x(b[len(b)-1-i])

    return qc.to_gate()


def inCk(N, name='inCK'):
    d_i = QuantumRegister(1, 'd_i')
    a = QuantumRegister(N, 'a')
    one = QuantumRegister(1, 'one')
    qc = QuantumCircuit(a, one, name=name)

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
    
    return qc.to_gate().control(1)



class BasisRuan:

    def __init__(self, threshold=3):
        self.t = threshold


    def _init_circuit(self, X, y):
        if len(X) != len(y):
            raise Exception("Size of X different from size of y")

        binary_lenght_X = same_lenght(X)
        binary_lenght_y = same_lenght(y)

        if self.t > binary_lenght_X: #avoids to allocate to many qubits if threshold >> then the maximum hamming_distance
            self.t = binary_lenght_X 
        if self.t < 0:
            raise Exception('Threshold cannot be a negative')
        
        self.l = int(math.pow(2, math.ceil(math.log2(binary_lenght_X)) - binary_lenght_X))  #l = 2^k - len(x_i). where 2^(k-1) <= N <= 2^k. More details in the paper (Ruan2017)

        #Quantum Registers
        self.v = QuantumRegister(binary_lenght_X, name='v')  #encode training vectors
        self.c = QuantumRegister(binary_lenght_y, name='c') #TODO: occhio remapping delle classi   #store class of training
        self.x = QuantumRegister(len(self.v), name='x') #encode test vector

        size_a =  len(bin(self.l+self.t+binary_lenght_X).replace("0b", "")) #Maximum size of value in register a
        self.a = QuantumRegister(size_a, name='a') #store the sum between Hamming Distance and the threshold  #a = l + t

        #Basis Encoding ancillary qubit
        self.u = QuantumRegister(2, 'u')
        
        #Classical Registers
        self.cv = ClassicalRegister(len(self.v), name='cv')
        self.cc = ClassicalRegister(len(self.c), name='cc')
        self.cx = ClassicalRegister(len(self.x), name='cx')
        self.c_overflow = ClassicalRegister(1, name='c_overflow')

        try:
            self.simulator = AerSimulator(method='statevector', shots=8192, device='GPU', cuStateVec_enable=True)
        except AerError as e:
            raise Exception('Simulator'+str(e))

        self.circuit = QuantumCircuit(self.v, self.c, self.u, self.x, self.a, self.cc, self.c_overflow)
        #self.circuit = QuantumCircuit(self.v, self.c, self.u, self.x, self.a, self.cx, self.cv, self.cc, self.c_overflow)


    def fit(self, X, y):
        self.circuit = None
        self._init_circuit(X,y)

        #Append binary encoding of y in X
        Xy = []
        for x_i, y_i in zip(X,y):
            Xy.append(y_i+x_i)
        
        #Basis encode dataset in the circuit
        self.circuit.append(basis_encode_dataset(Xy, len(self.v)+len(self.c)), self.v[0:]+self.c[0:]+self.u[0:]) 



    def predict(self, test):
        if self.circuit == None:
            raise Exception("Circuit not initialized")

        if len(test) != len(self.x):
            raise Exception("Binary encoding lenght of the test mismatch the lenght of training elements")

        
        #------------- STEP 1 ----------------
        #Encode test vector
        self.circuit.append(get_binary_value_gate(test, len(self.x), 'test:'), self.x[0:])
        self.circuit.barrier()
        #------------------------------------


        #------------- STEP 2 ----------------
        #Compute Hamming Distance between v and x
        self.circuit.append(get_hamming_distance_gate(len(self.v)), self.v[0:]+self.x[0:]) 

        #Flip hamming distance
        self.circuit.x(self.x)
        self.circuit.barrier()
        #------------------------------------


        #------------- STEP 3 ----------------
        value_a = get_binary_value(self.l+self.t, len(self.a))
        self.circuit.append(get_binary_value_gate(value_a, len(self.a), 'a:'), self.a[0:])
        self.circuit.barrier()
        #------------------------------------


        #------------- STEP 4 ----------------
        #Append Increment Circuit  
        one = QuantumRegister(1, 'one')
        self.circuit.add_register(one) 
        self.circuit.x(one) 
        for i in range(len(self.x)):
            self.circuit.append(inCk(len(self.a)), [self.x[i]]+self.a[0:]+one[0:])
        self.circuit.barrier()
        #------------------------------------


        #------------- STEP 5 ----------------
        #Append or_gate over register a
        most_significant_qubits =  len(bin(self.t).replace("0b", "")) #Maximum size of value in register a
        or_result = QuantumRegister(1, 'or_result') #ancillary qubits to compute or gate over register a
        self.circuit.add_register(or_result)
        self.circuit.append(OR(num_variable_qubits=most_significant_qubits), self.a[-most_significant_qubits:]+[or_result[0]])
        self.circuit.barrier()
        #------------------------------------



        #------------- Measurements ----------------
        self.circuit.measure(or_result, self.c_overflow)
        self.circuit.measure(self.c, self.cc)
        #self.circuit.measure(self.v, self.cv)
        #self.circuit.measure(self.x, self.cx)

        result = execute(self.circuit, self.simulator).result()
        counts = result.get_counts(self.circuit)
        print(self.circuit.draw())
        print(counts)
        #------------------------------------


        #------------- STEP 6 ----------------
        #'Post_selection: if training is below the threshold, then self.c_overflow is marked with "1"
        post_select = lambda counts: [(state, occurences) for state, occurences in counts.items() if state[0] == '1']
        postselection = Counts(dict(post_select(counts)))
        print(postselection)

        #Return the class with highest amplitude or -1 if no trainigns within the threshold
        if postselection: 
            prediction = postselection.most_frequent()[2:2+len(self.cc)] 
        else: 
            prediction = -1
        print(prediction) 
        return prediction
        #------------------------------------



'''
threshold = 0 : exact match
'''
    
br = BasisRuan(threshold=0)


X = ['1010','0100','0110']
y = ['00', '11', '10']


def randbingen(bin_len, N):
    dataset = []
    el = ''
    for i in range(N):
        for i in range(bin_len):
            x = random.randint(0,1)
            el += str(x)
        dataset.append(el) 
        el = ''
    return dataset

X = randbingen(4, 100)
y = randbingen(2, 100)
print(X)
print(y)

test = '0110'

br.fit(X, y)
br.predict(test)
