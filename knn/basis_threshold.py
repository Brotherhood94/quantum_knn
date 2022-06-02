from qiskit.providers.aer import AerSimulator, AerError
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.visualization import plot_histogram

from qiskit.circuit.library import MCXGate

import sys
sys.path.append('..')

import math
from utility.quantum_encoding.basis_encoding import *
import matplotlib.pyplot as plt


def _count_1s(qc, x, x_index, h, h_index, c1):
    qc.ccx(x[x_index], c1, h[h_index])
    
    for i in range(h_index):
        qc.x(h[i])
    
    qc.mct([x[x_index]]+h[:h_index+1], c1)

    for i in range(h_index):
        qc.x(h[i])


def _registers_switcher(circuit, value, qubit_index): #TODO: duplicare register
    bin_str_pattern = '{:0%sb}' % len(qubit_index)
    value = bin_str_pattern.format(value)#[::-1]  #Se tolgo, quelli più alti sono i meno significativi : check
    print(value)
    for idx, bit in enumerate(value):
        if not int(bit):
            circuit.x(qubit_index[idx])

    #print(circuit.draw()) #TODO: check encoding corretto

# Return the gate which compute the Hamming Distance        
def get_hamming_distance_gate(N, name="hamming_distance"): #TODO: già definito -> refactor
    n = int(math.ceil(math.log2(N+1))) # minimum number of qubits to represent the sum of Hamming Distance's 1s
    
    v = QuantumRegister(N, name='v')
    x = QuantumRegister(N, name='x')
    h = QuantumRegister(n+1, name='h')
    c1 = QuantumRegister(1, name='c1')
    
    qc = QuantumCircuit(v, x, h, c1, name=name)
    
    qc.x(c1) #Setting to 1
    
    for v_i, x_i in zip(v, x): 
        qc.cx(v_i, x_i)  #XOR(V,X)   
    
    qc.x(x)
    '''
    for x_index in range(len(x)):   #Sum 1s of XOR(V,X)
        for h_index in range(len(h)):
            _count_1s(qc, x, x_index, h, h_index, c1)
        qc.ccx(x[x_index], c1, h[len(h)-1])
        qc.cx(x[x_index],c1)
    '''

    return qc.to_gate()


def inCk(N, name='inCK'):
    d_i = QuantumRegister(1, 'd_i')
    a = QuantumRegister(N, 'a')
    one = QuantumRegister(1, 'one')
    qc = QuantumCircuit(d_i,a,one, name=name)


    for i in range(len(a)-1):
        qc.ccx(d_i, one, a[i]) #check ordine

        for p in reversed(range(i)):
            qc.x(a[p])

        qc.append(MCXGate(num_ctrl_qubits=len(d_i[0:])+len(a[:i])+1), d_i[0:]+a[:i+1]+one[0:])

        for p in reversed(range(i)):
            qc.x(a[p])


    qc.ccx(d_i, one, a[-1])
    qc.x(one)


    for p in range(len(a)-1):
        qc.x(a[p])

    qc.append(MCXGate(num_ctrl_qubits=len(d_i[0:])+len(a)-1), d_i[0:]+a[:-1]+one[0:])

    for p in range(len(a)-1):
        qc.x(a[p])
    
    #print(qc.draw())

    return qc.to_gate()



def _get_test_gate(bin_test, N):
    if len(bin_test) != N:
        raise Exception("len bin(test) {}, while N is {}".format(len(bin_test), N)) 
    b = QuantumRegister(N, name='b')
    qc = QuantumCircuit(b, name=str(int(bin_test,2)))

    for i in range(len(bin_test)):
        if bin_test[i] == '1':
            qc.x(b[len(b)-1-i])

    return qc.to_gate()


def or_gate(n):
    a = QuantumRegister(n, 'a')
    v = QuantumRegister(n-1, 'v')
    qc = QuantumCircuit(a,v, name='OR')

    qc.x(a)
    qc.ccx(a[0], a[1], v[0])
    for i in range(len(a[2:])):
        qc.ccx(a[i],v[i-2],v[i-1])

    qc.x(v[-1])


    #print(qc.draw())
    return qc.to_gate()



class BasisRuan:

    def __init__(self, precision=3, threshold=3): #TODO Non mi piace che gli setto io la precision
        self.N = precision #binary vallue lenght
        self.n = int(math.ceil(math.log2(self.N+1))+1) #minimum number of quibits to represent tehe sum of Hamming Distance
        self.t = threshold
        self.l = int(math.pow(2, math.ceil(math.log2(self.N))) - self.N)  #2^k - N. where 2^(k-1) <= N <= 2^k  
        print(self.l)


        self.alen = self.N

        self.one = None

        self.v = None #qubits for training
        self.c = None #TODO: store class of training
        self.bcl = 2 #TODO: binary class lenght
        self.x = None #qubits for test

        self.d = None #Hamming distance result
        self.c1 = None

        self.a = None 
        
    def _init_circuit(self):

        self.v = QuantumRegister(self.N, name='v')
        self.c = QuantumRegister(self.bcl, name='c') #TODO: occhio remapping delle classi
        self.x = QuantumRegister(self.N, name='x')
        self.d = QuantumRegister(self.n, name='d')
        self.c1 = QuantumRegister(1, name='c_1')

        self.one = QuantumRegister(1, name='one')

        #Basis Encoding
        self.u = QuantumRegister(2, 'u')

        #self.a = QuantumRegister(1, name='a')
        self.a = QuantumRegister(self.alen, name='a')  #nel papero dice che a è lungo N. Però dipende da quanto grande "t" e "l". Es: N = 4, t = 5 e l = 2^k - N. NB: se N=4, threshold max è 4
        
        self.cj = ClassicalRegister(1, name='cj')
        self.cv = ClassicalRegister(self.N, name='cv')

        try:
            self.simulator = AerSimulator(method='statevector', shots=8192, device='GPU')
        except AerErrorr as e:
            raise Exception('Simulator'+str(e))

        self.circuit = QuantumCircuit(self.x, self.v, self.c, self.a, self.d, self.u, self.c1, self.one, self.cv, self.cj)


    def fit(self, trainings, test):
        self.circuit = None
        self._init_circuit()

        self.circuit.append(basis_encode_dataset(trainings, self.N),  self.v[0:]+self.u[0:])

        self.circuit.append(_get_test_gate(test, self.N), self.x[0:])

        self.circuit.append(get_hamming_distance_gate(self.N), self.v[0:]+self.x[0:]+self.d[0:]+self.c1[0:]) #TODO: rename


        print(self.l+self.t)
        _registers_switcher(self.circuit, self.l+self.t, self.a) #encode l+t directly on circuit
        
        #self.circuit.append(_get_test_gate(test, self.N), self.a[0:]) #TODO: generalize because I use to encode l+t

        for i in range(len(self.x)):
            self.circuit.append(inCk(self.N), [self.x[i]]+self.a[0:]+self.one[0:])
        #self.circuit.decompose().draw('mpl')
        #plt.show()

        self.circuit.barrier()
        msq = int(math.ceil(math.log2(self.t)))
        print(msq)
        j = QuantumRegister(msq-1, 'j') #rename dentro lo chiamo v
        self.circuit.add_register(j)
        self.circuit.append(or_gate(msq), self.a[-msq:]+j[0:])

        print(self.circuit.draw())

        self.circuit.measure(j, self.cj)

        result = execute(self.circuit, self.simulator).result()
        counts = result.get_counts(self.circuit)

        #postselection = dict(post_select(counts))
        plot_histogram(counts)
        #self.circuit.decompose().draw('mpl')
        plt.show()

        
        

br = BasisRuan(precision=4, threshold=3)
br._init_circuit()

trainings = ['0011']#, '0101'] #salta
trainings = ['0101']#, '0101'] #non salta
test = '1111'

br.fit(trainings, test)

#TODO: dividere per step come nell'articolo
#TODO: lui riusa i qubit su x
#Tutto bello ma x l'ho perso quindi non so come viene classicato ciascun elem
