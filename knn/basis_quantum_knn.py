from qiskit.providers.aer import AerSimulator, AerError
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
import math
import sys
sys.path.append('..')
from utility.quantum_encoding.basis_encoding import *


#Counts the number of "1"
def _count_1s(qc, x, x_index, h, h_index, c1):
    qc.ccx(x[x_index], c1, h[h_index])
    
    for i in range(h_index):
        qc.x(h[i])
    
    qc.mct([x[x_index]]+h[:h_index+1], c1)

    for i in range(h_index):
        qc.x(h[i])

        
#Return the gate which compute the subtraction
def subs_gate(n, name="sub"): 
    s1 = QuantumRegister(n,'sub_1')
    s2 = QuantumRegister(n,'sub_2')
    r = QuantumRegister(n, 'result')
    qc = QuantumCircuit(s1,s2,r, name=name)
    
    for i in range(n):
        if i == n-1:
            qc.cx(s1[i],s2[i])

            qc.cx(s2[i],r[i])

            qc.cx(s1[i],s2[i])
        else:
            qc.ccx(s1[i], s2[i], r[i+1])
            qc.cx(s1[i],s2[i])

            qc.ccx(s2[i], r[i], r[i+1])
            qc.cx(s2[i], r[i])

            qc.cx(s1[i], s2[i])
    return qc.to_gate()

# Compute the 2complement of value in base n
def two_binary_complement_gate(value, n, name='value'): 
    if value >= 2**n:
        raise Exception("Unable to map value in 2^n bit") 
    bin_str_pattern = '{:0%sb}' % n
    binary_string = bin_str_pattern.format(value) #return the binary repr of i. 
    binary_string_list = list(binary_string)
    for i in range(len(binary_string_list)):
        if binary_string_list[i] == '0':
            binary_string_list[i] = '1'
        else:
            binary_string_list[i] = '0'
    bitwise = ''.join(binary_string_list)
    int_repr = int(bitwise,2)
    int_two_complement = int_repr + 1
    int_two_complement = int_two_complement%(2**n)
    bin_two_complement = bin_str_pattern.format(int_two_complement)
    
    b = QuantumRegister(n, name='b')
    qc = QuantumCircuit(b, name=name)
    
    for i in range(len(bin_two_complement)):
        if bin_two_complement[i] == '1':
            qc.x(b[len(b)-1-i])
    return qc.to_gate()



# Return the gate which compute the Hamming Distance        
def get_hamming_distance_gate(N, name="hamming_distance"):
    n = math.ceil(math.log2(N+1)) # minimum number of qubits to represent the sum of Hamming Distance's 1s
    
    v = QuantumRegister(N, name='v')
    x = QuantumRegister(N, name='x')
    h = QuantumRegister(n+1, name='h')
    c1 = QuantumRegister(1, name='c1')
    
    qc = QuantumCircuit(v, x, h, c1, name=name)
    
    qc.x(c1) #Setting to 1
    
    for v_i, x_i in zip(v, x): 
        qc.cx(v_i, x_i)  #XOR(V,X)   
    
    for x_index in range(len(x)):   #Sum 1s of XOR(V,X)
        for h_index in range(len(h)):
            _count_1s(qc, x, x_index, h, h_index, c1)
        qc.ccx(x[x_index], c1, h[len(h)-1])
        qc.cx(x[x_index],c1)

    return qc.to_gate()
    
# Binary Search
def _update_key(side, prev_key, minimum, maximum):
    if side == 1: #go_left 
        maximum = prev_key - 1
    else: #go_right
        minimum = prev_key + 1
    key = math.floor((maximum+minimum)/2)
    return key, minimum, maximum

# Get the gate that encode the test vector
def _get_test_gate(bin_test, N):
    if len(bin_test) != N:
        raise Exception("len bin(test) {}, while N is {}".format(len(bin_test), N)) 
    b = QuantumRegister(N, name='b')
    qc = QuantumCircuit(b, name=str(int(bin_test,2)))

    for i in range(len(bin_test)):
        if bin_test[i] == '1':
            qc.x(b[len(b)-1-i])

    return qc.to_gate()


class BasisQKNeighborsClassifier:

    def __init__(self, precision=3, n_neighbors=1):
        self.N = precision #binary value length
        self.n = math.ceil(math.log2(self.N+1))+1 #minimum number of qubits to represent the sum of Hamming Distance's 1s
        self.n_neighbors = n_neighbors
        self.v = None #Qubits storing the Trainings
        self.x = None #Qubits storing the Tests
        self.d = None #Hamming Distances
        self.c1 = None 

        self.a = None # Result of (di -key) (n+1 to store overflow qubit, a2)
        self.b = None # store key

        self.c_training = None 
        self.a2 = None 
        self.circuit = None
        self.template_circuit = None
        self.simulator = None

    def _init_circuit(self):
        #Quantum Register
        self.v = QuantumRegister(self.N, name='v')
        self.x = QuantumRegister(self.N, name='x')
        self.d = QuantumRegister(self.n, name='d')
        self.c1 = QuantumRegister(1, name='c_1') 
        self.a = QuantumRegister(self.n, 'sub')
        self.b = QuantumRegister(self.n, 'key')

        #Basis Encoding
        self.c = QuantumRegister(2, name='c')

        #Classical Register
        self.c_training = ClassicalRegister(self.N, 'c_training')
        self.c_a2 = ClassicalRegister(1, 'c_a2') #measurement of a2

        #Instantiate Simulator
        try:
            self.simulator = AerSimulator(method='statevector', shots=8192)
        except AerError as e:
            print(e)

        self.circuit = QuantumCircuit(self.v, self.c, self.x, self.d, self.c1, self.a, self.b)

    def fit(self, trainings):
        self.circuit = None
        self._init_circuit()

        self.template_circuit = self.circuit.copy()
        self.template_circuit.append(basis_encode_dataset(trainings, self.N), self.v[0:]+self.c[0:])


        # Appending Hamming Distance Gate --------------------#
        self.template_circuit.append(get_hamming_distance_gate(self.N), self.v[0:]+self.x[0:]+self.d[0:]+self.c1[0:])
        #-----------------------------------------------------
        
        # Appending Subtraction Gate -------------------------#
        #     (di - key) if di < key --> most significant bit is 0, otherwise 1
        self.template_circuit.append(subs_gate(self.n), self.d[0:]+self.b[0:]+self.a[0:])
        self.template_circuit = self.template_circuit.to_gate()
        #---------------------------------------------------#
        

    def predict(self, test):
        minimum = 1
        maximum = self.N
        ones = []
        side = 0
        key = math.floor((maximum+minimum)/2)

        prediction = []

        while minimum <= maximum:

            init_circuit = self.circuit.copy()

            #Loading Test
            self.circuit.append(_get_test_gate(test, self.N), self.x[0:])

            self.circuit.append(two_binary_complement_gate(key, self.n, name='key: '+str(key)), self.b[0:])
            self.circuit.append(self.template_circuit, self.circuit.qubits)

            #Appending qubits for measurements
            self.circuit.add_register(self.c_training)
            self.circuit.add_register(self.c_a2)
            self.circuit.measure(self.a[-1:],self.c_a2)
            self.circuit.measure(self.v, self.c_training)

            #------------- Simulation -------------------------#
            result = execute(self.circuit, self.simulator).result()
            counts = result.get_counts(self.circuit)

            post_select = lambda counts: [(state, occurences) for state, occurences in counts.items()]
            postselection = dict(post_select(counts))
            postselection = sorted(postselection.items(), key=lambda x: x[1], reverse=True)
                
            ones = []
            for elem,_ in postselection:
                splitted = elem.split(' ')
                overflow = int(splitted[0])
                if overflow == 1:
                    side = 1
                    ones.append(splitted[1])
                
            #Exponential Search stops when there is just one element, otherwise on until minimum <= maximum 
            if ones:
                prediction = ones
            #--------------------------------------------------#

            #------------ Key Update -------------------------#
            #print("key = {}".format(key))
            #print("side = {}".format(side))
            #print("K = 1 -> {}\n".format(prediction))
            key, minimum, maximum = _update_key(side, key, minimum, maximum)
            side = 0
            ones = []
            self.circuit = init_circuit.copy()

        #If more than one results, then return -1
        if len(prediction) == 1:
            return prediction[0]
        return -1
        
        #--------------------------------------------------#

