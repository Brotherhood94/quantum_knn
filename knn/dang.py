from qiskit.providers.aer import AerSimulator, AerError
from qiskit.circuit.library import MCXGate, IntegerComparator, MCMT, RYGate
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
import math
import sys
sys.path.append('..')
from utility.quantum_encoding.basis_encoding import *
import numpy as np

import random

#TODO: refactor also in the other files (ex: basis_threshold.py)
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


def _init_index_state_gate(register_size, n_elements, name='init_indexes'):
    q_index = QuantumRegister(register_size, 'q_index')
    flags = QuantumRegister(2, 'flags')
    comp_anc = QuantumRegister(len(q_index)-1, 'comp_anc') #TODO: check if M is encodable in qubits
    qc = QuantumCircuit(q_index, comp_anc, flags, name=name)

    qc.x(q_index)
    qc.append(MCXGate(num_ctrl_qubits=len(q_index)), q_index[0:]+[flags[1]])
    qc.x(q_index)

    print(len(q_index))
    print(len(comp_anc))
    qc.append(IntegerComparator(num_state_qubits=len(q_index), value=n_elements+1, geq=True), q_index[0:]+comp_anc[0:]+[flags[0]]) #q_index > M

    return qc

#-----------DUPLICATE----------------------- (amplitude mapper)
#Select a specific quantum register 
def _registers_switcher(circuit, value, qubit_index):
    bin_str_pattern = '{:0%sb}' % len(qubit_index)
    value = bin_str_pattern.format(value)[::-1] 
    for idx, bit in enumerate(value):
        if not int(bit):
            circuit.x(qubit_index[idx])

#Store the value vector[idx] to the corresponding register, the value vector[idx] in the target qubit
def _amplitude_mapper(circuit, vector, feature_qubits, control_qubits, target, qram=None):
    for idx in range(len(vector)):
        _registers_switcher(circuit, idx+1, feature_qubits) #TODO: qua c'è un idx+1
        circuit.append(MCMT(RYGate(vector[idx]), num_ctrl_qubits=len(control_qubits), num_target_qubits=1), control_qubits[0:]+[target])#mcry(vector[idx], control_qubits, target) #TODO: according to paper is 2*theta, not just theta
        _registers_switcher(circuit, idx+1, feature_qubits)
        circuit.barrier()

#-----------DUPLICATE----------------------- (amplitude mapper)


class DangQuantumKnn:

    def __init__(self, k=1):
        self.k = k


    def _init_circuit(self, M, N):

        #Valid 0 < j <= M
        m = int(math.ceil(math.log2(M+1))) #register size for indexing trainings
        n = int(math.ceil(math.log2(N+1))) #register size for indexing features

        bin_M = bin(M).replace("0b", "") #Binary representation of M
        print(bin_M)

        self.j = QuantumRegister(m, 'j') #qubits indexing trainings
        self.comp_anc_tr = QuantumRegister(len(self.j)-1, 'comp_anc_training')
        self.i = QuantumRegister(n, 'i') #qubits indexing features
        self.comp_anc_features = QuantumRegister(len(self.j)-1, 'comp_anc_features')
        self.c = QuantumRegister(1, 'tr_class') #TODO: Parametrize

        self.i_test = QuantumRegister(n, 'i_test') #qubits indexing features
        self.comp_anc_features_test = QuantumRegister(len(self.j)-1, 'comp_anc_features_test')

        self.b = QuantumRegister(1, 'tr_data')
        self.a = QuantumRegister(1, 'test_data')
        self.s = QuantumRegister(1, 'swap_test')


        self.flags_trainings = QuantumRegister(2, 'flags_trainings')
        self.flags_features = QuantumRegister(2, 'flags_features')
        self.flags_features_test = QuantumRegister(2, 'flags_features_test')


        self.c_j = ClassicalRegister(len(self.j), 'c_j')
        self.c_i = ClassicalRegister(len(self.i), 'c_i')
        self.c_flags_trainings = ClassicalRegister(2, 'c_flags_trainings')
        self.c_flags_features = ClassicalRegister(2, 'c_flags_features')

        self.circuit = QuantumCircuit(self.s, self.j, self.comp_anc_tr, self.flags_trainings, self.i, self.comp_anc_features, self.flags_features, self.i_test, self.comp_anc_features_test, self.flags_features_test, self.b, self.c, self.a, self.c_j, self.c_flags_trainings, self.c_i, self.c_flags_features)

        #self.circuit.append(get_binary_value_gate(bin_M, len(self.qbin_M), name='bin_M'), self.qbin_M)
        try:
            self.simulator = AerSimulator(method='statevector', shots=8192, device='CPU')
            #self.simulator = AerSimulator(method='statevector', shots=8192, device='GPU', cuStateVec_enable=True)
        except AerError as e:
            raise Exception('Simulator'+str(e))



    def fit(self, X_train, y_train):

        self.circuit = None

        #TODO: arcsin applicato fuori (check)
        #M = X_train.shape[0] #Number of trainings
        #N = X_train.shape[1] #Number of features
        M = 4
        N = 4

        print('M '+str(M))
        print('N '+str(N))

        self._init_circuit(M, N)

        self.circuit.h(self.j)
        self.circuit.append(_init_index_state_gate(len(self.j), M, name='init_indexes_trainings'), self.j[0:]+[self.flags_trainings[0]]+self.comp_anc_tr[0:]+[self.flags_trainings[1]]) #q_index > M

        self.circuit.h(self.i)
        self.circuit.append(_init_index_state_gate(len(self.i), N, name='init_indexes_features'), self.i[0:]+[self.flags_features[0]]+self.comp_anc_features[0:]+[self.flags_features[1]]) #q_index > M

        self.circuit.h(self.i_test)
        self.circuit.append(_init_index_state_gate(len(self.i_test), N, name='init_indexes_features_test'), self.i_test[0:]+[self.flags_features_test[0]]+self.comp_anc_features_test[0:]+[self.flags_features_test[1]]) #q_index > M
        self.circuit.barrier()


        #TODO: ricorda che loro partono da 1 e non da 0
        #------- BEGIN: ENCODE TRAINING VECTORS
        for idx, x_i, y_i in zip(range(len(X_train)), X_train, y_train):

            _registers_switcher(self.circuit, idx+1, self.j) #Switching index
            _registers_switcher(self.circuit, y_i, self.c) #Switching class

            self.circuit.barrier()

            _amplitude_mapper(self.circuit, x_i, self.i, self.i[0:]+self.j[0:]+self.c[0:], self.b[0])
            #TODO: in original algorithm seems not encoding classes

            _registers_switcher(self.circuit, idx+1, self.j) #undo index vector
            _registers_switcher(self.circuit, y_i, self.c) #undo class

            self.circuit.barrier()
        #-------------------------------------



    #TODO: alpha è su un'altro indice i 
    def predict(self, X_test):
        if self.circuit == None:
            raise Exception("Circuit not available. Please use method 'fit'")

        X_test = np.arcsin(X_test) #TODO: check come viene passato
        #------- BEGIN: ENCODE TEST VECTOR
        _amplitude_mapper(self.circuit, X_test[0], self.i_test, self.i_test[0:], self.a[0])
        self.circuit.barrier()

        #--- SWAP TEST
        self.circuit.h(self.s)
        for l in range(len(self.i)):
            self.circuit.cswap(self.s, self.i[l], self.i_test[l])
        self.circuit.cswap(self.s, self.a, self.b)
        self.circuit.h(self.s)
        #------------
        
        self.circuit.barrier()

        self.circuit.measure(self.j, self.c_j)
        self.circuit.measure(self.flags_trainings, self.c_flags_trainings)
        self.circuit.measure(self.i, self.c_i)
        self.circuit.measure(self.flags_features, self.c_flags_features)

        #why no post selection for qram?

        result = execute(self.circuit, self.simulator).result()
        counts = result.get_counts(self.circuit)
        print(self.circuit.draw())

#        for el in counts:
#            print(el)

        print(counts)


        

dqk = DangQuantumKnn()

#TODO: check order switch register
X_train = []
y_train = []
#test con n_items=16 out of index
n_items = 3
n_feature = 2
for i in range(n_items):
    el = []
    for i in range(n_feature):
        el.append(random.uniform(0, 1))
    X_train.append(el)
    y_train.append(random.randint(0, 1))

dqk.fit(X_train, y_train)

X_test = []
el = []
for i in range(n_feature):
    el.append(random.uniform(0, 1))
X_test.append(el)
print(X_test)
dqk.predict(X_test) 
