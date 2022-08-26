from qiskit.providers.aer import AerError, AerSimulator
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
import numpy as np
import math

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
        _registers_switcher(circuit, idx, feature_qubits)
        circuit.mcry(vector[idx], control_qubits, target, qram)
        _registers_switcher(circuit, idx, feature_qubits)
        circuit.barrier()


class AmplitudeQKNeighborsClassifier:

    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
        self.circuit = None
        self.qrama = None
        self.knna = None
        self.r = None
        self.i = None
        self.m = None
        self.c = None
        self.classical_r = None
        self.classical_knna = None
        self.classical_classes = None


    def _init_circuit(self, X_train, y_train):
        N = X_train.shape[1] #Number of features
        tr_size = X_train.shape[0] #Number of training inputs
        n_classes = len(np.unique(y_train)) #Assuming that training contains all type of classes 
        n_feature_indexes = int(math.ceil(math.log2(N))) #Needed qubit to represent all the feature of a given element
        n_training_indexes = int(math.ceil((math.log2(tr_size)))) #Needed qubit to represent all training inputs
        n_class_indexes = int(math.ceil(math.log2(n_classes))) #Needed qubit to represent all the classes

        self.qrama = QuantumRegister(n_training_indexes + n_feature_indexes + 1, 'qram') #Ancilla QRAM  
        self.r = QuantumRegister(1, 'r') #Rotation Register for QRAM
        self.i = QuantumRegister(n_feature_indexes, 'i') #Feature index

        #Qubits dealing classification
        self.knna = QuantumRegister(1, 'knna') #Ancilla for KNN. 
        self.m = QuantumRegister(n_training_indexes, 'm') #Training index
        self.c = QuantumRegister(n_class_indexes, 'class') #Class forx KNN

        self.classical_r = ClassicalRegister(1, 'classical_r') # r(1) 
        self.classical_knna = ClassicalRegister(1, 'classical_knna') # knna(1)
        self.classical_classes = ClassicalRegister(n_class_indexes, 'classical_classes') # n_class_indexes(n_class_indexes)

        self.circuit = QuantumCircuit(self.qrama, self.i, self.knna, self.m, self.r, self.c, self.classical_r, self.classical_knna, self.classical_classes)


    def fit(self, X_train, y_train):
        self.circuit = None #Reset Circuit

        X_train = np.arcsin(X_train)

        self._init_circuit(X_train, y_train)

        #Superpose
        self.circuit.h(self.knna)
        self.circuit.h(self.m)
        self.circuit.h(self.c)
        self.circuit.h(self.i)

        #Starting branch: training
        #------- BEGIN: ENCODE TRAINING VECTORS
        for idx, x_i, y_i in zip(range(len(X_train)), X_train, y_train):

            _registers_switcher(self.circuit, idx, self.m) #Switching index
            _registers_switcher(self.circuit, y_i, self.c) #Switching class

            self.circuit.barrier()

            _amplitude_mapper(self.circuit, x_i, self.i, self.i[0:]+self.knna[0:]+self.m[0:]+self.c[0:], self.r[0], self.qrama)

            _registers_switcher(self.circuit, idx, self.m) #undo index vector
            _registers_switcher(self.circuit, y_i, self.c) #undo class

            self.circuit.barrier()
        #------- END: ENCODE TRAINING VECTORS



    def predict(self, X_test, n_shots=8192):
        if self.circuit == None:
            raise Exception("Circuit not available. Please use method 'fit'")

        X_test = np.arcsin(X_test)

        # Switching to training
        self.circuit.x(self.knna) 
        self.circuit.barrier()

        #------- BEGIN: ENCODE TEST VECTOR
        _amplitude_mapper(self.circuit, X_test[0], self.i, self.i[0:]+self.knna[0:], self.r[0], self.qrama) 
        self.circuit.barrier()
        #------- END: ENCODE TEST VECTOR

        #Measurements
        self.circuit.measure(self.r[0], self.classical_r)
        self.circuit.h(self.knna) #Classify
        self.circuit.measure(self.knna, self.classical_knna)

        self.circuit.measure(self.c, self.classical_classes)

        try:

            simulator = AerSimulator(method='statevector', shots=8192)
            #simulator = Aer.get_backend('qasm_simulator')
        except AerError as e:
            print(e)

        result = execute(self.circuit, simulator, shots=n_shots).result()
        counts = result.get_counts(self.circuit)

        #Post Selection on ' 0 1'
        post_select = lambda counts: [(state, occurences) for state, occurences in counts.items() if state[-1] == '1' and state[-3] == '0' ]
        postselection = dict(post_select(counts))
        sorted_votes = sorted(postselection.items(), key=lambda x: x[1], reverse=True)
        
        majority_vote = sorted_votes[0][0][0:-4] #removing values for knna and r ' 0 1'
        self.circuit = None
        return int(majority_vote,2)

##########################################################################
