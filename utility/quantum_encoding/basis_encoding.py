from qiskit import QuantumCircuit, QuantumRegister
from qiskit.extensions import UnitaryGate



def get_binary_value(value, encoding_length): 
    bin_str_pattern = '{:0%sb}' % encoding_length
    value = bin_str_pattern.format(value)
    return value

def get_binary_value_gate(binary_value, binary_lenght, name=''):
    if len(binary_value) != binary_lenght:
        raise Exception("len bin(test) {}, while binary_lenght is {}".format(len(binary_value), binary_lenght)) 

    b = QuantumRegister(binary_lenght, name='b')
    qc = QuantumCircuit(b, name=name+' '+str(int(binary_value, 2)))

    for i in range(len(binary_value)):
        if binary_value[i] == '1':
            qc.x(b[len(b)-1-i])

    return qc.to_gate()

def _get_binary_dataset(dataset, N):
    binary_dataset = []
    #Binary Length
    #bin_str_pattern = '{:0%sb}' % N
    #Convert dataset in binary values in N qubits
    for elem in dataset:
        binary_elem = get_binary_value(elem, N)  #return the binary representation of i
        #binary_elem = bin_str_pattern.format(elem)
        binary_dataset.append(binary_elem)
    return binary_dataset


def _c0not(qc, control, target):
    qc.x(control)
    qc.cx(control, target)
    qc.x(control)


def _mu_rotation(qc, c, mu):
    rotation_matrix = [ 
                        [((mu-1)/mu)**(1/2), 1/(mu**(1/2))], 
                        [-1/(mu**(1/2)), ((mu-1)/mu)**(1/2)] 
                       ]

    u = UnitaryGate(rotation_matrix, label='rotation')
    qc_temp = QuantumCircuit(1)
    qc_temp.append(u, [0])
    custom = qc_temp.to_gate().control(1)
    qc.append(custom, [c[0], c[1]])


def basis_encode_dataset(dataset, N):
    binary_dataset = dataset

    #Quantum Circuit
    x = QuantumRegister(N, name='x') #Register of N qubits, where N is the lenght of the binary representation
    c = QuantumRegister(2, name='c')

    qc = QuantumCircuit(x,c, name='Basis Encoded Dataset')
    
    for m in range(len(binary_dataset)): #iterate over inputs
        for n in range(N):
            bit = int(binary_dataset[m][n])
            if bit: #If equals to 1, then _c0not(c[1], n)
                _c0not(qc,c[1],N-n-1) 

        _c0not(qc, c[1], c[0]) #Flip c_0 if c_1 == 0

        mu = len(binary_dataset) + 1 - (m  + 1)
        _mu_rotation(qc, c, mu)


        for n in range(N):
            bit = int(binary_dataset[m][n])
            if not bit:
                qc.x(x[N-n-1])

        qc.mct(x[0:], c[0])
    
        for n in range(N):
            bit = int(binary_dataset[m][n])
            if not bit:
                qc.x(x[N-n-1])

        #reset input only on the "processing branch" c2 == 0
        for n in range(N):
            bit = int(binary_dataset[m][n])
            if bit:
                _c0not(qc, c[1], N-n-1) 

    return qc.to_gate()
 



