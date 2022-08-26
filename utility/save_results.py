from pathlib import Path


def _printer(f, exps):
    for elems in exps:
        f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(elems['dataset'],elems['n_neighbors'],elems['n_classes'],elems['n_features_real'],elems['method'],elems['n_bits'],elems['enc_type'],elems['training_type'],elems['training_size'],elems['nbr_features'],elems['pca'],elems['test_id'],elems['exp_id'],
                                                                           elems['train_time'],elems['test_time'],elems['y_true'],elems['y_pred']))

def print_to_file_old(name, KNN_exps, aQKNN_exps, bKNN_exps, bQKNN_exps):
    if bQKNN_exps == [] or aQKNN_exps == [] or KNN_exps == [] or bKNN_exps == []:
        print("-------------------------------------------")
        raise Exception()
        return
    if aQKNN_exps != None:
        sub_dir_name = aQKNN_exps[0]['dataset']+'/amplitude'
    if bQKNN_exps != None:
        sub_dir_name = bQKNN_exps[0]['dataset']+'/basis/'+bQKNN_exps[0]['enc_type']
    Path("./results/"+sub_dir_name).mkdir(parents=True, exist_ok=True)
    name = './results/'+sub_dir_name+'/'+name+'.csv'
    if Path(name).is_file():
        f = open(name ,'a')
    else:
        f = open(name ,'w')
        f.write('dataset,n_neighbors,n_classes,n_features_real,method,n_bits,basis_encoding_type,training_type,training_size,nbr_features,pca,test_id,exp_id,train_time,test_time,y_true,y_pred\n')
    if KNN_exps != None:
        _printer(f, KNN_exps)
    if aQKNN_exps != None:
        _printer(f, aQKNN_exps)
    if bQKNN_exps != None:
        _printer(f, bQKNN_exps)
    if bKNN_exps != None:
        _printer(f, bKNN_exps)
    f.close()


'''
methodology: amplitude / basis / basis_thhresold
enc_type: -1 if not basis encoding
'''
def print_to_file(name, classic_exps, quantum_exps):        
    if quantum_exps == [] or classic_exps == []:
        print('-------------')
        raise Exception()
        return
    if quantum_exps[0]['enc_type'] == -1:
        sub_dir_name = quantum_exps[0]['dataset']+'/'+quantum_exps[0]['methodology']
    else:
        sub_dir_name = quantum_exps[0]['dataset']+'/'+quantum_exps[0]['methodology']+'/'+quantum_exps[0]['enc_type']
    Path("./results/"+sub_dir_name).mkdir(parents=True, exist_ok=True)
    name = './results/'+sub_dir_name+'/'+name+'.csv'
    if Path(name).is_file():
        f = open(name ,'a')
    else:
        f = open(name ,'w')
        f.write('dataset,n_neighbors,n_classes,n_features_real,method,n_bits,basis_encoding_type,training_type,training_size,nbr_features,pca,test_id,exp_id,train_time,test_time,y_true,y_pred\n')
    _printer(f, quantum_exps)
    _printer(f, classic_exps)
    f.close()

