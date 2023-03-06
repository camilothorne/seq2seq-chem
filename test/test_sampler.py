import sys
sys.path.insert(0,'../../seq2seq-chem')

from utils.onehotencode import OneHotEncode

xdata_path   = '../data/all-smi2smi.tsv'     # Path to the data file on disk.
xnum_samples = 1000                          # Max number of samples to train on.
  
print('\n----------------')

char_table1 = OneHotEncode()
char_table1.build_char_table(xdata_path, ['smiles','smiles-out'])
char_table1.corpus_stats()
 
print('\n----------------')      

(x,y,z,w) = char_table1.select_sample(0, xnum_samples*3, print_stats=True)
print('source','\t','target')
for p in range(0, min(w.shape[0],10)):
    print(w[p,0],'\t',w[p,1])
    
print('\n----------------')      

char_table1.select_sample(xnum_samples, xnum_samples*2, print_stats=True)

print('\n----------------')                    

char_table2 = OneHotEncode()
char_table2.build_char_table(xdata_path, ['smiles','smiles-out'])
char_table2.corpus_stats()
     
print('\n----------------')         
     
char_table2.select_sample(0, xnum_samples, print_stats=True)

print('\n----------------')      

char_table2.select_sample(xnum_samples, xnum_samples+200, print_stats=True)

print('\n----------------')

del char_table1
del char_table2
char_table3 = OneHotEncode()
char_table3.build_char_table(xdata_path, ['smiles','smiles-out'])
char_table3.to_pickle('../data/test_pickle.pk')
del char_table3

char_table4 = OneHotEncode()
char_table4.from_pickle('../data/test_pickle.pk')
char_table4.corpus_stats()

print('\n----------------')

char_table4.select_sample(0, xnum_samples, print_stats=True)

print('\n----------------')
