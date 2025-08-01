from framework import file_txt

file = '../data/NI/i_phase/BQ_PC_1.txt' #path to the file in the directory prior
data = file_txt.read(file, fm=60) #load the structure
print(type(data))