from framework import file_txt
import os

directory = '../data/NI/i_phase' #directory where all the batch files are stored
file = os.listdir(directory) #list all the files in the i_phase directory
file = [os.path.join(directory, f) for f in file] #relative path with respect to "../data/"
data = file_txt.read(file, fm=60, batch=True) #load the structures in batch format
print(type(data))