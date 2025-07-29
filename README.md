# About anDSPys
Supporting codes (under development) for the paper "Fault diagnosis of broken rotor bars using the Fast Fourier Transform" by Nascimento, C. P. 

# Running the framework
In both "examples", you may find some examples of how to use our tool. 

## file_csv
The only infrastructure-wise requirement is to add the .csv files from ANSYS in a directory at the root of the repository. The name of the directory itself can be arbitrary. However, both current and speed files must obey the following rules:
```
current -> ./<data_directory>/<simulation_directory>/corrente {load percentage}.csv
speed -> ./<data_directory>/<simulation_directory>/velocidade {load percentage}.csv
```

For example:
```
+---anDSPys   
|   \---data
|       +---\1_broken_bar
|       |   +---corrente 25.csv
|       |   |       
|       |   +---corrente 50.csv
|       |   |       
|       |   +---corrente 75.csv
|       |   |       
|       |   +---corrente 100.csv
|       |   |       
|       |   +---velocidade 25.csv
|       |   |       
|       |   +---velocidade 50.csv
|       |   |       
|       |   +---velocidade 75.csv
|       |   |       
|       |   +---velocidade 100.csv 
|       |           
.       .
.       .
.       .              
```

## file_sensor_mat
The only infrastructure-wise requirement is to add the .MAT files from the bench testing in a directory at the root of the repository. The name of the directory itself can be arbitrary. However, the files must obey the following rule:
```
./<data_directory>/<files_directory>/experimento_{experiment number}_carga_{load percentage}__19200Hz_192000Hz.MAT
```

For example:
```
+---anDSPys   
|   \---data
|       +---\benchtesting_PD
|       |   +---experimento_1_carga_100__19200Hz_19200Hz.MAT
|       |   |       
|       |   +---experimento_2_carga_100__19200Hz_19200Hz.MAT
|       |   |       
|       |   +---experimento_3_carga_100__19200Hz_19200Hz.MAT
|       |   |       
|       |   +---experimento_4_carga_100__19200Hz_19200Hz.MAT
|       |   |       
|       |   +---experimento_5_carga_100__19200Hz_19200Hz.MAT
|       |   |       
|       |   +---experimento_6_carga_100__19200Hz_19200Hz.MAT
|       |   |       
|       |   +---experimento_7_carga_100__19200Hz_19200Hz.MAT
|       |   |       
|       |   +---experimento_8_carga_100__19200Hz_19200Hz.MAT
|       |           
.       .
.       .
.       .  
```

# Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
