# About DSP4IM
DSP4IM is a Python toolbox developed for applying standard digital signal processing techniques to all induction motor signals acquired by the **PRISMA (Process, Robotics, Intelligent Systems, and MAchines) laboratory** team. Given the multiple types of utilized acquisition hardware and their data formats, DSP4IM provides methods to automatically extract the required attributes and organize them in custom data structures. As of the latest version of the framework, the supported files are as follows: ANSYS simulation CSV files, Quantum X-MX840B MAT files, National Instruments hardware TXT files, and Matlab structs from the LAIPSE induction motor database (DOI: https://dx.doi.org/10.21227/fmnm-bn95).

# Running the framework
To set up the toolbox, first clone this repository on your machine:

```
cd <desired_path>
git clone https://github.com/ceciliapag/DSP4IM.git
cd DSP4IM
```

Then, configure a Python environment. The framework has been successfully tested in Python 3.10, 3.11, and 3.12. Also, in both Linux and Windows operating systems. Lastly, install the required packages listed in the requirements text file at the root of the repository with:

```
pip install -r requirements.txt
```

Once all packages are installed, the toolbox is ready for use. In the following sections, you may find some examples of how to use the DSP4IM tools. 


## ANSYS simulation files
The infrastructure requirement is to add the CSV files from ANSYS in a directory at the root of the repository. The name of the directory itself can be arbitrary. However, both current and speed files must obey the following rules:
```
current -> ./<data_directory>/<simulation_directory>/corrente {load percentage}.csv
speed -> ./<data_directory>/<simulation_directory>/velocidade {load percentage}.csv
```

For example:
```
+---DSP4IM   
|   \---<data_directory>
|       +---\<simulation_directory>
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

To read the contents of the file into a SimuData structure:
```
from framework import file_csv
data = file_csv.read(<data_directory>, <load_percentage>, <synchronous_speed>, fm=<fundamental_frequency>, n_periods=<number_of_periods_for_FFT>, normalize_by=<FFT_normalization_function>)
```

## Quantum X files

The infrastructure requirement is to add the MAT files from Quantum X to a directory at the root of the repository. The directory and file names can be arbitrary. For example, most files follow the given structure:
```
./<data_directory>/<quantumx_directory>/experimento_{experiment number}_carga_{load percentage}__{sampling frequency}Hz_{sampling frequency}Hz.MAT
```

To read the contents of the file into a SensorData structure:
```
from framework import file_sensor_mat
data = file_sensor_mat.read(<data_directory>, <load_percentage>, <synchronous_speed>, fm=<fundamental_frequency>, n_periods=<number_of_periods_for_FFT>, normalize_by=<FFT_normalization_function>)
```

## National Instruments hardware files
The infrastructure requirement is to add the text files from the hardware to a directory at the root of the repository. The directory and file names can be arbitrary. For example, most files follow the given structure:
```
./<data_directory>/<nihardware_directory>/<filename>.txt
```

To read the contents of the file into a NIHardwareData structure:
```
from framework import file_txt
data = file_txt.read(<data_directory>, fm=<fundamental_frequency>, n_periods=<number_of_periods_for_FFT>, normalize_by=<FFT_normalization_function>)
```

##  LAIPSE induction motor database files
The infrastructure requirement is to add the text files from the database to a directory at the root of the repository. The directory name can be arbitrary. However, the files must keep their original names, such as:
```
./<data_directory>/<laipse_directory>/struct_r<number_of_broken_rotor_bars>b_R1.mat
```

To read the contents of the file into a LaipseData structure:
```
from framework import file_mat
data = file_mat.read(<data_directory>, <torque>, fs=<sampling_frequency>, fm=<fundamental_frequency>, n_periods=<number_of_periods_for_FFT>, exp_num=<number_of_the_experiment>, transient=<True/False>, normalize_by=<FFT_normalization_function>)
```

## User guides
In-depth guides of the available methods are available at the [examples](https://github.com/ceciliapag/DSP4IM/tree/main/examples) directory, and are highly recommended for first-time users!

- **SimuData**: https://github.com/ceciliapag/DSP4IM/blob/main/examples/read_ansys_data.py
- **SensorData**: https://github.com/ceciliapag/DSP4IM/blob/main/examples/read_sensor_data.py
- **BatchSensorData**: https://github.com/ceciliapag/DSP4IM/blob/main/examples/read_sensor_data_batch.py
- **NIHardwareData**: https://github.com/ceciliapag/DSP4IM/blob/main/examples/read_NI_data.py
- **LaipseData**: https://github.com/ceciliapag/DSP4IM/blob/main/examples/read_LAIPSE_data.py

# Related works
- **Nascimento, P. C.** (2025). Development of electromagnetic models of induction motors with broken rotor bars. \[Master's Dissertation\, Programa de Pós-Graduação em Sistemas de Energia - Universidade Tecnológica Federal do Paraná]
- **Gremonini, L.** (2025). Modelagem Numérica de um Motor de Indução Trifásico Aplicado Ao Diagnóstico de Defeitos em Barras de Rotor e Enrolamentos de Armadura. \[Master's Dissertation\, Programa de Pós-Graduação em Sistemas de Energia - Universidade Tecnológica Federal do Paraná]

# DSP4IM collaborators
- Bruno Akihiro Tanno Iamamura
- Cecilia Pagnozzi do Nascimento
- Lucas Gremonini
- Mateus Yamada Müller
- Narco Afonso Ravazzoli Maciejewski
- Thiago de Paula Machado Bazzo

# Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.