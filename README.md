
# Matrix Multiply Benchmark
Matrix Multiply Benchmark on GPU and CPU using OpenMP, OpenACC, HIP, rocBLAS, CUDA, cuBLAS


## Author

- [@Kass](https://www.github.com/Kassouley) 

![Kass](https://cdn.discordapp.com/attachments/705826516520665191/1116698582557397062/canvas100.png)


## Installation

Advice : use the script in the 'script/' folder

No particulary installation needed.
Just build with :
```bash
make measure KERNEL=[KERNEL_NAME] CLOCK=[RDTSC|MS] GPU=[NVIDIA|AMD]
make check KERNEL=[KERNEL_NAME] GPU=[NVIDIA|AMD]
```

KERNEL_NAME should be in uppercase.

Then run with :
```bash
./measure {matrix_size} {n_warmup} {n_rep}
./check {matrix_size} {output file}
```

- matrix_size is the size n of an n x n matrix
- n_warmup is the number of warmup before starting the bench
- n_rep is the number of repetitions to dampen the accuracy of the timer
    
## Code Features

- Shows us the time of a kernel in millisecond or RDTSC-Cycles
- Benchmark on CPU using OpenMP and CBLAS (on NVIDIA & AMD)
- Benchmark on GPU using OpenMP with and without the data transfer (on NVIDIA & AMD)
- Benchmark on GPU using OpenACC with and without the data transfer (on NVIDIA)
- Benchmark on GPU using HIP with and without the data transfer (on AMD)
- Benchmark on GPU using rocBLAS with and without the data transfer (on AMD)
- Benchmark on GPU using CUDA with and without the data transfer (on NVIDIA)
- Benchmark on GPU using cuBLAS with and without the data transfer (on NVIDIA)
- Checker for all these benchmarks

## Script Features

### Bash script

measure.sh :
- Starts a series of executions with the kernels as arguments
- Output can be saved in a text file
- Can generate a graph based on benchmark outputs (only for RDTSC metrics)

check.sh :
- Check all kernel in arguments
- Automatically compares kernels outputs with the basis kernel
- Shows us the pourcentage of similarities between two kernel output

calibrate.sh (TODO) :
- Auto calibrate the number of warmup and repetition of the kernels in argument
- Generate graphs about this calibration

### Python script

generate_graph.py :
- take in argument a matrix size, an output file from the measure script and a output png file
- Generate a graph based on benchmark outputs from the measure script (only for RDTSC metrics)

## Kernel List

On AMD :

- basis 
- cpu_omp 
- cblas 
- gpu_omp 
- gpu_omp_wo_dt 
- hip 
- hip_wo_dt 
- rocblas 
- rocblas_wo_dt

On NVIDIA :

- basis 
- cpu_omp 
- cblas 
- gpu_omp 
- gpu_omp_wo_dt 
- openacc 
- openacc_wo_dt 
- cuda 
- cuda_wo_dt 
- cublas
- cublas_wo_dt 


## Documentation

[Intership report on GPU](https://www.overleaf.com/read/cjpngdgvjckd)

## Usage/Examples

By using the script :

```bash
./script/measure.sh {options} [problem size] <kernels>
```

Use the '-h' option for more information

```bash
./script/check.sh {options} [problem size] <kernels>
```
Use the '-h' option for more information

```
.......''',,,',;::ccccc:;,'............ 
........''';cldkO000KK00Oxoc;'..........
''''''..',cxO0000KKKKKK00000kdc'........
,,,,,,,;lk0Ooccd0KKKKKKK0klcokOd:'......
,,,,,,:x00d'   .:OKKKKK0o.   .lOkl'.....
''''';x00x'     .oK0000d.     .lOko'....
.''',o000l       :00000c       ;kkkl....
....:OK00:       :0K000c       ;kkkd,...
 ..'l0000l      .oKKKKKo.      :kkkx, ..
   .l000Kk,    .c0KKKKKO;     'dkkkk;  .
   .:O0000kl;;:d0KK00000kc'.':xkkkkx;...
....;k0000OOOO00000000000Okxxkkkkkkd,...
....'lOOOOOOOOOO0000000OOOOkxxkkxxxc....
.....,dOOkkOOOOOOO0000OOkkkkkxxxxxl.....
......,dkkOOOOOOOOkxdxOOkkkkkxxxdc.... .
........cxkOkkkkkx:..;dxkkkkOkxo,. .....
.........':odxdddolccloooddddo;.     ...  
............,;cllloooolllc:,..  ........        
..................'''.....  ............        
```