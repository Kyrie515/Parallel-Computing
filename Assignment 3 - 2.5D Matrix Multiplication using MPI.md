## Assignment 3 - 2.5D Matrix Multiplication using MPI

#### <center>Chengyuan Li</center>

### 1. Project Files

```bash
project_directory
  ├── 2.5mm.cpp     \\The main file of the algorithm
  ├── Makefile			
  ├── README				
  ├── any.sh					
  ├── environment.sh
  ├── hello.cpp
  ├── interactive.sh
  ├── landing.sh		\\ sh file help setup the env
  ├── reduce_avg.cpp
  ├── reservation.sh
  ├── scavenge.sh
  ├── submit.sbatch
  └── test.cpp
```

### 2. Algorithm

This project implements the 2.5D matrix multiplication algorithm presented in the paper *Communication-optimal parallel 2.5D matrix multiplication and LU factorization algorithms* .This algorithm combines the 3D matrix production algorithm and the Cannon algorithm. By duplicating the matrix on each plane along the Z direction (with the size of `c`) to reduce the complexity.

### Bandwidth and Latency Lower Bounds

In the alignment step, the algorithm first shifts the matrix `A` and `B` based on the index of the plane in the Z direction. The bandwidth cost `W` and latency cost `S` are $ W=\Omega(\frac{n^2}{\sqrt{p*c}}) $ and $ S=\Omega(\sqrt\frac{p}{c^3}) $ .

In the corner case where `c=1`, the algorithm degrade to a normal cannon algorithm with high bandwidth cost and latency cost. In the Z direction extension of matrix replication, use more memory to leverage the communication cost since the loop steps reduced from $ \sqrt{p} $ to $ \sqrt{\frac{p}{c^3}} $.

### Implementation

Let us start with an example: ![截屏2024-04-23 09.36.38](/Users/kyrie/Library/Application Support/typora-user-images/截屏2024-04-23 09.36.38.png)

the matrix size of `A`,`B` and `C` is `4*4` and 4 MPI processes on each plane. The submatrices should be distributed to 4 processes as divided by the red grid. And we create variables like `pMatrixA` for the local matrix of each process.

So according to the algorithm code in the paper, my implementation can be divided into different parts:

#### MPI Environment and Grid Setup

1. **Initialization**:
   - The MPI environment is initialized and the rank and total number of processes are determined.
   - Command line arguments are used to set the matrix size (`N`) and the replication factor (`C`).
2. **Cartesian Communicator**:
   - A 3D Cartesian topology (`cartComm`) is created using dimensions based on the square root of `(procs / C)` for the first two dimensions, and `C` for the third. This topology aids in logical process arrangement and communication.
3. **Sub-Matrix Size**:
   - Each process computes the size of its sub-matrix as `subN = N / sqrt(procs / C)`, determining how the original matrices are split among processes.

#### Matrix Allocation and Initialization

1. **Matrix Allocation**:
   - Each process allocates memory for its portion of matrices `A`, `B`, `C`, and additional matrices for temporary storage during computations.
2. **Initial Setup**:
   - If the process resides in the first layer (`cartCoords[2] == 0`), it initializes its sub-matrices `A` and `B`. Matrix `A` is filled based on a formula that varies with its indices, and `B` is initialized to 1. The expected result matrix is set for verification purposes.

#### Communication for Data Distribution

1. **Broadcasting**:
   - Data necessary for computation is broadcasted within each layer, ensuring all processes in the same layer have the required data blocks.
2. **Shifting Data for Computation**:
   - The code performs `MPI_Sendrecv` to correctly rotate sub-matrices of `A` and `B` among processes. This step is crucial for aligning the blocks for local multiplication per the 2.5D algorithm requirements.
   - Additional data shifting occurs cyclically among processes to facilitate different stages of the multiplication.

#### Matrix Multiplication

1. **Local Computation**:
   - The `MatrixMul` function is called to multiply local sub-matrices. This function implements a straightforward cubic complexity multiplication algorithm for the sub-matrices held by each process.
2. **Iterative Computation**:
   - For multiple rounds determined by the grid dimension along the third axis divided by `C`, matrices are shifted and locally multiplied in each iteration to progressively build the final sub-matrix of `C`.

#### Final Reduction and Verification

1. **Summation of Results**:
   - `MPI_Reduce` is used to sum up all local sub-matrices of `C` across the layers to form the final matrix `C`.
2. **Checking Results**:
   - The main process checks the final results against expected values to verify the correctness of the computation.

#### Timing and Clean-up

1. **Performance Timing**:
   - The execution time is measured and output by the main process to evaluate performance.
2. **Resource Clean-up**:
   - MPI communicators are freed and the MPI environment is finalized to release resources.

### Experiments and Results

#### Parameter Configuration

When configuring the hyperparameter of total processors number `P` and the Z dimension size `c`, restrained by the following constrains:
$$
\frac{P}{c}, \sqrt{\frac{P}{c^3}}, \frac{n * c}{P} \in \real
$$
So after considering this, I use a parameter like `n = 7560` rather than `7500`.

| P    | Plane Processors | C    | Collapse Time(s) |
| ---- | ---------------- | ---- | ---------------- |
| 4    | 2 * 2 = 4        | 1    | 770.4            |
| 9    | 3 * 3 = 9        | 1    | 220.3            |
| 8    | 2 * 2 = 4        | 2    | 385.3            |
| 32   | 4 * 4 = 16       | 2    | 37.25            |
| 27   | 3 * 3 = 9        | 3    | 31.73            |
| 64   | 4 * 4 = 16       | 4    | 16.44            |

#### Collective Communication

It is obvious that I use Collective Communication in several ways: To make a replica on each plane, I use `MPI_Bcast` to broadcast the data from the front plane to all the planes along the Z direction. At the end of the matrix multiplication, `MPI_Reduce` is used to collect and sum up all the computed sub-matrix sections of `C` from each process in the split communicator (`splitComm`). The root process in this communicator gathers these sums to assemble the complete matrix `C`. This operation is crucial to obtain the final results of the distributed matrix multiplication.

### Jumpshot

`P=3*3*3=27` with `c=3`, first set `n=756` to have a close look at each steps:

![截屏2024-04-23 11.19.21](/Users/kyrie/Library/Application Support/typora-user-images/截屏2024-04-23 11.19.21.png)

Through the screeshot we can see:

1. **Communication**: The lines connecting the processes represent communication between them. These are typically MPI send and receive calls. The pattern of the lines indicates how the data is being transferred between processes. In your screenshot, there is a dense mesh of lines at the beginning, indicating a lot of initial communication between processes. This could correspond to a setup phase where data is being distributed to the processes.
2. **Time Synchronization**: The red dots along the timelines indicate synchronization points between processes. This could be where collective communication operations force processes to wait until all processes reach the same point, such as with `MPI_Barrier`, `MPI_Bcast`, or `MPI_Reduce`.
3. **Execution Time**: The horizontal axis represents time. This timeline allows you to see the start and finish of different operations and to measure the time spent on each operation.

### HPCToolKit

For given parameter `p=4*4*2=32, c=2, n=756`, we can see the hpctoolkit figure:![1](/Users/kyrie/Desktop/1.png)

As we can see in above, the timeline bar indicates the running time. We can see from the timeline bar that the read operation costs much time for my program.