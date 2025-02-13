# CUDA_Conway

## Conway's Game of Life in CUDA  

This repository contains an implementation of Conway's Game of Life using CUDA. The program leverages GPU parallelism to efficiently compute the evolution of cellular automata.  

### 🚀 Features  
- GPU-accelerated execution using CUDA  
- Efficient parallel processing for large grids  
- Simple and fast implementation of Conway's Game of Life  
- Uses a square grid with finite borders (no toroidal topology)  

### 🛠️ Requirements  
- CUDA-enabled GPU  
- NVIDIA CUDA Toolkit  
- C++ Compiler with CUDA support  

### 🔧 Compilation & Execution  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```
2. Compile using NVCC:  
   ```bash
   nvcc -o game_of_life main.cu
   ```
3. Run the program:  
   ```bash
   ./game_of_life <initial_state> <grid-size> <BlockDim> <Num-of-generations> --options
   
   list options
   --verbose “print result in a .txt file”
   --check “(for "check_game_of_life.cu" version only) check the correctness of the result”
   ```  
