# CUDA_Conway

## Conway's Game of Life in CUDA  

This repository contains multiple implementations of Conway's Game of Life using CUDA. The program leverages GPU parallelism to efficiently compute the evolution of cellular automata. For a detailed explanation of each technique and their performance comparison, see the
presentation.pdf.

### Features  
- GPU-accelerated execution using CUDA  
- Efficient parallel processing for large grids  
- Simple and fast implementation of Conway's Game of Life  
- Uses a square grid with finite borders (no toroidal topology)  

### Requirements  
- CUDA-enabled GPU  
- NVIDIA CUDA Toolkit  (Tested with CUDA v10.1)
- C++ Compiler with CUDA support  

### Compilation & Execution
1. Clone the repository:  
   ```bash
   git clone https://github.com/tucob97/CUDA_Conway.git
   cd CUDA_Conway
   ```
2. Compile and generate NxN random grid of 0 & 1:  
   ```bash
   gcc -o generate_grid generate_grid.c
   ./generate_grid initial_state.txt <N>  #Square grid
   ```
3. Compile using NVCC:  
   ```bash
   nvcc -o game_of_life check_game_of_life.cu  #Or substitute with <version_type>.cu  
   ```
4. Run the program:  
   ```bash
   ./game_of_life <initial_state> <grid-size> <BlockDim> <Num-of-generations> --options
   
   list options
   --verbose #print result in a .txt file with other statistics
   --check #(for "check_game_of_life.cu" version only) check the correctness of the result
   ```  
