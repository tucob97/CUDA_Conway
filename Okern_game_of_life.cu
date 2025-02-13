#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// We have to know if there is a runtime error
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void compute_next_generation(int *d_grid, int *d_new_grid, int *d_alive_count, int *d_consecutive_alive_count, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row

    int idx = y * N + x; // cell index
	// initialize neighbors counter
    int live_neighbors = 0;
	// Ensure thread is within the bounds of grid
	if (x < N && y < N){
	//printf("_Cell_num-%d,%d,%d-_\n", idx, x, y);
		// Calculate the number of live neighbors
		for (int i = -1; i <= 1; ++i) {
			for (int j = -1; j <= 1; ++j) {
				if (i == 0 && j == 0) continue;
				int nx = x + i;
				int ny = y + j;
				// Boundary condition handling
				if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
					//printf("_live_neigh_-%d,%d,%d-_\n", idx, nx, ny);
					live_neighbors += d_grid[ny * N + nx];
					
				}
				
			}
		}

		// Apply the Game of Life rules
		if (d_grid[idx] == 1) {  // Cell is alive
			if (live_neighbors >= 4 || live_neighbors <= 1) {
				d_new_grid[idx] = 0;  // Cell dies
			} else {
				d_new_grid[idx] = 1;  // Cell survives
			}
		} else {  // Cell is dead
			if (live_neighbors == 3) {
				d_new_grid[idx] = 1;  // Cell becomes alive
			} else {
				d_new_grid[idx] = 0;  // Cell remains dead
			}
		}
		// Update the counts based on the new grid state
		if (d_new_grid[idx] == 1) {
			d_alive_count[idx]++;  // Increment total alive count
			
			if (d_grid[idx] == 1){
				d_consecutive_alive_count[idx]++;  // Increment consecutive alive count
			}
		}
	}
}


void read_initial_state(const char *filename, int *h_grid, int N) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < N * N; ++i) {
        fscanf(file, "%d", &h_grid[i]);
    }

    fclose(file);
}


void save_results(const char *filename, int N, double execution_time, int blockDim, int gridDim, int generations) {
    FILE *file = fopen(filename, "a"); // Open file in append mode 
    if (file == NULL) {
		FILE *file = fopen(filename, "w"); // Open file in write mode (create file)
			if (file == NULL) {
			perror("Failed to open file");
			exit(EXIT_FAILURE);
			}
    }

    // Write variables to the file on a single line
    fprintf(file, "%d %.3f %d %d %d\n",
            N, execution_time, blockDim, gridDim, generations);

    fclose(file); // Close the file
}


// MAIN
int main(int argc, char *argv[]) {
    if (argc == 0) {
        fprintf(stderr, "Input Error - Usage: game_of_life.exe <initial_state_file> <N> <blockDim> <generations>\n", argv[0]);
        return 1;
    }
	// read input parameters
    const char *filename = argv[1];
    int N = atoi(argv[2]);
    int blockDimX = atoi(argv[3]);
    int blockDimY = blockDimX;    //atoi(argv[4]);
	if( blockDimX * blockDimY > 1024){
	printf("Error: too much threads in a block (>1024)");
	return 1;
	}
    int generations = atoi(argv[4]);

	// allocate grid 
    size_t grid_size = N * N * sizeof(int);
    int *h_grid = (int *)malloc(grid_size);
    int *h_alive_count = (int *)malloc(grid_size);
    int *h_consecutive_alive_count = (int *)malloc(grid_size);

    read_initial_state(filename, h_grid, N);

	// Initialize counts
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            h_alive_count[y * N + x] = h_grid[y * N + x];// cell initially alive 
			h_consecutive_alive_count[y * N + x] = 0;  // No consecutive counts initially
        }
    }	
	
	// allocate device grids
    int *d_grid, *d_new_grid, *d_alive_count, *d_consecutive_alive_count;
    gpuErrchk(cudaMalloc(&d_grid, grid_size));
    gpuErrchk(cudaMalloc(&d_new_grid, grid_size));
    gpuErrchk(cudaMalloc(&d_alive_count, grid_size));
    gpuErrchk(cudaMalloc(&d_consecutive_alive_count, grid_size));

    gpuErrchk(cudaMemcpy(d_grid, h_grid, grid_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_alive_count, h_alive_count, grid_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_consecutive_alive_count, h_consecutive_alive_count, grid_size, cudaMemcpyHostToDevice));

	// CUDA Block Grid dimension 
    dim3 blockDim(blockDimX, blockDimY);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

	printf("CUDA gridDim:= %d x %d",(N + blockDim.x - 1) / blockDim.x  ,(N + blockDim.y - 1) / blockDim.y );
	printf("\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Run the kernel for the specified number of generations
    for (int gen = 0; gen < generations; ++gen) {
        compute_next_generation<<<gridDim, blockDim>>>(d_grid, d_new_grid, d_alive_count, d_consecutive_alive_count, N);

        // Synchronize the kernel to ensure all blocks are done
        cudaDeviceSynchronize();
	
		// Swap pointers on host
        int* temp = d_grid;
        d_grid = d_new_grid;
        d_new_grid = temp;
		
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time spent on (One Kernel) computation of %d x %d grid: %f ms\n", N, N, milliseconds);
    
    cudaMemcpy(h_grid, d_grid, grid_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_alive_count, d_alive_count, grid_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_consecutive_alive_count, d_consecutive_alive_count, grid_size, cudaMemcpyDeviceToHost);
	
	
	
	// Check for "--verbose" flag
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
            break;
        }
    }
	// If verbose flag is set, print the file contents
    if (verbose) {
		// Output file
		FILE *output_file = fopen("result_Okern.txt", "w");
		if (output_file == NULL) {
			perror("Error opening result_Okern.txt");
			return 1;
		}
	

		// Write the final grid state to the file
		fprintf(output_file, "Final grid state:\n");
		for (int y = 0; y < N; ++y) {
			for (int x = 0; x < N; ++x) {
				fprintf(output_file, "%c ", h_grid[y * N + x] ? 'x' : '_');
			}
			fprintf(output_file, "\n");
		}

		// Write the cell statistics to the file
		fprintf(output_file, "Cell statistics:\n");
		for (int y = 0; y < N; ++y) {
			for (int x = 0; x < N; ++x) {
				int idx = y * N + x;
				fprintf(output_file, "Cell (%d,%d): Total Alive: %d, Consecutive Alive: %d\n", x, y, h_alive_count[idx], h_consecutive_alive_count[idx]);
			}
		}

		// Close the output file
		fclose(output_file);
	}
	
	save_results("data_Okern.txt", N, milliseconds, blockDimX, gridDim.x, generations);
	
	//free memory
    cudaFree(d_grid);
    cudaFree(d_new_grid);
    cudaFree(d_alive_count);
    cudaFree(d_consecutive_alive_count);
    free(h_grid);
    free(h_alive_count);
    free(h_consecutive_alive_count);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
