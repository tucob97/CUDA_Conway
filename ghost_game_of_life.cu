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


__global__ void compute_next_generation(int *d_grid, int *d_new_grid, int N) {
 
	// global indices
    int x = blockIdx.x * blockDim.x + threadIdx.x; //column
    int y = blockIdx.y * blockDim.y + threadIdx.y; //row
	
	int idx = y * (N+2) + x;// cell index 
	
	// initialize neighbors counter
	int live_neighbors = 0;
	
	// exclude Ghost cell
	if (x > 0 && x < N+1 && y > 0 && y < N+1) {
		
		// Sum all the live neighbors
        live_neighbors += d_grid[(y-1) * (N+2) + (x-1)]; // Top-left
        live_neighbors += d_grid[(y-1) * (N+2) + (x)];    // Top
        live_neighbors += d_grid[(y-1) * (N+2) + (x+1)]; // Top-right
        live_neighbors += d_grid[(y) * (N+2) + (x-1)];    // Left
        live_neighbors += d_grid[(y)* (N+2) + (x+1)];    // Right
        live_neighbors += d_grid[(y+1) * (N+2) + (x-1)]; // Bottom-left
        live_neighbors += d_grid[(y+1)* (N+2) + (x)];    // Bottom
        live_neighbors += d_grid[(y+1) * (N+2) + (x+1)]; // Bottom-right

		
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
	
	}

}

__global__ void update_counts(int *d_grid, int *d_new_grid, int *d_alive_count, int *d_consecutive_alive_count, int N) {

    int x = blockIdx.x * blockDim.x + threadIdx.x; // column
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row

    int idx = y * (N+2) + x; // cell index
	
	// exclude Ghost cell
	if (x > 0 && x < N+1 && y >0 && y < N+1){
		// Update the counts based on the new grid state
		if (d_new_grid[idx] == 1) {
			d_alive_count[idx]++;  // Increment total alive count
			
			if (d_grid[idx] == 1){
				d_consecutive_alive_count[idx]++;  // Increment consecutive alive count
			}
		}

		// Swap the grids for the next iteration
		d_grid[idx] = d_new_grid[idx];
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
	
	// read input parameter
    const char *filename = argv[1];
    int N = atoi(argv[2]);
    int blockDimX = atoi(argv[3]);
    int blockDimY = blockDimX;    
	if( blockDimX * blockDimY > 1024){
	printf("Error: too much threads in a block (>1024)");
	return 1;
	}
    int generations = atoi(argv[4]);

	//allocate grids
    size_t grid_size = (N+2) * (N+2) * sizeof(int);
	size_t grid_size_read = (N) * (N) * sizeof(int);
	
    int *h_grid_2 = (int *)malloc(grid_size_read);
	int *h_grid = (int *)malloc(grid_size);
    int *h_alive_count = (int *)malloc(grid_size);
    int *h_consecutive_alive_count = (int *)malloc(grid_size);

    read_initial_state(filename, h_grid_2, N);


	// Initialize counts
    for (int y = 0; y < N+2; ++y) {
        for (int x = 0; x < N+2; ++x) {
			// real cell
			if(x<N && y<N)
			{
				h_grid[(y+1) * (N+2) + x+1] = h_grid_2[y * N + x];
				h_alive_count[(y+1) * (N+2) + x+1] = h_grid_2[y * N + x];// cell initially alive 
				h_consecutive_alive_count[(y+1) * (N+2) + x+1] = 0;  // No consecutive counts initially
			}
			//Ghost cell
			if(x==0 || y==0 || x==N+1 || y==N+1){
				h_grid[y * (N+2) + x] = 0 ;
				h_alive_count[y * (N+2) + x] = 0;// ghost cell
				h_consecutive_alive_count[y * (N+2) + x] = 0;  // ghost cell
			}
        }
    }	
	


	//allocate device grids
    int *d_grid, *d_new_grid, *d_alive_count, *d_consecutive_alive_count;
    gpuErrchk(cudaMalloc(&d_grid, grid_size));
    gpuErrchk(cudaMalloc(&d_new_grid, grid_size));
    gpuErrchk(cudaMalloc(&d_alive_count, grid_size));
    gpuErrchk(cudaMalloc(&d_consecutive_alive_count, grid_size));

    gpuErrchk(cudaMemcpy(d_grid, h_grid, grid_size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_new_grid, h_grid, grid_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_alive_count, h_alive_count, grid_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_consecutive_alive_count, h_consecutive_alive_count, grid_size, cudaMemcpyHostToDevice));
	
	// CUDA Block Grid dimension 
    dim3 blockDim(blockDimX, blockDimY);
    dim3 gridDim((N+2 + blockDim.x - 1) / blockDim.x, (N+2 + blockDim.y - 1) / blockDim.y);

	printf("CUDA gridDim:= %d x %d",(N+2 + blockDim.x - 1) / blockDim.x  ,(N+2 + blockDim.y - 1) / blockDim.y );
	printf("\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Run the kernel for the specified number of generations
    for (int gen = 0; gen < generations; ++gen) {
        compute_next_generation<<<gridDim, blockDim>>>(d_grid, d_new_grid, N);

        // Synchronize the kernel to ensure all blocks are done
        cudaDeviceSynchronize();

        // Update the counts after the new generation is computed
        update_counts<<<gridDim, blockDim>>>(d_grid, d_new_grid, d_alive_count, d_consecutive_alive_count, N);

        // Synchronize the kernel again to ensure the counts are updated before the next generation
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time spent on (GHOST) computation of %d x %d grid: %f ms\n", N, N, milliseconds);

	// copy result to host
    gpuErrchk(cudaMemcpy(h_grid, d_grid, grid_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_alive_count, d_alive_count, grid_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_consecutive_alive_count, d_consecutive_alive_count, grid_size, cudaMemcpyDeviceToHost));
	
	
	
	// Check for "--verbose" flag
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
            break;
        }
    }
	// If verbose flag is set, write the final result
    if (verbose) {
		// Output file
		FILE *output_file = fopen("result_ghost.txt", "w");
		if (output_file == NULL) {
			perror("Error opening result.txt");
			return 1;
		}
	

		// Write the final grid state to the file
		fprintf(output_file, "Final grid state:\n");
		for (int y = 0; y < N; ++y) {
			for (int x = 0; x < N; ++x) {
				fprintf(output_file, "%c ", h_grid[(y+1) * (N+2) + x + 1] ? 'x' : '_');
			}
			fprintf(output_file, "\n");
		}

		// Write the cell statistics to the file
		fprintf(output_file, "Cell statistics:\n");
		for (int y = 0; y < N; ++y) {
			for (int x = 0; x < N; ++x) {
				int idx = (y+1) * (N+2) + x + 1;
				fprintf(output_file, "Cell (%d,%d): Total Alive: %d, Consecutive Alive: %d\n", x, y, h_alive_count[idx], h_consecutive_alive_count[idx]);
			}
		}

		// Close the output file
		fclose(output_file);
	}
	
	save_results("data_ghost.txt", N, milliseconds, blockDimX, gridDim.x, generations);
	
	//free memory
    cudaFree(d_grid);
    cudaFree(d_new_grid);
    cudaFree(d_alive_count);
    cudaFree(d_consecutive_alive_count);
	free(h_grid_2);
    free(h_grid);
    free(h_alive_count);
    free(h_consecutive_alive_count);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
