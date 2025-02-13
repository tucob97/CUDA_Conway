#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>



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


__global__ void compute_next_generation(int *d_grid, int *d_new_grid, int *d_alive_count, int *d_consecutive_alive_count, int N, int border,int rows) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	//int half_N= N/2;
    int idx = y * N + x;
    int live_neighbors = 0;
    //printf("_Cell_num_Before_IF-%d,%d,%d--_Thread_ID==%d_\n", idx, x, y,border);
	if (x < N  && y >= border && y < rows+border){
	//printf("_Cell_num-%d,%d,%d--_Thread_ID==%d_\n", idx, x, y,border);
		// Calculate the number of live neighbors
		for (int i = -1; i <= 1; ++i) {
			for (int j = -1; j <= 1; ++j) {
				if (i == 0 && j == 0) continue;
				int nx = x + i;
				int ny = y + j;
				// Boundary condition handling
				if (nx >= 0 && nx < N && ny >= 0 && ny < rows+1) {
					//printf("_live_neigh_-%d,%d,%d- Thread N.%d_\n", idx, nx, ny,border);
					live_neighbors += d_grid[ny * N + nx];
					
				}
				//printf("%d\n", live_neighbors);
				
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


void run_game_of_life_multi_gpu(int *h_grid,int *h_alive_count, int *h_consecutive_alive_count, int N, int generations, int blockDimX, int verbose) {
    
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    if (num_gpus < 2) {
        fprintf(stderr, "This program requires at least 2 GPUs\n");
        exit(EXIT_FAILURE);
    }
	
	// rows to GPU
	int blockDimY= blockDimX;
    int half_N = N / 2;
    int extra_rows = N % 2;        // extra row to assign to gpu_0
	
	// rows per GPU
	int rows_gpu[2];
	rows_gpu[0]= half_N + extra_rows;
	rows_gpu[1]= half_N;
	//printf("r_0 == %d, r_1 ==%d \n", rows_gpu[0], rows_gpu[1]);		
	// Array to hold grid sizes
    size_t grid_size[2];

    // Assign values to each grid size in a loop
    grid_size[0] = N * (rows_gpu[0] + 1) * sizeof(int); // Extra row for boundary exchange
    grid_size[1] = N * (half_N + 1) * sizeof(int);     // Extra row for boundary exchange

	// initialize subgrids
    int *d_grid[2], *d_new_grid[2], *d_alive_count[2], *d_consecutive_alive_count[2];
	
	
	//#pragma omp parallel for num_threads(2)
    for (int i = 0; i < 2; ++i) {
        gpuErrchk(cudaSetDevice(i));
        gpuErrchk(cudaMalloc(&d_grid[i], grid_size[i]));
        gpuErrchk(cudaMalloc(&d_new_grid[i], grid_size[i]));
		gpuErrchk(cudaMalloc(&d_alive_count[i], grid_size[i]));
        gpuErrchk(cudaMalloc(&d_consecutive_alive_count[i], grid_size[i]));
        gpuErrchk(cudaMemset(d_consecutive_alive_count[i], 0, grid_size[i]));
		
        if (i == 0) {
            // GPU 0: Copy its portion of the grid plus the boundary 
            gpuErrchk(cudaMemcpy(d_grid[0], h_grid, N * (rows_gpu[0]+1) * sizeof(int), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(d_alive_count[0], h_grid, N * (rows_gpu[0]+1) * sizeof(int), cudaMemcpyHostToDevice));
           
        } else {
            // GPU 1: Copy its portion of the grid plus the boundary 
            gpuErrchk(cudaMemcpy(d_grid[1], h_grid + N * (rows_gpu[0]-1), N * (half_N+1) * sizeof(int), cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(d_alive_count[1], h_grid + N * (rows_gpu[0]-1), N * (half_N+1) * sizeof(int), cudaMemcpyHostToDevice));
			
        }
    }

    dim3 blockDim(blockDimX, blockDimY);
	dim3 gridDims[2];
	
	// calculate and assign grid dimension
	gridDims[0] = dim3((N + blockDim.x - 1) / blockDim.x, (rows_gpu[0]+1 + blockDim.y - 1) / blockDim.y);
    gridDims[1] = dim3((N + blockDim.x - 1) / blockDim.x, (half_N+1 + blockDim.y - 1) / blockDim.y);
	
	//printf("grid_dims[0](x,y)==%d,%d [1](x,y)==%d,%d \n", gridDims[0].x, gridDims[0].y, gridDims[1].x, gridDims[1].y);	
	int gen=0;

    float milliseconds;

    cudaEvent_t start, stop;
    
	#pragma omp parallel private(gen)
	{
		int tid = omp_get_thread_num();  // Each thread will get its own device
		//printf("Thread ID == %d\n", tid);
		cudaSetDevice(tid);  // Set the device based on the thread number
		
		if(tid==0){
		cudaEventCreate(&start);
    		cudaEventCreate(&stop);
    		cudaEventRecord(start);
		}


		for (gen = 0; gen < generations; ++gen) {

			// Compute the next generation for the main part of the grid
			compute_next_generation<<<gridDims[tid], blockDim>>>(d_grid[tid], d_new_grid[tid], d_alive_count[tid], d_consecutive_alive_count[tid], N, tid, rows_gpu[tid]);
			cudaDeviceSynchronize();

			// Swap grids for the next iteration
			int *temp = d_grid[tid];
			d_grid[tid] = d_new_grid[tid];
			d_new_grid[tid] = temp;

			// Synchronize threads before exchanging boundaries
			#pragma omp barrier

			// Only one thread performs the boundary exchange
			#pragma omp single
			{
				// Exchange boundary rows between GPUs
				// send below row border from GPU 0 to GPU 1 
				cudaMemcpyPeer(d_grid[1],1, d_grid[0] + (rows_gpu[0] - 1) * N, 0, N * sizeof(int));
				// send top row border from GPU 1 to GPU 0
				cudaMemcpyPeer(d_grid[0] + rows_gpu[0] * N, 0, d_grid[1] + N, 1, N * sizeof(int));
			}

			// Synchronize threads after the boundary exchange
			#pragma omp barrier
		}

	 if(tid==0){
	  cudaEventRecord(stop);
      cudaEventSynchronize(stop);
	
	  //float tid_mill;
      cudaEventElapsedTime(&milliseconds, start, stop);
	  //milliseconds= tid_mill;	

       	  cudaEventDestroy(start);
          cudaEventDestroy(stop);	
         }

	}

    
    printf("Time spent on computation(2_MULTI GPU): %f ms\n", milliseconds);

	// can be parallelize but does not influence performance
	//#pragma omp parallel for num_threads(2)
    // Copy the results back to the host
    for (int i = 0; i < 2; ++i) {
        cudaSetDevice(i);
        cudaMemcpy(h_grid + i * N * rows_gpu[0], d_grid[i] + N*i, N * rows_gpu[i] * sizeof(int), cudaMemcpyDeviceToHost);
	    cudaMemcpy(h_alive_count + i * N * rows_gpu[0], d_alive_count[i] + N*i, N * rows_gpu[i] * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_consecutive_alive_count + i * N * rows_gpu[0], d_consecutive_alive_count[i] + N*i, N * rows_gpu[i] * sizeof(int), cudaMemcpyDeviceToHost);
    }

	// If verbose flag is set, print the file contents
    if (verbose) {
		// Output file
		FILE *output_file = fopen("result_2_Multi.txt", "w");
		if (output_file == NULL) {
			perror("Error opening result_2_Multi.txt");
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



    save_results("data_2_Multi.txt", N, milliseconds, blockDimX, -1, generations);

	
    // Free resources
    for (int i = 0; i < 2; ++i) {
        cudaFree(d_grid[i]);
        cudaFree(d_new_grid[i]);
		cudaFree(d_alive_count[i]);
		cudaFree(d_consecutive_alive_count[i]);
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

	// initialize grids
    size_t grid_size = N * N * sizeof(int);
    int *h_grid = (int *)malloc(grid_size);
    int *h_alive_count = (int *)malloc(grid_size);
    int *h_consecutive_alive_count = (int *)malloc(grid_size);

    omp_set_num_threads(2);
    read_initial_state(filename, h_grid, N);

	// Initialize counts
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            h_alive_count[y * N + x] = h_grid[y * N + x];// cell initially alive 
			h_consecutive_alive_count[y * N + x] = 0;  // No consecutive counts initially
        }
    }	
	// Check for "--verbose" flag
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
            break;
        }
    }
	
	
	// run simulation
	run_game_of_life_multi_gpu( h_grid, h_alive_count, h_consecutive_alive_count, N, generations, blockDimX, verbose);
	
	//free memory
    free(h_grid);
    free(h_alive_count);
    free(h_consecutive_alive_count);
	
    return 0;
}
