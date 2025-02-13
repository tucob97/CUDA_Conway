#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void generate_initial_state(const char *filename, int N) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
	
    // Use the current time as the seed
    srand(time(0));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            fprintf(file, "%d ", rand() % 2);  // Randomly set cell state (0 or 1)
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <filename> <N>\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    int N = atoi(argv[2]);

    generate_initial_state(filename, N);

    return 0;
}
