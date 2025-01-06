#define MYUTILS_IMPLEMENTATION
#include "myutils.h"
#include "mpi.h"

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get total process count and calling process rank
    int process_id, process_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    const int mpi_root = 0; // Main process
    int n_count; // How many numbers the process is going to receive
    int* receive_counts = calloc(process_count, sizeof(int)); // How many numbers each process will receive via Scatterv
    int* displacements = calloc(process_count, sizeof(int)); // Displacements for Scatterv
    float *vector = NULL;

    if(process_id == mpi_root) { // Start of main process logic
        int length;
        fputs("Input the vector's length: ",  stdout);
        fflush(stdout);
        read_int(&length, 1, 0);
        putc('\n', stdout);
        vector = malloc(length * sizeof(float));

        for(int i = 0; i < length; i++) {
            // For grammatical consistency
            char* nth = NULL;
            switch(i) {
                case 0:
                    nth = "st";
                break;
                case 1:
                    nth = "nd";
                break;
                case 2:
                    nth = "rd";
                break;
                default:
                    nth = "th";
                break;
            }

            printf("Input %d%s element (%d/%d): ",i + 1, nth, i + 1, length );
            fflush(stdout);
            read_float(&vector[i], 1, 1);
            putc('\n', stdout);
        }

        const int base_partition = length / process_count;
        const int remainder = length % process_count;

        // Set the load for each process to a base split
        for(int i = 0; i < process_count; i++) {
            receive_counts[i] = base_partition;
        }

        // If the load isn't perfectly divisible, distribute the remainder amongst some processes
        int k = remainder;
        int j = -1;
        while(k) {
            receive_counts[j = (j + 1) % process_count]++; // var = (var + 1) % cap increments till cap and wraps around
            k--;
        }

        // Displacements calculation for Scatterv
        for(int i = 1; i < process_count; ++i) {
            displacements[i] = displacements[i - 1] + receive_counts[i - 1];
        }

        // DEBUG
        printf("Base: %d\nRem: %d\n\n", base_partition, remainder);

        puts("Load:");
        for(int i = 0; i < process_count; i++) {
            fflush(stdout);
            printf("P%d | %d\n", i, receive_counts[i]);
        }
        putc('\n', stdout);
    } // End of main process logic

    // Inform each process how many numbers it is going to receive
    MPI_Scatter(receive_counts, 1, MPI_INT, &n_count, 1, MPI_INT, mpi_root, MPI_COMM_WORLD );
    float* receive_buffer = calloc(n_count, sizeof(float));

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatterv(vector, receive_counts, displacements, MPI_FLOAT, receive_buffer, n_count,
        MPI_INT, mpi_root, MPI_COMM_WORLD);

    printf("Rank: %d | n_count: %d\n", process_id, n_count);
    MPI_Barrier(MPI_COMM_WORLD);

    // DEBUG
    // for(int i = 0; i < n_count; i++) {
    //       printf("%.2f\n", receive_buffer[i]);
    // }

    free(receive_counts);
    receive_counts = NULL;

    free(displacements);
    displacements = NULL;

    free(vector);
    vector = NULL;

    free(receive_buffer);
    receive_buffer = NULL;

    MPI_Finalize();
    return EXIT_SUCCESS;
}