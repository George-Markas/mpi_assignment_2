#define MYUTILS_IMPLEMENTATION
#include "myutils.h"
#include "mpi.h"

#define ROOT_RANK 0 // rank of the main process

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get total process count and calling process rank
    int process_id, process_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    int length; // Length of the vector
    int n_count; // How many numbers the process is going to receive
    int* receive_counts = calloc(process_count, sizeof(int)); // How many numbers each process will receive via Scatterv
    int* displacements = calloc(process_count, sizeof(int)); // Displacements for Scatterv
    float *vector = NULL;

    if(process_id == ROOT_RANK) {
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
    }

    // Inform each process how many numbers it is going to receive
    MPI_Scatter(receive_counts, 1, MPI_INT, &n_count, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD );
    float* receive_buffer = calloc(n_count, sizeof(float));

    // Send each process its predefined portion of the load
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatterv(vector, receive_counts, displacements, MPI_FLOAT, receive_buffer, n_count,
        MPI_INT, ROOT_RANK, MPI_COMM_WORLD);

    //printf("Rank: %d | n_count: %d\n", process_id, n_count);
    MPI_Barrier(MPI_COMM_WORLD);

    // Calculate partition average
    float partition_sum = 0;
    for(int i = 0; i < n_count; i++) {
        partition_sum += receive_buffer[i];
    }

    // DEBUG
    printf("%d) Part sum: %.3f\n", process_id, partition_sum);

    // Gather all the partition sums and sum the up
    float vector_sum, vector_max, vector_min;

    // Using a temporary copy of receive_buffer to avoid undefined behaviour on subsequent reductions
    float* temp = calloc(n_count, sizeof(float));
    memcpy(temp, receive_buffer, n_count * sizeof(float));


    // Get the min, max and the sum of values held by the processes
    MPI_Reduce(receive_buffer, &vector_min, n_count, MPI_FLOAT, MPI_MIN, ROOT_RANK, MPI_COMM_WORLD);
    MPI_Reduce(temp, &vector_max, n_count, MPI_FLOAT, MPI_MAX, ROOT_RANK, MPI_COMM_WORLD);
    MPI_Reduce(&partition_sum, &vector_sum, 1, MPI_FLOAT, MPI_SUM, ROOT_RANK, MPI_COMM_WORLD);

    free(temp);
    temp = NULL;

    // Prints to be moved, here for reference
    float vector_mean = vector_sum / (float) length;
    if(process_id == ROOT_RANK) {
        printf("Min: %.3f\n", vector_min);
        printf("Max: %.3f\n", vector_max);
        printf("Mean: %.3f\n", vector_mean);
    }

    // float variance_subpart = 0;
    // for(int i = 0; i < n_count; i++) {
    //     float x = receive_buffer[i] - vector_mean;
    //     printf("pow %.3f\n", x);
    //     variance_subpart += x * x;
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