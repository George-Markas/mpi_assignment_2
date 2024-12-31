#define MYUTILS_IMPLEMENTATION
#include "myutils.h"
#include "mpi.h"

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Main process
    int mpi_root = 0;

    // Get the total process count and the calling process rank
    int process_id, process_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    int* send_counts = calloc(process_count, sizeof(int));
    int* displacements = calloc(process_count, sizeof(int));
    float *vector = NULL;

    if(process_id == mpi_root) {
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

        int base_partition = length / process_count;
        int remainder = length % process_count;

        // Set the load for each process to a base split
        for(int i = 0; i < process_count; i++) {
            send_counts[i] = base_partition;
        }

        // If the load isn't perfectly divisible, distribute the remaining amongst some processes
        int k = remainder;
        if(k) {
            for(int i = 0; i < process_count; i++) {
                send_counts[i]++;
                k--;
                if(!k) // stop when all remaining numbers have been distributed
                    break;
            }
        }

        // Displacements calculation for MPI_Scatterv()
        for(int i = 1; i < process_count; i++) {
            displacements[i] = displacements[i - 1] + send_counts[i - 1];
        }

        // DEBUG
        puts("========= DEBUG =========");
        printf("Base: %d\nRem: %d\n\n", base_partition, remainder);

        puts("Load:");
        for(int i = 0; i < process_count; i++) {
            fflush(stdout);
            printf("P%d | %d\n", i, send_counts[i]);
        }
        puts("=========================");

        free(vector);
        vector = NULL;
    } // end of main process logic

    // Inform each process how many numbers it is going to receive
    int n_count;
    MPI_Scatter(send_counts, 1, MPI_INT, &n_count, 1, MPI_INT, mpi_root, MPI_COMM_WORLD );
    float* receive_buffer = calloc(n_count, sizeof(float));

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatterv(vector, send_counts, displacements, MPI_FLOAT, receive_buffer, n_count,
         MPI_INT, mpi_root, MPI_COMM_WORLD);

    // DEBUG
    // for(int i = 0; i < n_count; i++) {
    //     printf("Rank: %d | %f\n", process_id, receive_buffer[i]);
    // }

    free(send_counts);
    send_counts = NULL;

    free(displacements);
    displacements = NULL;

    free(receive_buffer);
    receive_buffer = NULL;

    MPI_Finalize();
    return EXIT_SUCCESS;
}