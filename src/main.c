#define MYUTILS_IMPLEMENTATION
#include <math.h>
#include "myutils.h"
#include "mpi.h"

#define ROOT_RANK 0 // rank of the main process

struct delta_element {
    float num;
    int index;
};

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
        // printf("Base: %d\nRem: %d\n\n", base_partition, remainder);
        //
        // puts("Load:");
        // for(int i = 0; i < process_count; i++) {
        //     fflush(stdout);
        //     printf("P%d | %d\n", i, receive_counts[i]);
        // }
        // putc('\n', stdout);

    }

    // Inform each process how many numbers it is going to receive
    MPI_Scatter(receive_counts, 1, MPI_INT, &n_count, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD );
    float* receive_buffer = calloc(n_count, sizeof(float));

    // Send each process its predefined portion of the load
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatterv(vector, receive_counts, displacements, MPI_FLOAT, receive_buffer, n_count,
        MPI_INT, ROOT_RANK, MPI_COMM_WORLD);

    // DEBUG
    //printf("Rank: %d | n_count: %d\n", process_id, n_count);
    MPI_Barrier(MPI_COMM_WORLD);

    // Calculate partition average
    float partition_sum = 0;
    for(int i = 0; i < n_count; i++) {
        partition_sum += receive_buffer[i];
    }

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

    // Calculate mean
    float vector_mean;
    if(process_id == ROOT_RANK)
        vector_mean = vector_sum / (float) length;

    // Sending the min, max and mean to non-root processes to be used for later calcualtions
    MPI_Bcast(&vector_min, 1, MPI_FLOAT, ROOT_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&vector_max, 1, MPI_FLOAT, ROOT_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&vector_mean, 1, MPI_FLOAT, ROOT_RANK, MPI_COMM_WORLD);

    // Variance subsection calculation, aka (Xn - m)^2, for all the values in the receive_buffer of each process
    float variance_subsection = 0;
    for(int i = 0; i < n_count; i++) {
        float x = receive_buffer[i] - vector_mean;
        // DEBUG
        // printf("p%d: %.3f - %.3f = %.3f\n", process_id, receive_buffer[i], vector_mean, x);
        variance_subsection += x * x;
    }

    float variance;
    // Gathering back all the subsections and summing them to divide the resulting sum by n
    MPI_Reduce(&variance_subsection, &variance, 1, MPI_FLOAT, MPI_SUM, ROOT_RANK, MPI_COMM_WORLD);
    if(process_id == ROOT_RANK) {
        variance = variance / (float) length;
        printf("Variance: %.3f\n", variance);
    }

    // DEBUG
    //printf("p%d | Min: %.3f | Max: %.3f\n", process_id, vector_min, vector_max);

    // Calculating the value of each element of the delta vector, as well as their offsets
    struct delta_element* delta_subsection = malloc(sizeof(struct delta_element) * n_count);
    for(int i = 0; i < n_count; i++) {
        delta_subsection[i].num = ((receive_buffer[i] - vector_min) / (vector_max - vector_min)) * 100;
        delta_subsection[i].index = process_id + (length * i);
    }

    // DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    printf("p%d: d = %.3f | idx = %d\n", process_id, delta_subsection[0].num, delta_subsection[0].index);

    /* DEBUG FROM HERE ONWARDS */

    // To store the subsections of delta elements gathered from the processes
    // struct delta_element* gather_subsections = malloc(sizeof(struct delta_element) * length);
    //
    // // Create MPI datatype for delta_element
    // MPI_Datatype delta_struct;
    // int blocklengths[] = {1, 1};
    // MPI_Datatype types[] = {MPI_FLOAT, MPI_INT};
    // MPI_Aint offsets[2];
    //
    // // Calculate offsets
    // MPI_Aint base_address, address_1, address_2;
    // struct delta_element dummy;
    //
    // MPI_Get_address(&dummy, &base_address);
    // MPI_Get_address(&dummy.num, &address_1);
    // MPI_Get_address(&dummy.index, &address_2);
    //
    // offsets[0] = address_1 - base_address;
    // offsets[1] = address_2 - base_address;
    //
    // // Create and commit the datatype
    // MPI_Type_create_struct(2, blocklengths, offsets, types, &delta_struct);
    // MPI_Type_commit(&delta_struct);
    //
    // MPI_Gather(delta_subsection, n_count, delta_struct, gather_subsections, n_count,
    //     delta_struct, ROOT_RANK, MPI_COMM_WORLD);
    //
    // float* delta = calloc(length, sizeof(float));
    // if(process_id == ROOT_RANK) {
    //     for(int i = 0; i < length; i++) {
    //         delta[gather_subsections[i].index] = gather_subsections[i].num;
    //     }
    //
    //     for(int i = 0; i < length; i++) {
    //         printf("%.3f ", delta[i]);
    //     }
    // }

    free(receive_counts);
    receive_counts = NULL;

    free(displacements);
    displacements = NULL;

    free(vector);
    vector = NULL;

    free(receive_buffer);
    receive_buffer = NULL;

    // free(delta);
    // delta = NULL;
    //
    // free(gather_subsections);
    // gather_subsections = NULL;

    free(delta_subsection);
    delta_subsection = NULL;

    MPI_Finalize();
    return EXIT_SUCCESS;
}