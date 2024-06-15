#include <stdio.h>
#include <stdint.h>
#include "utils.h"

int num_multi_processors;
int num_blocks;
int num_threads;
int num_gpus;
int max_threads_per_mp;
unsigned long long int target_cycles;

// Greatest cmmon denominator
// Used in gpu_init() to calculate block_size
int gcd(int a, int b)
{
    return (a == 0) ? b : gcd(b % a, a);
}

// Initializes gpu parameters. Initializes local variables on the host.
// If using multiple (heterogeneous) gpus, this will overwrite device parameters!
extern "C" void gpu_init(int id)
{
    cudaDeviceProp device_prop;
    int block_size;

    cudaError_t cudaerr = cudaGetDeviceProperties(&device_prop, id);
    if (cudaerr != cudaSuccess)
    {
        printf("getting properties for device failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
        exit(EXIT_FAILURE);
    }

    num_threads = device_prop.maxThreadsPerBlock;
    num_multi_processors = device_prop.multiProcessorCount;
    max_threads_per_mp = device_prop.maxThreadsPerMultiProcessor;
    block_size = (max_threads_per_mp / gcd(max_threads_per_mp, num_threads));
    num_threads = 256; // / block_size;
    num_blocks = block_size * num_multi_processors;
    target_cycles = device_prop.clockRate * 1000 * 5; // clockRate is in kHz, mine for 55 seconds
}