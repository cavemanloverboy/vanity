#ifndef UTILS_H
#define UTILS_H

extern "C" void gpu_init(int id);
extern "C" int get_gpu_count();

extern int num_multi_processors;
extern int num_blocks;
extern int num_threads;
extern int num_gpus;
extern int max_threads_per_mp;
extern unsigned long long int target_cycles;

// Greatest common denominator
// Used in gpu_init() to calculate block_size
int gcd(int a, int b);

#endif