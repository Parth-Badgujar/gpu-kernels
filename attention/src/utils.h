#ifndef UTILS_INC
#define UTILS_INC
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cooperative_groups.h>

class CUDAtimer {
public :
    cudaEvent_t start; 
    cudaEvent_t stop; 
    CUDAtimer(){
        cudaEventCreate(&this->start); 
        cudaEventCreate(&this->stop); 
    }
    void start_timer(){
        cudaEventRecord(this->start, 0); 
    }
    void stop_timer(){
        cudaEventRecord(this->stop, 0); 
        cudaEventSynchronize(this->stop); 
    }
    float get_time(){
        float ms; 
        cudaEventElapsedTime(&ms, this->start,this->stop); 
        return ms; 
    }
    ~CUDAtimer() {
        cudaEventDestroy(this->start); 
        cudaEventDestroy(this->stop); 
    }
}; 


#define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__)
static void check(cudaError_t err, char const* func, char const* file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK_LAST() check_last(__FILE__, __LINE__)
static void check_last(char const* file, int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static void print(std::vector<float>& arr, int num)
{
    for(int i = 0 ; i < num; i++){
        std::cout << arr[i] <<  " "; 
    }
    std::cout << std::endl; 
}

static void print(std::vector<float>& arr, int num, const char* name)
{
    std::cout << name << " : "; 
    for(int i = 0 ; i < num; i++){
        std::cout << arr[i] <<  " "; 
    }
    std::cout << std::endl; 
}

static int cdiv(int a, int b)
{
    return (a + b - 1) / b; 
}


enum class Device {
    CPU, 
    GPU,
}; 



__device__ __forceinline__
float4 load_bypass_cache(float4* ptr) {
    float4 val;
    asm volatile(
        "ld.global.cs.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
        : "l"(ptr)
    );
    return val;
}


__device__ __forceinline__
void prefetch_L2(float4* ptr) {
    asm volatile(
        "prefetch.global.L2 [%0];"
        :: "l"(ptr)
    );
}

__device__ __forceinline__
void prefetch_L2(float* ptr) {
    asm volatile(
        "prefetch.global.L2 [%0];"
        :: "l"(ptr)
    );
}

template <class T>
__device__ __forceinline__
void cluster_all_reduce(T* data, int cluster_rank, cooperative_groups::cluster_group& cluster){
    int cluster_dim = cluster.dim_blocks().x;
    #pragma unroll
    for(int offset = 1; offset < cluster_dim; offset *= 2){
        T* dsmem_data = cluster.map_shared_rank(data, cluster_rank ^ offset); //Butterfly ALL-REDUCE
        float val = dsmem_data[0];
        cluster.sync();
        data[0] += val;
        cluster.sync();
    }
}

template <class T, size_t BLOCK_SIZE>
__device__ __forceinline__
void block_reduce(T* smem1){
    #pragma unroll              
    for(int stride = BLOCK_SIZE / 2; stride > 16; stride >>= 1){
        if (threadIdx.x < stride){
            T sval = smem1[threadIdx.x + stride]; 
            smem1[threadIdx.x]  += sval;
        }
        __syncthreads();
    }
    if (threadIdx.x < 32){
        auto FULL_MASK = __activemask();
        T val1 = smem1[threadIdx.x];
        val1 += __shfl_down_sync(FULL_MASK, val1, 16);
        val1 += __shfl_down_sync(FULL_MASK, val1, 8);
        val1 += __shfl_down_sync(FULL_MASK, val1, 4);
        val1 += __shfl_down_sync(FULL_MASK, val1, 2);
        val1 += __shfl_down_sync(FULL_MASK, val1, 1);
        if(threadIdx.x == 0){
            smem1[0] = val1;
        }    
    }
    __syncthreads();
}

template <class T, size_t BLOCK_SIZE>
__device__ __forceinline__
void double_block_reduce(T* smem1, T* smem2){
    static_assert(BLOCK_SIZE >= 64);
    #pragma unroll              
    for(int stride = BLOCK_SIZE / 2; stride > 16; stride >>= 1){
        if (threadIdx.x < stride){
            T sval1 = smem1[threadIdx.x + stride]; 
            smem1[threadIdx.x]  += sval1;
        }
        if (threadIdx.x >= (BLOCK_SIZE - stride)){
            T sval2 = smem2[threadIdx.x - stride];
            smem2[threadIdx.x] += sval2;
        }
        __syncthreads();
    }
    if (threadIdx.x < 32){
        auto FULL_MASK = __activemask();
        T val1 = smem1[threadIdx.x];
        val1 += __shfl_down_sync(FULL_MASK, val1, 16);
        val1 += __shfl_down_sync(FULL_MASK, val1, 8);
        val1 += __shfl_down_sync(FULL_MASK, val1, 4);
        val1 += __shfl_down_sync(FULL_MASK, val1, 2);
        val1 += __shfl_down_sync(FULL_MASK, val1, 1);
        if((threadIdx.x & 31) == 0){
            smem1[0] = val1;
        }    
    }
    if (threadIdx.x >= (BLOCK_SIZE - 32)){
        auto FULL_MASK = 0xFFFFFFFF;
        T val2 = smem2[threadIdx.x];
        val2 += __shfl_down_sync(FULL_MASK, val2, 16);
        val2 += __shfl_down_sync(FULL_MASK, val2, 8);
        val2 += __shfl_down_sync(FULL_MASK, val2, 4);
        val2 += __shfl_down_sync(FULL_MASK, val2, 2);
        val2 += __shfl_down_sync(FULL_MASK, val2, 1);
        if((threadIdx.x & 31) == 0){
            smem2[0] = val2;
        }
    }
    __syncthreads();
}

static size_t next_power_of_2(size_t num){
    int cntr = 0;
    int one_bit = 0;
    while(num > 0){
        one_bit += (num & 1);
        num >>= 1;
        cntr += 1;
    }
    if (one_bit > 1){
        return (1 << cntr);
    }
    else {
        return (1 << (cntr+1));
    }
}

#endif
