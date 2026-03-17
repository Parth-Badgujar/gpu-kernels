#include <cuda.h>
#include <cuda_runtime.h>

class CUDAtimer {
  public:
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDAtimer() {
        cudaEventCreate(&this->start);
        cudaEventCreate(&this->stop);
    }
    void start_timer() { cudaEventRecord(this->start, 0); }
    void stop_timer() {
        cudaEventRecord(this->stop, 0);
        cudaEventSynchronize(this->stop);
    }
    float get_time() {
        float ms;
        cudaEventElapsedTime(&ms, this->start, this->stop);
        return ms;
    }
    ~CUDAtimer() {
        cudaEventDestroy(this->start);
        cudaEventDestroy(this->stop);
    }
};

__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)::"memory");
    return t;
}

enum class ProfilerTag {
    TMA_LOAD,
    COMPUTE,
    STORE_C,
};

struct Event {
    uint64_t start;
    uint64_t end;
    uint32_t warp_id;
};

struct Profiler {
    uint32_t num_envet;
    ProfilerTag tag;
    uint32_t sm_id;
    uint32_t num_events;
    Event *traces;
    __device__ __forceinline__ void init(ProfilerTag tag, void *trace_ptr,
                                         int num_events) {
        this->traces = (Event *)trace_ptr;
        this->tag = tag;
        asm volatile("mov.u32 %0, %smid;\n" : "=r"(this->sm_id));
        this->num_events = num_events;
    }
    __device__ __forceinline__ void start_event(int iter) {
        this->traces[iter].start = globaltimer();
    }
    __device__ __forceinline__ void end_event(int iter) {
        this->traces[iter].end = globaltimer();
    }
    __device__ __forceinline__ void copy_to_gmem(void *gmem_addr) {
        uint32_t num_bytes = sizeof(Event) * this->num_events;
        for (int i = 0; i < num_bytes / 16; i++) {
            reinterpret_cast<uint4 *>(gmem_addr)[i] =
                reinterpret_cast<uint4 *>(this->traces)[i];
        }
    }
};