#include <cuda_runtime.h>
#include <cuda.h>
#include <utils.h>
#include <map>
#include <stdexcept>
#include <math_constants.h>
#include <cuda_bf16.h>
typedef unsigned int uint;

template <typename T>
__device__ __forceinline__
T warp_allreduce_max_xor(T v, unsigned mask = 0xFFFFFFFFu) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        T other = __shfl_xor_sync(mask, v, offset);
        v = v > other ? v : other;
    }
    return v; 
}

template <typename T>
__device__ __forceinline__
T warp_allreduce_sum_xor(T v, unsigned mask = 0xFFFFFFFFu) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        T other = __shfl_xor_sync(mask, v, offset);
        v += other;
    }
    return v; 
}

typedef __nv_bfloat16 bf16;

template<size_t Br, size_t Bc, size_t D>
__global__ void _flash_attention_simple_v1_fwd(const bf16* Q, const bf16* K, const bf16* V, bf16* O, float* l, float* m, const size_t B, const size_t H, const size_t N, const size_t stride_B, const size_t stride_H, const size_t stride_N, const size_t stride_D, const size_t Tr, const size_t Tc){
    extern __shared__ bf16 smem[];
    constexpr uint O_offset = Br * D;
    constexpr uint K_offset = O_offset + Br * D;
    constexpr uint V_offset = K_offset + Bc * D;
    constexpr uint S_offset = V_offset + Bc * D;
    constexpr uint l_offset = S_offset + Br * Bc;
    constexpr uint m_offset = l_offset + Br;
    nv_bfloat16 (*Q_smem)[D]  = reinterpret_cast<nv_bfloat16 (*)[D]>(smem);
    nv_bfloat16 (*O_smem)[D]  = reinterpret_cast<nv_bfloat16 (*)[D]>(smem + O_offset);
    nv_bfloat16 (*K_smem)[D]  = reinterpret_cast<nv_bfloat16 (*)[D]>(smem + K_offset);
    nv_bfloat16 (*V_smem)[D]  = reinterpret_cast<nv_bfloat16 (*)[D]>(smem + V_offset);
    nv_bfloat16 (*S_smem)[Bc] = reinterpret_cast<nv_bfloat16 (*)[Bc]>(smem + S_offset);
    float *l_smem = reinterpret_cast<float*>(smem + l_offset);
    float *m_smem = l_smem + Br;
    
    const float scale_factor = rsqrtf(D);

    const uint qkv_offset = stride_B * blockIdx.x + stride_H * blockIdx.y;
    const nv_bfloat16* Q_base = Q + qkv_offset;
    const nv_bfloat16* K_base = K + qkv_offset;
    const nv_bfloat16* V_base = V + qkv_offset;

    const uint base_offset = H * N * blockIdx.x + N * blockIdx.y;
    bf16* O_base = O + qkv_offset;
    float* l_base = l + base_offset; //(N, )
    float* m_base = m + base_offset; //(N, )

    const uint q_tid_x = threadIdx.x / Bc; //(0, Br)
    const uint q_tid_y = threadIdx.x % Bc; //(0, Bc)

    const uint k_tid_x = threadIdx.x / Br; //(0, Bc)
    const uint k_tid_y = threadIdx.x % Br; //(0, Br)

    auto load_kv = [&](int idx){
        #pragma unroll
        for(uint tile = 0; tile < D; tile += Br){
            K_smem[k_tid_x][k_tid_y + tile] = K_base[(k_tid_x + idx * Bc) * stride_N + (k_tid_y + tile) * stride_D];
            V_smem[k_tid_x][k_tid_y + tile] = V_base[(k_tid_x + idx * Bc) * stride_N + (k_tid_y + tile) * stride_D];
        }
    };

    auto load_qo = [&](int idx){
        #pragma unroll
        for(uint tile = 0; tile < D; tile += Bc){
                Q_smem[q_tid_x][q_tid_y + tile] = Q_base[(q_tid_x + idx * Br) * stride_N + (q_tid_y + tile) * stride_D];
                O_smem[q_tid_x][q_tid_y + tile] = O_base[(q_tid_x + idx * Br) * stride_N + (q_tid_y + tile) * stride_D];
        }
    };
    
    auto store_o = [&](int idx){
        #pragma unroll
        for(uint tile = 0; tile < D; tile += Bc){
            O_base[(q_tid_x + idx * Br) * stride_N + (q_tid_y + tile) * stride_D] = O_smem[q_tid_x][q_tid_y + tile];
        }      
    };

    #pragma unroll
    for(uint j = 0; j < Tc; j++){
        load_kv(j);
        #pragma unroll
        for(uint i = 0; i < Tr; i++){
            load_qo(i);
            if (j == 0){
                if(q_tid_y == 0) m_smem[q_tid_x] = -CUDART_INF_F;
                if(q_tid_y == 1) l_smem[q_tid_x] = 0.0f;    
            } else {
                if(q_tid_y == 0) m_smem[q_tid_x] = m_base[q_tid_x + i * Br];
                if(q_tid_y == 1) l_smem[q_tid_x] = l_base[q_tid_x + i * Br];    
            }
            __syncthreads();
            float mat_val(0.0f); //Value of QK^T
            for(int idx = 0; idx < D; idx++){
                mat_val += __bfloat162float(Q_smem[q_tid_x][idx] * K_smem[q_tid_y][idx]);
            }
            mat_val *= scale_factor;

            float max_row = warp_allreduce_max_xor<float>(mat_val); //One row = one warp
            mat_val = __expf(mat_val - max_row);
            S_smem[q_tid_x][q_tid_y] = __float2bfloat16(mat_val);
            float l_curr  = warp_allreduce_sum_xor<float>(mat_val);
                  
            float old_max_row(0.0f);
            float new_max_row(0.0f);
            float l_new(0.0f), l_old(0.0f), first_exp(0.0f), second_exp(0.0f);
            if (q_tid_y == 0){ 
                old_max_row = m_smem[q_tid_x];
                new_max_row = fmaxf(max_row, old_max_row);
                m_smem[q_tid_x] = new_max_row;
                
                first_exp  = __expf(old_max_row - new_max_row);
                second_exp = __expf(max_row     - new_max_row);

                l_old = l_smem[q_tid_x];
                l_new = first_exp * l_old + second_exp * l_curr;
                l_smem[q_tid_x] = l_new;
            }
            
            l_new      = __shfl_sync(0xFFFFFFFF, l_new, 0);
            l_old      = __shfl_sync(0xFFFFFFFF, l_old, 0);
            first_exp  = __shfl_sync(0xFFFFFFFF, first_exp, 0);
            second_exp = __shfl_sync(0xFFFFFFFF, second_exp, 0);

            for(int idx = 0; idx < D; idx += Bc){
                float acc(0.0f);      
                #pragma unroll
                for(int k = 0; k < Bc; k++){
                    acc += __bfloat162float(S_smem[q_tid_x][k] * V_smem[k][q_tid_y + idx]);   
                }
                O_smem[q_tid_x][q_tid_y + idx] = __float2bfloat16((
                            first_exp * __bfloat162float(O_smem[q_tid_x][q_tid_y + idx]) * (l_old)                             + second_exp * acc
                    ) / l_new);   
            }
            __syncthreads();
            store_o(i);
            if(q_tid_y == 0) m_base[q_tid_x + i * Br] = m_smem[q_tid_x];
            if(q_tid_y == 1) l_base[q_tid_x + i * Br] = l_smem[q_tid_x];
        }
    }
}

inline size_t dyn_bytes(int Br, int Bc, int D) {
    return ((2 * (Br * D) + 2 * (Bc * D) + Br * Bc + 2 * Br)) * 4;
}

using Key = std::tuple<int,int>;
using KernelPtr = void*;

// static const std::map<Key, KernelPtr> LUT = {
//     {{4,64 },  (KernelPtr)_flash_attention_simple_v1_fwd<4, 32, 64 >},
//     {{4,128},  (KernelPtr)_flash_attention_simple_v1_fwd<4, 32, 128>},
//     {{8,64 },  (KernelPtr)_flash_attention_simple_v1_fwd<8, 32, 64 >},
//     {{8,128},  (KernelPtr)_flash_attention_simple_v1_fwd<8, 32, 128>},
//     {{16,64},  (KernelPtr)_flash_attention_simple_v1_fwd<16, 32, 64>},
//     {{16,128}, (KernelPtr)_flash_attention_simple_v1_fwd<16, 32, 128>},
//     {{32,64},  (KernelPtr)_flash_attention_simple_v1_fwd<32, 32, 64>},
//     {{32,128}, (KernelPtr)_flash_attention_simple_v1_fwd<32, 32, 128>}
// };

void attention_simple(const bf16* Q, const bf16* K, const bf16* V, bf16* O,
                   const size_t B, const size_t H, const size_t N, const size_t D)
{
    constexpr uint Bc = 32;
    constexpr uint Br = 32;
    // const uint Br = std::min<uint>(N, 32);

    const uint stride_B = H * N * D;
    const uint stride_H = N * D;
    const uint stride_N = D;
    const uint stride_D = 1;
    const uint Tr = (N + Br - 1) / Br;
    const uint Tc = (N + Bc - 1) / Bc;
    
    float *l, *m;
    cudaMalloc(&l, sizeof(float) * B * H * N);
    cudaMalloc(&m, sizeof(float) * B * H * N);

    dim3 grid(B, H);
    dim3 block(Br * Bc);

    const size_t bytes = dyn_bytes(Br, Bc, D);
    cudaFuncSetAttribute(_flash_attention_simple_v1_fwd<Br, Bc, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)0x17000);
    _flash_attention_simple_v1_fwd<Br, Bc, 128><<<grid, block, 0x17000>>>(
        Q, K, V, O, l, m, 
        B, H, N,
        stride_B, stride_H, stride_N, stride_D,
        Tr, Tc
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}
