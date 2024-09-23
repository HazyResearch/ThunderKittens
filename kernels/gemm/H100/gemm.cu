#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
template<int M_BLOCK=2, int N_BLOCK=4>
struct matmul_layout {
	using  base_tile      = st_bf<64, 64>;
	using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
	struct globals        { global_layout A, B, C; };
	struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
	struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };
	struct producer_state { kittens::index coords; };
	struct consumer_state { kittens::index coords;
							            rt_fl<16, N_BLOCK*base_tile::cols> accum; };
};
template<int M_BLOCK=2, int N_BLOCK=4, int SUPER_M=12>
struct matmul_template {
	using layout    = matmul_layout<M_BLOCK, N_BLOCK>;
	using wide_tile = st_bf<64, 64*N_BLOCK>;
	static constexpr int NUM_CONSUMER_WARPS = M_BLOCK*4;
  // Helper functions
	__host__ static inline dim3 grid(int M, int N, int K) {
		return dim3(M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
	}
	__device__ static inline void get_coords(kittens::index &coords, typename layout::globals &g, int id) {
		int Rblocks = g.A.rows / (M_BLOCK*64), Cblocks = g.A.cols / (N_BLOCK*64);
		int super_rows = (Rblocks/SUPER_M)*SUPER_M,
        final_rows = Rblocks - super_rows,
        super_repeat = SUPER_M*Cblocks;
		if (blockIdx.x < super_rows * Cblocks)
			coords = { SUPER_M*(blockIdx.x/super_repeat) + blockIdx.x%SUPER_M,
					       (blockIdx.x%super_repeat)/SUPER_M };
		else {
			int remainder_id = blockIdx.x - super_rows*Cblocks;
			coords = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
		}
		coords = { iters(g), coords.r*M_BLOCK + id, coords.c*N_BLOCK };
	}
  // ThunderKittens template functions
	__device__ static inline int iters(typename layout::globals &g) { return g.A.cols/64; }
	struct producer {
		__device__ static void setup(producer_setup_args<layout> args) {
			warpgroup::producer_registers(); // decrease registers for producers
			get_coords(args.state.coords, args.globals, 0);
		}
		__device__ static void load(producer_load_args<layout> args) {
			if(warpgroup::warpid() == 0) {
				tma::expect(args.inputs_arrived, args.input);
				for(int i = 0; i < M_BLOCK; i++)
					tma::load_async(args.input.a[i], args.globals.A,
									        {args.state.coords.r+i, args.iter}, args.inputs_arrived);
				for(int i = 0; i < N_BLOCK; i++)
					tma::load_async(args.input.b[i], args.globals.B,
									        {args.iter, args.state.coords.c+i}, args.inputs_arrived);
				arrive(args.inputs_arrived, 3);
			}
		}
	};
	struct consumer {
		__device__ static void setup(consumer_setup_args<layout> args) {
			warpgroup::consumer_registers<NUM_CONSUMER_WARPS/4>(); // increase registers for consumers
			get_coords(args.state.coords, args.globals, warpgroup::groupid());
			zero(args.state.accum);
		}
		__device__ static void work(consumer_work_args<layout> args) {
			warpgroup::mma_AB(
				args.state.accum, // dest registers
				args.input.a[warpgroup::groupid()], // A matrix
				reinterpret_cast<wide_tile&>(args.input.b) // B matrix
			);
			warpgroup::mma_async_wait();
			if(warpgroup::laneid() == 0) arrive(args.inputs_finished, 4);
		}
		__device__ static void finish(consumer_finish_args<layout> args) {
			warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
			warpgroup::sync();
			if(warpgroup::warpid() == 0) for(int i = 0; i < N_BLOCK; i++)
				tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i],
								         {args.state.coords.r, args.state.coords.c+i});
		}
	};
};


constexpr bool NCU = false;
#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <omp.h>

void cpu_gemm(float* a, float* b, float* c, int M, int N, int K) {
	#pragma omp parallel for collapse(2) // otherwise the CPU version takes for everrrrrr
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			float sum = 0.0f;
			for (int k = 0; k < K; k++) {
				sum += a[i * K + k] * b[k * N + j];
			}
			c[i * N + j] = sum;
		}
	}
}

template<typename mmt, bool use_cache>
void inner_run(bf16 *d_A, bf16 *d_B, bf16 *d_C, int M, int N, int K, dim3 grid, dim3 block, kittens::tma::tma_cache & tma_cache) {
	using global_layout = typename mmt::layout::global_layout;
	using globals  = typename mmt::layout::globals;
	if constexpr (use_cache) {
		global_layout Ag{d_A, nullptr, nullptr, M, K, tma_cache};
		global_layout Bg{d_B, nullptr, nullptr, K, N, tma_cache};
		global_layout Cg{d_C, nullptr, nullptr, M, N, tma_cache};
		globals G{Ag, Bg, Cg};
		prototype::pc<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
	} else {
		global_layout Ag{d_A, nullptr, nullptr, M, K};
		global_layout Bg{d_B, nullptr, nullptr, K, N};
		global_layout Cg{d_C, nullptr, nullptr, M, N};
		globals G{Ag, Bg, Cg};
		prototype::pc<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
	}
	// cudaDeviceSynchronize();
}

template<typename mmt, bool use_cache>
int run_benchmark(size_t M, size_t N, size_t K) {
	cudaError_t cudaStatus;

	std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------  ";
	std::cout << (use_cache ? "USING TMA CACHE" : "NOT USING TMA CACHE") << std::endl;

	// Allocate host memory
	float *h_A = new float[M * K];
	float *h_B = new float[K * N];
	float *h_C = new float[M * N];
	float *h_C_ref = new float[M * N];

	std::cout << "Allocated host memory" << std::endl;

	// Initialize random number generator
	std::random_device rd;
	std::mt19937 gen(42);
	std::uniform_real_distribution<> dis(-0.5, 0.5);

	// Initialize matrices with random values
	for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
	for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);

	std::cout << "Initialized matrices" << std::endl;

	// Perform CPU matrix multiplication for reference
	if(M < 8192) cpu_gemm(h_A, h_B, h_C_ref, M, N, K);

	std::cout << "Performed CPU matrix multiplication" << std::endl;

	// Allocate device memory
	__nv_bfloat16 *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, M*K*sizeof(__nv_bfloat16));
	cudaMalloc(&d_B, K*N*sizeof(__nv_bfloat16));
	cudaMalloc(&d_C, M*N*sizeof(__nv_bfloat16));

	// Check for CUDA errors
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
		// Optionally, you might want to exit the program or handle the error in some way
		return -1;
	}

	std::cout << "Allocated device memory" << std::endl;

	// Convert to __nv_bfloat16 and copy to device
	__nv_bfloat16 *h_A_bf16 = new __nv_bfloat16[M * K];
	__nv_bfloat16 *h_B_bf16 = new __nv_bfloat16[K * N];
	for (int i = 0; i < M * K; ++i) h_A_bf16[i] = __float2bfloat16(h_A[i]);
	for (int i = 0; i < K * N; ++i) h_B_bf16[i] = __float2bfloat16(h_B[i]);

	cudaMemcpy(d_A, h_A_bf16, M*K*2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B_bf16, K*N*2, cudaMemcpyHostToDevice);

	std::cout << "Copied matrices to device" << std::endl;

	unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
	cudaFuncSetAttribute(prototype::pc<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

	kittens::tma::tma_cache tma_cache;

	// Launch kernel
	dim3 grid(mmt::grid(M, N, K));
	dim3 block(prototype::num_threads<mmt>);
	std::cout << "Launching warmup kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
	for(int i = 0; i < (NCU ? 0 : 2); i++) { // warmup
		inner_run<mmt, use_cache>(d_A, d_B, d_C, M, N, K, grid, block, tma_cache);
	}

	// Start timing
	cudaDeviceSynchronize();
	std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n";
	auto start = std::chrono::high_resolution_clock::now();

	constexpr int ITERS = (NCU ? 1 : 10);
	for(int i = 0; i < ITERS; i++) {
		inner_run<mmt, use_cache>(d_A, d_B, d_C, M, N, K, grid, block, tma_cache);
	}
	cudaDeviceSynchronize();

	// End timing
	auto end = std::chrono::high_resolution_clock::now();

	// Calculate duration
	std::chrono::duration<double> diff = end - start;
	double useconds = diff.count() * 1e6 / ITERS;

	// Calculate TFLOPs
	double flops = double(2.0) * M * N * K; // 2 FLOPs per multiply-add
	double tflops = (flops / useconds) / 1e6;

	std::cout << "Avg Kernel execution time: " << useconds << " us\n";
	std::cout << "Achieved performance: " << tflops << " TFLOPs\n";
	
	// Check for CUDA errors
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
		// Optionally, you might want to exit the program or handle the error in some way
		return -1;
	}

	// Copy result back to host
	__nv_bfloat16 *h_C_bf16 = new __nv_bfloat16[M * N];
	cudaMemcpy(h_C_bf16, d_C, M*N*2, cudaMemcpyDeviceToHost);

	std::cout << "Copied result back to host" << std::endl;

	// Convert result back to float for comparison
	for (int i = 0; i < M * N; ++i) h_C[i] = __bfloat162float(h_C_bf16[i]);

	std::cout << "Converted result back to float" << std::endl;

	// Check result
	float max_error = 0.0f;
	int error_count = 0;
	for (int i = 0; i < M * N; ++i) {
		float error = std::abs(h_C[i] - h_C_ref[i]);
		if(error > 1.0) { // large because of bf16 vs fp32 numerics
			if(error_count < 20) std::cout << "Error at row " << i / N << " col " << i % N << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
			else if(error_count == 21) std::cout << "Too many errors to show them all.\n";
			error_count++;
		}
		max_error = std::max(max_error, error);
	}

	std::cout << "Max error: " << max_error << std::endl;
	std::cout << "Error count: " << error_count << std::endl;

	// Clean up
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;
	delete[] h_C_ref;
	delete[] h_A_bf16;
	delete[] h_B_bf16;
	delete[] h_C_bf16;
	// Ag.cleanup();
	// Bg.cleanup();
	// Cg.cleanup();
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}

int main() {
	// int Cblocks = 22, Rblocks = 24;
	// int Cblocks192 = 20, Rblocks192 = 16;
	// run_benchmark<matmul_template<4>>(4096, 4096, 4096, Rblocks, Cblocks, Rblocks192, Cblocks192);
	// run_benchmark<matmul_template<8>>(4096, 4096, 4096, Rblocks, Cblocks, Rblocks192, Cblocks192);
	// run_benchmark<matmul_template<12>>(4096, 4096, 4096, Rblocks, Cblocks, Rblocks192, Cblocks192);
	int N;
	// N = 2048;
	// run_benchmark<matmul_template_128_256<8>>(N, N, N, N/128, N/256, N/128, N/256, dim3(N*N/(128*256)));
	// N = 3072;
	// run_benchmark<matmul_template_192_192<8>>(N, N, N, N/192, N/192, N/192, N/192, dim3(N*N/(192*192)));
	// run_benchmark<matmul_template_128_256<8>>(N, N, N, N/128, N/256, N/128, N/256, dim3(N*N/(128*256)));
	N = 4096;
	// run_benchmark<matmul_template_192_192<8>>(N, N, N, N/192, N/192, N/192, N/192, dim3(N*N/(192*192)));
	run_benchmark<matmul_template<2,4,8>, false>(N, N, N);
	run_benchmark<matmul_template<2,4,8>, true>(N, N, N);
	// N = 6144;
	// run_benchmark<matmul_template_192_192<8>>(N, N, N, N/192, N/192, N/192, N/192, dim3(N*N/(192*192)));
	// run_benchmark<matmul_template_128_256<8>>(N, N, N, N/128, N/256, N/128, N/256, dim3(N*N/(128*256)));
	// N = 8192;
	// run_benchmark<matmul_template_128_256<8>>(N, N, N, N/128, N/256, N/128, N/256, dim3(N*N/(128*256)));
	// N = 9216;
	// run_benchmark<matmul_template_192_192<8>>(N, N, N, N/192, N/192, N/192, N/192, dim3(N*N/(192*192)));
	// run_benchmark<matmul_template_128_256<8>>(N, N, N, N/128, N/256, N/128, N/256, dim3(N*N/(128*256)));
	// N = 12288;
	// run_benchmark<matmul_template_192_192<8>>(N, N, N, N/192, N/192, N/192, N/192, dim3(N*N/(192*192)));
	// run_benchmark<matmul_template_128_256<8>>(N, N, N, N/128, N/256, N/128, N/256, dim3(N*N/(128*256)));
	N = 16384;
	run_benchmark<matmul_template<2,4,12>, false>(N, N, N);
	run_benchmark<matmul_template<2,4,12>, true>(N, N, N);
	return 0;
}