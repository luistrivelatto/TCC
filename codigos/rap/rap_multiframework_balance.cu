#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "multiframework_balance.c"
#include "cuda_prototypes.c"
#include "saida_tcc.c"

int MF_THRESHOLD = 0;

// Resource Allocation Problem (RAP)

// Resolução com PD O(N * M^2)
// A tabela G[i][j] é preenchida linha a linha, com a linha i dependendo da linha i-1
// O laço mais interno é irregular, na j-ésima iteração ele custa O(j)

// G[N+1][M+1], P[N+1][M+1]
#define TAM 10001
int G[TAM][TAM], P[TAM][TAM];
int N = 5000, M = 10000;

void inicializar_problema()
{
    for(int i = 0; i <= N; i++)
        for(int j = 0; j <= M; j++)
            G[i][j] = 0, P[i][j] = 1;
}

void check_result(int res, int n)
{
    assert(res == n);
    printf("check_result ok\n");
}

void print_parametros_problema()
{
    printf("Parametros: N = %d, M %d\n", N, M);
}

void inicializar_variaveis_ambiente(int *proc, int *num_proc, int *num_threads, int *num_devices)
{
    MPI_Comm_rank(MPI_COMM_WORLD, proc);
    MPI_Comm_size(MPI_COMM_WORLD, num_proc);

    // Já que omp_get_max_threads() e cudaGetDeviceCount estão dando treta, usar valores manuais
    // 0 = Cleison, 1 = Chiquinha, 2 = Rebinha
    int NUM_DEVICES[3] = {0, 1, 0};
    int NUM_THREADS[3] = {2, 4, 2};

    *num_devices = NUM_DEVICES[*proc];
    *num_threads = NUM_THREADS[*proc];
}

__global__ void kernel_compute(int *d_G, int *d_P, int i, int begin, int end)
{
    #define _G(i, j) (d_G[(i) * TAM + (j)])
    #define _P(i, j) (d_P[(i) * TAM + (j)])

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int j = begin + index; j < end; j += stride)
    {
        int aqui = _P(i, 0);
        for(int x = 0; x <= j; x++)
        {
            int fij = _G(i-1, j-x) + _P(i, x);
            if(aqui < fij)
                aqui = fij;
        }
        _G(i, j) = aqui;
    }

    #undef _G
    #undef _P
}

int main(int argc, char *argv[])
{
    MPI_Init(NULL, NULL);

    int proc, num_proc, num_threads, num_devices;

    inicializar_variaveis_ambiente(&proc, &num_proc, &num_threads, &num_devices);
    conferir_argc_argv(argc, argv);
    redirecionar_saida(argc, argv, proc);
	inicializar_problema();

	tipo_execucao tipo_teste = get_tipo_execucao(argc, argv);
	if(!usar_gpu(tipo_teste))
		num_devices = 0;

    int count[num_proc];
    int displ[num_proc];
	double tempos_gpu[max(1, num_devices)][N][4];
    double tempo_total;

    tempo_total = omp_get_wtime();

    Multiframework_Init_lib(0, M+1, num_threads, proc, num_proc, MPI_COMM_WORLD, balancear(tipo_teste));

    #pragma omp parallel num_threads(num_threads) default(shared)
    {
        int tid = omp_get_thread_num();
        int *d_G, *d_P;

        if(tid < num_devices)
        {
            cudaSetDevice(tid);
            cudaMalloc(&d_G, sizeof(G));
            cudaMalloc(&d_P, sizeof(P));
            cudaMemcpy(d_P, P, sizeof(P), cudaMemcpyHostToDevice);
        }

        for(int i = 1; i <= N; i++)
        {
            int it = i-1, begin, end;

            Multiframework_begin_section(it, count, displ, MF_THRESHOLD, tid, &begin, &end);

            if(tid < num_devices)
            {
				tempos_gpu[tid][it][0] = omp_get_wtime();
				// A linha comentada deveria ser equivalente à de baixo, não? Mas a comentada dá Seg Fault e a outra não
                //cudaMemcpy(d_G + (i-1) * TAM, G + (i-1) * TAM, sizeof(int) * TAM, cudaMemcpyHostToDevice);
                cudaMemcpy(&d_G[(i-1) * TAM + 0], &G[i-1][0], sizeof(int) * TAM, cudaMemcpyHostToDevice);
				tempos_gpu[tid][it][1] = omp_get_wtime();

                int block_size = 1024;
                int num_blocks = (end - begin + block_size - 1) / block_size;
                kernel_compute<<<num_blocks, block_size>>>(d_G, d_P, i, begin, end);
                cudaDeviceSynchronize();

				tempos_gpu[tid][it][2] = omp_get_wtime();
                cudaMemcpy(&G[i][begin], &d_G[i * TAM + begin], sizeof(int) * (end - begin), cudaMemcpyDeviceToHost);
				tempos_gpu[tid][it][3] = omp_get_wtime();
            }
            else
            {
                for(int j = begin; j < end; j++)
                {
                    G[i][j] = P[i][0];
                    for(int x = 0; x <= j; x++)
                    {
                        int fij = G[i-1][j-x] + P[i][x];
                        if(G[i][j] < fij)
                            G[i][j] = fij;
                    }
                }
            }

            Multiframework_end_section(it, tid);

            #pragma omp barrier
            #pragma omp single
            MPI_Allgatherv(&G[i][displ[proc]], count[proc], MPI_INT,
                           &G[i][0], count, displ, MPI_INT, MPI_COMM_WORLD);
        }

        if(tid < num_devices)
        {
            cudaFree(d_G);
            cudaFree(d_P);
        }
    }

    tempo_total = omp_get_wtime() - tempo_total;

	print_parametros_problema();
	print_info_ambiente(N, proc, num_threads, num_devices, tempo_total);
    Multiframework_print_debug(N);
    print_gpu_debug(num_devices, N, &tempos_gpu[0][0][0]);

    check_result(G[N][M], N);

    Multiframework_Finalize_lib();
    MPI_Finalize();

    return 0;
}
