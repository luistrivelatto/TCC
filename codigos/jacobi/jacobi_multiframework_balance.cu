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
#include "saida_tcc.c"

int MF_THRESHOLD = 0;

// Método de Jacobi para sistema de equações lineares

// Complexidade O(iterações * N^2)
// a[N][N] é a matriz dos coeficientes, b[N] é o vetor dos termos independentes.
// x[N] é o vetor dos resultados, new_x[N] é um vetor auxiliar.
// ITERACOES define o número de iterações, quanto mais iterações mais preciso o método;
// aprox. 10 * N iterações converge suficiente pro check_result dar ok.

#define N 10000
const int ITERACOES = 5000;
double A[N][N], B[N];
double x[N], new_x[N];

void inicializar_problema()
{
    // Inicializamos a matriz dos coeficientes a[i][i] = N e os demais coeficientes = 1, garantindo
    // que o método converge. O vetor dos termos independentes é inicializado com N * (i+1) * cos(i) sem nenhuma
    // razão em particular.
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            A[i][j] = (i == j ? N : 1);
    for(int i = 0; i < N; i++)
        B[i] = N * (i+1) * cos(i);

    // Inicializamos x com um chute inicial para os valores = 0
    for(int i = 0; i < N; i++)
        x[i] = 0;
}

void check_result()
{
	// Confere a corretude do resultado calculado comparado com um arquivo
	// com os resultados obtidos pela execução sequencial.

	char filename[100];
	sprintf(filename, "resultados_jacobi_seq_%d_%d.data", N, ITERACOES);

	FILE *f = fopen(filename, "r");
	assert(f != NULL);
	fread(new_x, sizeof(new_x), 1, f);
	fclose(f);

	double EPS = 1e-9;

	for(int i = 0; i < N; i++)
		if(fabs(x[i] - new_x[i]) > EPS)
			printf("erro: %f vs %f (%f)\n", x[i], new_x[i], fabs(x[i] - new_x[i])), assert(0);

    printf("check_result ok\n");
}

void print_parametros_problema()
{
    printf("Parametros: N = %d\n", N);
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

__global__ void kernel_compute(double *d_A, double *d_x, double *d_soma, int i)
{
    #define _A(i, j) (d_A[(i) * N + (j)])

	extern __shared__ double sdata[];

	int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < N) sdata[tid] = _A(i, index) * d_x[index];
    else 		  sdata[tid] = 0;

    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
		__syncthreads();
		if(tid < s)
			sdata[tid] += sdata[tid + s];
	}

	if(tid == 0)
		d_soma[i * gridDim.x + blockIdx.x] = sdata[0];

    #undef _A
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

    Multiframework_Init_lib(0, N, num_threads, proc, num_proc, MPI_COMM_WORLD, balancear(tipo_teste));

    #pragma omp parallel num_threads(num_threads) default(shared)
    {
        int tid = omp_get_thread_num();
        double *d_A, *d_x, *reduc_soma, *d_reduc_soma;
        int num_blocks, block_size;

        if(tid < num_devices)
        {
            cudaSetDevice(tid);
            cudaMalloc(&d_A, sizeof(A));
            cudaMalloc(&d_x, sizeof(x));
            cudaMemcpy(d_A, A, sizeof(A), cudaMemcpyHostToDevice);
            cudaMemcpy(d_x, x, sizeof(x), cudaMemcpyHostToDevice);

            // reduc_soma e d_reduc_soma representam matrizes de tamanho N * número de blocos
            // (para processar o i-ésimo valor cada bloco armazena sua soma em função de i)
			block_size = 1024;
			num_blocks = (N + block_size - 1) / block_size;

            int num_bytes = sizeof(double) * N * num_blocks;
            reduc_soma = (double*)malloc(num_bytes);
            cudaMalloc(&d_reduc_soma, num_bytes);
		}

		for(int it = 0; it < ITERACOES; it++)
		{
			int begin, end;

            Multiframework_begin_section(it, count, displ, MF_THRESHOLD, tid, &begin, &end);

            if(tid < num_devices)
            {
				// Usamos CUDA para paralelizar o laço interno, que calcula o somatório de
				// A[i][j] * x[j] para j = [0, N), (ou seja, um reduction), e cada bloco armazena
				// seu resultado em d_reduc_soma (em devices versão >= 6.x, ficaria mais simples
				// usando atomicAdd() em doubles).
				// O resultado é trazido de volta à CPU, que faz outro laço para acumular
				// os valores retornados em d_reduc_soma e calcular a última etapa.

				tempos_gpu[tid][it][0] = omp_get_wtime();
				cudaMemcpy(d_x, x, sizeof(x), cudaMemcpyHostToDevice);
				tempos_gpu[tid][it][1] = omp_get_wtime();

                for(int i = begin; i < end; i++)
					kernel_compute<<<num_blocks, block_size, sizeof(double) * block_size>>>(d_A, d_x, d_reduc_soma, i);
                cudaDeviceSynchronize();

				tempos_gpu[tid][it][2] = omp_get_wtime();
				cudaMemcpy(&reduc_soma[begin * num_blocks + 0], &d_reduc_soma[begin * num_blocks + 0],
							num_blocks * sizeof(double) * (end - begin), cudaMemcpyDeviceToHost);
				tempos_gpu[tid][it][3] = omp_get_wtime();

				for(int i = begin; i < end; i++)
				{
					double soma = 0;
					for(int j = 0; j < num_blocks; j++)
						soma += reduc_soma[i * num_blocks + j];
					soma -= A[i][i] * x[i];
					new_x[i] = (B[i] - soma) / A[i][i];
				}
            }
            else
            {
                for(int i = begin; i < end; i++)
				{
					double soma = 0;
					for(int j = 0; j < N; j++)
						soma += A[i][j] * x[j];
					soma -= A[i][i] * x[i];
					new_x[i] = (B[i] - soma) / A[i][i];
				}
            }

            Multiframework_end_section(it, tid);

			// Esse Allgatherv junta os dados do vetor new_x e armazena no vetor x, dessa forma
			// fazendo "implicitamente" um cópia (x[i] = new_x[i])
			#pragma omp barrier
			#pragma omp single
			MPI_Allgatherv(&new_x[displ[proc]], count[proc], MPI_DOUBLE,
							x, count, displ, MPI_DOUBLE, MPI_COMM_WORLD);
		}

        if(tid < num_devices)
        {
            cudaFree(d_A);
            cudaFree(d_x);
            cudaFree(d_reduc_soma);
            free(reduc_soma);
        }
	}

    tempo_total = omp_get_wtime() - tempo_total;

	print_parametros_problema();
	print_info_ambiente(ITERACOES, proc, num_threads, num_devices, tempo_total);
    Multiframework_print_debug(ITERACOES);
    print_gpu_debug(num_devices, ITERACOES, &tempos_gpu[0][0][0]);

    check_result();

    Multiframework_Finalize_lib();
    MPI_Finalize();

    return 0;
}
