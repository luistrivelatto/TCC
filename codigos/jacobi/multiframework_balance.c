#ifndef MULTIFRAMEWORK_BALANCE

#define MULTIFRAMEWORK_BALANCE

#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <assert.h>

double *mf_t_thread;    // Vetor que armazena o tempo de execução de cada thread deste host.
double *mf_t_proc;      // Vetor que armazena o tempo de execução de todos os processos em execução. Este
                        // vetor é compartilhado entre os processos por uma operação coletiva.

// Estes vetores são o equivalente aos vetores count e displ para os processos MPI, os quais
// são passados como parâmetro pois o código externo geralmente utiliza-os (ao passo que
// mf_thread_count e mf_thread_displ são internos à biblioteca).
int *mf_thread_count;   // Vetor que armazena o tamanho da carga de trabalho de cada thread deste host.
int *mf_thread_displ;   // Vetor que armazena o offset da carga de trabalho de cada thread deste host.

// Váriaveis que indicam o intervalo de trabalho do problema, i.e., a carga será distribuída
// entre [mf_problem_begin, mf_problem_end) (tipicamente, [0, N))
int mf_problem_begin;
int mf_problem_end;

int mf_num_threads;     // Número de threads neste host, inicializado em Init_lib.
int mf_proc;            // Id deste processo, inicializado em Init_lib.
int mf_num_proc;        // Número de processos em execução, inicializado em Init_lib.
MPI_Comm mf_mpi_comm;   // Comunicador MPI do grupo em execução, inicializado em Init_lib.

int mf_aux_threads_cont;    // Variável auxiliar para contar quantas threads já chamaram a função de balanceamento.
int mf_balance;             // Variável que indica se o balanceamento deve ser realizado ou não (no caso, por padrão seu valor é 1,
							// e só é usado 0 para rodar os testes sem balancear aproveitando as funções para medir tempos e cargas).

int MF_DEBUG = 1;

struct mf_t_debug_info {
    double tempo, delay;
    int ini, fim;
} mf_debug_info[33][10000]; // pra simplificar, criamos a tabela com 32 threads + 1 proc e 10000 iterações

void Multiframework_Init_lib(int problem_begin, int problem_end, int num_threads, int proc_id, int num_proc, MPI_Comm comm, int balance = 1)
{
	if(MF_DEBUG)
	{
		// Assert para garantir que não haverá erro acesso inválido na tabela mf_debug_info,
		// que é alocada estaticamente para 32 threads.
		assert(num_threads <= 32);
	}

	mf_problem_begin = problem_begin;
	mf_problem_end = problem_end;
    mf_num_threads = num_threads;
    mf_proc = proc_id;
    mf_num_proc = num_proc;
    mf_mpi_comm = comm;
    mf_balance = mf_balance;

    mf_t_thread = (double*) malloc(sizeof(double) * mf_num_threads);
    mf_t_proc   = (double*) malloc(sizeof(double) * mf_num_proc);
    mf_thread_count = (int*) malloc(sizeof(int) * mf_num_threads);
    mf_thread_displ = (int*) malloc(sizeof(int) * mf_num_threads);

    mf_aux_threads_cont = 0;
}

void Multiframework_Finalize_lib()
{
    free(mf_t_thread);
    free(mf_t_proc);

    free(mf_thread_count);
    free(mf_thread_displ);
}

/*
 * Método que confere que os vetores *count e *displ, de tamanho n, cobrem exatamente o intervalo [l, r).
 */
void __assert_balancing(int *count, int *displ, int n, int l, int r)
{
    assert(displ[0] == l);
    for(int i = 0; i+1 < n; i++)
        assert(displ[i] + count[i] == displ[i+1]);
    assert(displ[n-1] + count[n-1] == r);
}

/*
 * Método que corrige os arredondamentos no vetor *count (para que o total de elementos calculados seja
 * exatamente igual ao intervalo desejado) e calcula o vetor *displ de acordo. n é o tamanho dos vetores
 * count/displ e [l, r) é o intervalo da carga de trabalho que se deseja cobrir.
 */
void __adjust_count_displ(int *count, int *displ, int n, int l, int r)
{
    int problem_size = r - l;
    int sum = 0;
    for(int i = 0; i+1 < n; i++)
        sum += count[i];
    count[n - 1] = problem_size - sum;

    displ[0] = l;
    for(int i = 1; i < n; i++)
        displ[i] = displ[i-1] + count[i-1];
}

/*
 * Método que aplica o algoritmo de balanceamento para recalcular os vetores *count e *displ, de
 * tamanho n, de acordo com os tempos dados no vetor *t, para balancear o intervalo [l, r).
 * Retorna 1 se houve balanceamento, 0 se manteve igual.
 */
int __balance(int *count, int *displ, double *t, int n, int l, int r, int threshold)
{
    double tmax = t[0], tmin = t[0];

    for(int i = 1; i < n; i++)
    {
        if(t[i] > tmax) tmax = t[i];
        if(t[i] < tmin) tmin = t[i];
    }

    if(100 - tmin * 100 / tmax <= threshold)
        return 0;

    double rp[n], srp = 0;
    for(int i = 0; i < n; i++)
    {
        rp[i] = count[i] / t[i];
        srp += rp[i];
    }

    // É possível que o intervalo [l, r) vire vazio (geralmente só acontece com entradas pequenas),
    // nesse caso count[i] = 0 para todas as threads do processo, srp = 0 e rp[i] / srp = -infinito.
    // Então só calculamos rp[i] / srp se srp for diferente de 0.
    int problem_size = r - l;
    for(int i = 0; i < n; i++)
        count[i] = round(problem_size * (srp == 0 ? 0 : rp[i] / srp));
    __adjust_count_displ(count, displ, n, l, r);

    return 1;
}

void Multiframework_begin_section(int iteration, int *count, int *displ, int threshold, int thread_id, int *my_begin, int *my_end)
{
	if(MF_DEBUG)
	{
		// Assert para garantir que não haverá erro acesso inválido na tabela mf_debug_info,
		// que é alocada estaticamente para 10000 iterações.
		assert(iteration < 10000);
	}

	#pragma omp single
	{
		if(iteration == 0)
		{
			for(int i = 0; i < mf_num_proc; i++)    mf_t_proc[i]   = 1, count[i] = 1;
			for(int i = 0; i < mf_num_threads; i++) mf_t_thread[i] = 1, mf_thread_count[i] = 1;
		}
		else
		{
			MPI_Allgather(&mf_t_proc[mf_proc], 1, MPI_DOUBLE,
						   mf_t_proc, 1, MPI_DOUBLE,
						   mf_mpi_comm);
		}

		if(iteration == 0 || mf_balance)
		{
			int alterou_nos = __balance(count,
										displ,
										mf_t_proc,
										mf_num_proc,
										mf_problem_begin,
										mf_problem_end,
										iteration == 0 ? -1 : threshold);

			__balance(mf_thread_count,
					  mf_thread_displ,
					  mf_t_thread,
					  mf_num_threads,
					  displ[mf_proc],
					  displ[mf_proc] + count[mf_proc],
					  iteration == 0 || alterou_nos ? -1 : threshold);
		}
	}

	*my_begin = mf_thread_displ[thread_id];
	*my_end = mf_thread_displ[thread_id] + mf_thread_count[thread_id];

	double time;

	#pragma omp critical
	{
		time = omp_get_wtime();
		mf_t_thread[thread_id] = time;
		if(++mf_aux_threads_cont == 1)
		{
			mf_t_proc[mf_proc] = time;

			if(MF_DEBUG)
			{
				mf_debug_info[mf_num_threads][iteration].ini = displ[mf_proc];
				mf_debug_info[mf_num_threads][iteration].fim = displ[mf_proc] + count[mf_proc];
			}
		}
	}

	if(MF_DEBUG)
	{
		mf_debug_info[thread_id][iteration].delay = time - mf_t_proc[mf_proc];
		mf_debug_info[thread_id][iteration].ini = *my_begin;
		mf_debug_info[thread_id][iteration].fim = *my_end;

        __assert_balancing(count, displ, mf_num_proc, mf_problem_begin, mf_problem_end);
        __assert_balancing(mf_thread_count, mf_thread_displ, mf_num_threads, displ[mf_proc], displ[mf_proc] + count[mf_proc]);
	}
}

void Multiframework_end_section(int iteration, int thread_id)
{
	#pragma omp critical
	{
		double time = omp_get_wtime();
		mf_t_thread[thread_id] = time - mf_t_thread[thread_id];
		if(++mf_aux_threads_cont == 2 * mf_num_threads)
		{
			mf_aux_threads_cont = 0;
			mf_t_proc[mf_proc] = time - mf_t_proc[mf_proc];

			if(MF_DEBUG)
				mf_debug_info[mf_num_threads][iteration].tempo = mf_t_proc[mf_proc];
		}
	}

	if(MF_DEBUG)
		mf_debug_info[thread_id][iteration].tempo = mf_t_thread[thread_id];
}

void Multiframework_print_debug(int num_iterations)
{
    printf("Multiframework_print_debug (%d iteracoes):\n", num_iterations);
    printf("Iteracao\tThread id\tIni\tFim\tTempo\tDelay\n");
    for(int it = 0; it < num_iterations; it++)
    {
        printf("%d\t\t%d\t%d\t%f\n", it,
									 mf_debug_info[mf_num_threads][it].ini,
									 mf_debug_info[mf_num_threads][it].fim,
									 mf_debug_info[mf_num_threads][it].tempo);

        for(int j = 0; j < mf_num_threads; j++)
            printf("%d\t%d\t%d\t%d\t%f\t%f\n", it,
											   j,
											   mf_debug_info[j][it].ini,
											   mf_debug_info[j][it].fim,
											   mf_debug_info[j][it].tempo,
											   mf_debug_info[j][it].delay);
    }
    printf("---------------------------------------------\n");
}

#endif // MULTIFRAMEWORK_BALANCE
