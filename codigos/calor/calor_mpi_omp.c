#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <assert.h>

#define TAM 10000

const int ITERACOES = 10000;
const double lambda = 0.0125;

double grid[TAM+2][TAM+2], novo_grid[TAM+2][TAM+2];

void check_result()
{
	// Confere a corretude do resultado calculado comparado com um arquivo
	// com os resultados obtidos pela execução sequencial.
	
	char filename[100];
	sprintf(filename, "simulacao_calor_seq_%d_%d.data", TAM, ITERACOES);
	
	FILE *f = fopen(filename, "r");
	assert(f != NULL);
	fread(novo_grid, sizeof(novo_grid), 1, f);
	fclose(f);
	
	double EPS = 1e-9;
	
	for(int i = 0; i < TAM+2; i++)
		for(int j = 0; j < TAM+2; j++)
			assert(fabs(grid[i][j] - novo_grid[i][j]) <= EPS);
}

int main(int argc, char *argv[])
{
	MPI_Init(NULL, NULL);
	
	// Variáveis para id do processo e número de processos.
	int proc, num_proc;
	
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    
    MPI_Request req[2];
    
    // Criamos um tipo MPI que representa uma linha do grid
    MPI_Datatype datatype_line;
    MPI_Type_contiguous(TAM + 2, MPI_DOUBLE, &datatype_line);
    MPI_Type_commit(&datatype_line);
    
    // Redirecionamos a saída para um arquivo baseado no processo / teste
    char teste[100];
    if(argc == 2) strcpy(teste, argv[1]);
    else 		  strcpy(teste, "");
			
    char filename[20];
    sprintf(filename, "out_mpi_omp_teste%s_proc_%d_%d.txt", teste, proc, num_proc);
    freopen(filename, "w", stdout);
    
    // Inicializamos o grid com 0s e 500 na borda superior (todos os processos inicializam todo o grid)
	for(int i = 0; i < TAM+2; i++)
		for(int j = 0; j < TAM+2; j++)
			grid[i][j] = 0;
	for(int i = 0; i < TAM+2; i++)
		grid[0][i] = 500;
	
	// Inicializamos os vetores count e displ. Count representa o número de linhas que cada processo
	// calculará (ou seja, o número de elementos datatype_line), e displ é o offset do começo do
	// intervalo de cada processo. Na verdade sempre temos que somar 1 ao valor de displ, pois a primeira
	// linha calculada é a linha 1, não a linha 0.
    int count[num_proc], displ[num_proc];
    int ncol = TAM / num_proc;
    for(int i = 0; i < num_proc; i++)
    {
        count[i] = ncol;
        displ[i] = i * ncol;
    }
    // O último count é calculado especialmente como o valor que falta pra atingir 'TAM'
    count[num_proc-1] = TAM - (num_proc-1) * ncol;
	
    int num_threads = omp_get_max_threads();
	
    double tempos[2], tempo_apos[ITERACOES];
    tempos[0] = omp_get_wtime();
    
    // O trecho que realiza a simulação.
    #pragma omp parallel num_threads(num_threads) default(shared)
    for(int it = 0; it < ITERACOES; it++)
    {
		// 'ini' e 'fim' guardam os limites de iteração do processo atual. Somamos 1 para desconsiderar
		// a linha 0 do grid, que é pulada.
		int ini = displ[proc] + 1, fim = displ[proc] + count[proc] + 1;
		
		// Calcula os novos valores.
		#pragma omp for
		for(int i = ini; i < fim; i++)
			for(int j = 1; j <= TAM; j++)
				novo_grid[i][j] = grid[i][j] + lambda * (grid[i+1][j] - 2 * grid[i][j] + grid[i-1][j])
                                             + lambda * (grid[i][j+1] - 2 * grid[i][j] + grid[i][j-1]);
		
		// Copia os novos valores para a outra matriz.
		#pragma omp for
		for(int i = ini; i < fim; i++)
			for(int j = 1; j <= TAM; j++)
				grid[i][j] = novo_grid[i][j];
		
		// Compartilha as linhas das bordas superior/inferior. Este processo envia a primeira linha
		// que calculou para seu vizinho acima, e o vizinho envia a linha antes da primeira, e o mesmo
		// para a borda inferior. Podemos usar datatype_line, assim cada processo envia 1 datatype_line.
		#pragma omp single
		{
			if(proc-1 >= 0)
			{
				MPI_Isend(&grid[ini][0],  1, datatype_line, proc-1, 0, MPI_COMM_WORLD, &req[0]);
				MPI_Recv(&grid[ini-1][0], 1, datatype_line, proc-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Request_free(&req[0]);
			}
			if(proc+1 < num_proc)
			{
				MPI_Isend(&grid[fim-1][0], 1, datatype_line, proc+1, 0, MPI_COMM_WORLD, &req[1]);
				MPI_Recv(&grid[fim][0],    1, datatype_line, proc+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Request_free(&req[1]);
			}
			
			tempo_apos[it] = omp_get_wtime();
		}
	}
    
    // Realiza uma operação coletiva pra que todos os processos tenham o grid inteiro com os resultados.
    // Usando o tipo datatype_line, podemos usar os vetores count e displ normalmente. Novamente,
    // a linha 0 deve ser pulada dos dados a serem enviados.
    MPI_Allgatherv(&grid[1 + displ[proc]][0], count[proc], datatype_line,
				   &grid[1][0], count, displ, datatype_line, MPI_COMM_WORLD);

    tempos[1] = omp_get_wtime();

	printf("Tamanho: %d, iteracoes: %d\n", TAM, ITERACOES);
    printf("Processo %d, num threads = %d, intervalo [%d, %d)\n", proc, num_threads, displ[proc], displ[proc] + count[proc]);
    printf("Total: %f s\n", tempos[1] - tempos[0]);
    printf("Iteracao\tTempo total\n");
    for(int i = 0; i < ITERACOES; i++)
		printf("%d\t%f\n", i, tempo_apos[i] - tempos[0]);
	
    check_result();
    
    MPI_Finalize();

    return 0;
}
