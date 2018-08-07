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

void escrever_arquivo()
{
	char filename[100];
	sprintf(filename, "simulacao_calor_seq_%d_%d.data", TAM, ITERACOES);
	
    FILE *f = fopen(filename, "w");
    fwrite(grid, sizeof(grid), 1, f);
    fclose(f);
}

int main(int argc, char *argv[])
{
    char teste[100];
    if(argc == 2) strcpy(teste, argv[1]);
    else 		  strcpy(teste, "");
    
    char filename[20];
    sprintf(filename, "out_seq_teste%s_%d_%d.txt", teste, TAM, ITERACOES);
    freopen(filename, "w", stdout);
    
	for(int i = 0; i < TAM+2; i++)
		for(int j = 0; j < TAM+2; j++)
			grid[i][j] = 0;
	for(int i = 0; i < TAM+2; i++)
		grid[0][i] = 500;
	
    double tempos[2];
    tempos[0] = omp_get_wtime();
    
    for(int it = 0; it < ITERACOES; it++)
    {
		for(int i = 1; i <= TAM; i++)
			for(int j = 1; j <= TAM; j++)
				novo_grid[i][j] = grid[i][j] + lambda * (grid[i+1][j] - 2 * grid[i][j] + grid[i-1][j])
                                             + lambda * (grid[i][j+1] - 2 * grid[i][j] + grid[i][j-1]);
                                             
		for(int i = 1; i <= TAM; i++)
			for(int j = 1; j <= TAM; j++)
				grid[i][j] = novo_grid[i][j];
	}

    tempos[1] = omp_get_wtime();

	printf("Tamanho: %d, iteracoes: %d\n", TAM, ITERACOES);
    printf("Total: %f s\n", tempos[1] - tempos[0]);
    
    escrever_arquivo();
    check_result();

    return 0;
}
