#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <assert.h>

// Problema: sistema de equações lineares com 1024 váriaveis.
// a[N][N] é a matriz dos coeficientes, b[N] é o vetor dos termos independentes.
// x[N] é o vetor dos resultados, new_x[N] é um vetor auxiliar.

#define N 10000
const int ITERACOES = 5000;
double a[N][N], b[N];
double x[N], new_x[N];

void escrever_arquivo()
{
	char filename[100];
	sprintf(filename, "resultados_jacobi_seq_%d_%d.data", N, ITERACOES);
	
    FILE *f = fopen(filename, "w");
    fwrite(x, sizeof(x), 1, f);
    fclose(f);
}

void print()
{
    printf("x: [\n");
    for(int i = 0; i < N; i++) printf("  %10.3f\n", x[i]);
    printf("]\n");
}

int main(int argc, char *argv[])
{
	char teste[100];
	if(argc == 2) strcpy(teste, argv[1]);
	else 		  strcpy(teste, "");
	
    // Inicializamos a matriz dos coeficientes a[i][i] = N e os demais coeficientes = 1, garantindo
    // que o método converge. O vetor dos termos independentes é inicializado com N * (i+1) * cos(i) sem nenhuma
    // razão em particular.
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            a[i][j] = (i == j ? N : 1);
    for(int i = 0; i < N; i++)
        b[i] = N * (i+1) * cos(i);

    // Inicializamos x com um chute inicial para os valores = 0
    for(int i = 0; i < N; i++)
        x[i] = 0;

    double tempo_total[2], tempo_iteracao[2 * ITERACOES];
    tempo_total[0] = omp_get_wtime();

    char filename[20];
    sprintf(filename, "out_seq_teste%s.txt", teste);
    freopen(filename, "w", stdout);

    // Método de Jacobi: O(iterações * N^2)
    for(int it = 0; it < ITERACOES; it++)
    {
        tempo_iteracao[2 * it] = omp_get_wtime();
        
        for(int i = 0; i < N; i++)
        {
            double soma = 0;
            for(int j = 0; j < N; j++)
                soma += a[i][j] * x[j];
            soma -= a[i][i] * x[i];
            new_x[i] = (b[i] - soma) / a[i][i];
        }

        for(int i = 0; i < N; i++)
            x[i] = new_x[i];
            
        tempo_iteracao[2 * it + 1] = omp_get_wtime();
    }

    tempo_total[1] = omp_get_wtime();

    escrever_arquivo();
    
	printf("Parametros: N = %d, ITERACOES = %d\n", N, ITERACOES);
    printf("Total: %f s\n", tempo_total[1] - tempo_total[0]);
    printf("Iteracao\tTempo\n");
    for(int i = 0; i < ITERACOES; i++)
        printf("%d\t%f\n", i, tempo_iteracao[2*i + 1] - tempo_iteracao[2 * i]);

    return 0;
}
