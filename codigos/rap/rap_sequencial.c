#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <assert.h>

void check_result(int res, int n)
{
    assert(res == n);
}

// Resource Allocation Problem (RAP)

// Resolução com PD O(N * M^2)
// A tabela G[i][j] é preenchida linha a linha, com a linha i dependendo da linha i-1
// O laço mais interno é irregular, na j-ésima iteração ele custa O(j)

// G[N+1][M+1], P[N+1][M+1]
#define TAM 10001
int G[TAM][TAM], P[TAM][TAM];
int N = 5000, M = 10000;

int main(int argc, char *argv[])
{   
    for(int i = 0; i <= N; i++)
        for(int j = 0; j <= M; j++)
            G[i][j] = 0, P[i][j] = 1;
    
	char teste[100];
	if(argc == 2) strcpy(teste, argv[1]);
	else 		  strcpy(teste, "");
	
    char filename[20];
    sprintf(filename, "out_seq%s.txt", teste);
    freopen(filename, "w", stdout);

    double tempo_total[2], tempo_iteracao[2 * N];

    tempo_total[0] = omp_get_wtime();

    // Resolução com PD O(N * M^2)
    // A tabela G[i][j] é preenchida linha a linha, com a linha i dependendo da linha i-1
    // O laço mais interno é irregular, na j-ésima iteração ele custa O(j)
    for(int i = 1; i <= N; i++)
    {
        int it = i-1;
        tempo_iteracao[2 * it] = omp_get_wtime();

        for(int j = 0; j <= M; j++)
        {
            G[i][j] = P[i][0];
            for(int x = 0; x <= j; x++)
            {
                int fij = G[i-1][j-x] + P[i][x];
                if(G[i][j] < fij)
                    G[i][j] = fij;
            }
        }

        tempo_iteracao[2 * it + 1] = omp_get_wtime();
    }

    tempo_total[1] = omp_get_wtime();

    printf("Total: %f s\n", tempo_total[1] - tempo_total[0]);
    printf("Iteracao\tTempo\n");
    for(int i = 0; i < N; i++)
        printf("%d\t%f\n", i, tempo_iteracao[2*i + 1] - tempo_iteracao[2 * i]);

    check_result(G[N][M], N);

    return 0;
}
