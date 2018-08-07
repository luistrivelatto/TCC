#ifndef SAIDA_TCC

#define SAIDA_TCC

#include <string.h>

enum tipo_execucao {
	SEM_GPU_SEM_BALANCEAMENTO,
	SEM_GPU_COM_BALANCEAMENTO,
	COM_GPU_SEM_BALANCEAMENTO,
	COM_GPU_COM_BALANCEAMENTO
};

const char *tipo_execucao_str[] = {
	"sem-gpu-sem-balanceamento",
	"sem-gpu-com-balanceamento",
	"com-gpu-sem-balanceamento",
	"com-gpu-com-balanceamento",
};

int usar_gpu(tipo_execucao tipo)
{
	return tipo == COM_GPU_SEM_BALANCEAMENTO
		|| tipo == COM_GPU_COM_BALANCEAMENTO;
}

int balancear(tipo_execucao tipo)
{
	return tipo == SEM_GPU_COM_BALANCEAMENTO
		|| tipo == COM_GPU_COM_BALANCEAMENTO;
}

void conferir_argc_argv(int argc, char *argv[])
{
	int ok = 1;
	
	if(argc != 3) ok = 0;
	else if(!(strcmp(argv[1], tipo_execucao_str[0]) == 0 ||
		      strcmp(argv[1], tipo_execucao_str[1]) == 0 ||
		      strcmp(argv[1], tipo_execucao_str[2]) == 0 ||
		      strcmp(argv[1], tipo_execucao_str[3]) == 0)) ok = 0;
		      
	if(!ok)
	{
		printf("Uso: nome.x [tipo do teste] [nome do teste]\nTipo do teste = {%s, %s, %s, %s}\n",
			   tipo_execucao_str[0],
			   tipo_execucao_str[1],
			   tipo_execucao_str[2],
			   tipo_execucao_str[3]);
		exit(-1);
	}
}

tipo_execucao get_tipo_execucao(int argc, char *argv[])
{
	if(strcmp(argv[1], tipo_execucao_str[0]) == 0) return SEM_GPU_SEM_BALANCEAMENTO;
	if(strcmp(argv[1], tipo_execucao_str[1]) == 0) return SEM_GPU_COM_BALANCEAMENTO;
	if(strcmp(argv[1], tipo_execucao_str[2]) == 0) return COM_GPU_SEM_BALANCEAMENTO;
	return COM_GPU_COM_BALANCEAMENTO;
}

void redirecionar_saida(int argc, char *argv[], int proc)
{
	tipo_execucao tipo = get_tipo_execucao(argc, argv);
	
	char filename[512];
	sprintf(filename, "out-%s_teste%s_p%d.txt", tipo_execucao_str[tipo], argv[2], proc);
    freopen(filename, "w", stdout);
}

void print_info_ambiente(int iteracoes, int proc, int num_threads, int num_devices, double tempo_total)
{
	printf("Iteracoes = %d\n", iteracoes);
    printf("Processo %d, num threads = %d, num devices = %d\n", proc, num_threads, num_devices);
    printf("Total: %f s\n", tempo_total);
}

void print_gpu_debug(int num_devices, int iteracoes, double *tempos_gpu)
{
	#define _tempos_gpu(a, b, c) (*(tempos_gpu + a*iteracoes*4 + 4*b + c))
	
    for(int i = 0; i < num_devices; i++)
    {
		printf("GPU %d:\n", i);
		printf("Iteracao\tTempo copia->GPU\tTempo exec\tTempo copia->CPU\n");
		for(int j = 0; j < iteracoes; j++)
			printf("%d\t%f\t%f\t%f\n", j,
									   _tempos_gpu(i, j, 1) - _tempos_gpu(i, j, 0),
									   _tempos_gpu(i, j, 2) - _tempos_gpu(i, j, 1),
									   _tempos_gpu(i, j, 3) - _tempos_gpu(i, j, 2));
	}
	
	#undef _tempos_gpu
}

#endif // SAIDA_TCC
