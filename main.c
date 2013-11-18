#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <string.h>

// UFSCar Sorocaba
// Parallel Computing
// Final Project
// Rodrigo Barbieri, Rafael Machado and Guilherme Baldo


#ifdef _WIN32
double log2(double n){ //windows' math.h does not have log2 function  
    return log(n) / log(2);  
}
#endif

typedef struct ProgramInfo { //Structure responsible for maintaining program information and state
	long n; //number of elements for a specific test case
	int p; //number of threads selected by the user
	char print; //whether the user chose to print the sorted array
} ProgramInfo;

ProgramInfo *p_info;

typedef struct BMP_HEADER {
	short signature;
	long size;
	short reserved1;
	short reserved2;
	long offset_start;
	long header_size;
	long width;
	long height;
	short planes;
	short bits;
	long compression;
	long size_data;
	long hppm;
	long vppm;
	long colors;
	long important_colors;
} BMP_HEADER;

void initialize_header(BMP_HEADER *header){
	header->signature = 0;
	header->size = 0;
	header->reserved1 = 0;
	header->reserved2 = 0;
	header->offset_start = 0;
	header->header_size = 0;
	header->width = 0;
	header->height = 0;
	header->planes = 0;
	header->bits = 0;
	header->compression = 0;
	header->size_data = 0;
	header->hppm = 0;
	header->vppm = 0;
	header->colors = 0;
	header->important_colors = 0;

}

void print_header(BMP_HEADER *header){
	printf("signature: %hd,\nsize: %ld,\nreserved1: %hd,\nreserved2: %hd,\noffset_start: %ld,\nheader_size: %ld,\nwidth: %ld,\nheight: %ld,\nplanes: %hd,\nbits: %hd,\ncompression: %ld,\nsize_data: %ld,\nhppm: %ld,\nvppm: %ld,\ncolors: %ld,\nimportant_colors: %ld\n",header->signature,header->size,header->reserved1,header->reserved2,header->offset_start,header->header_size,header->width,header->height,header->planes,header->bits,header->compression,header->size_data,header->hppm,header->vppm,header->colors,header->important_colors);
}

void read_header(BMP_HEADER *header,FILE *f){
	fread(&(header->signature),2,1,f);
	fread(&(header->size),4,1,f);
	fread(&(header->reserved1),2,1,f);
	fread(&(header->reserved2),2,1,f);
	fread(&(header->offset_start),4,1,f);
	fread(&(header->header_size),4,1,f);
	fread(&(header->width),4,1,f);
	fread(&(header->height),4,1,f);
	fread(&(header->planes),2,1,f);
	fread(&(header->bits),2,1,f);
	fread(&(header->compression),4,1,f);
	fread(&(header->size_data),4,1,f);
	fread(&(header->hppm),4,1,f);
	fread(&(header->vppm),4,1,f);
	fread(&(header->colors),4,1,f);
	fread(&(header->important_colors),4,1,f);

}

void myMemCpy(int* dest,int* src,int size){ //copy an amount of data from one array to another
	int i;
	for (i = 0; i < size; i++)
		dest[i] = src[i];
}

int* initialize(int *n,int *p, int *t,FILE *f,int *rank){ //all processes initialize here, process 0 reads input and broadcasts, other processes just receive the broadcast
	int r;
	int max = 0;
	int i = 0;
	int *array = NULL;
	int value;
	int adjust = 0;
	int message[2];

	if (*rank == 0){
		if (f != NULL){
			fscanf(f,"%d\n",t);	//number of times to repeat the test case

			if (*t != 0){ //abort if 0

				fscanf(f,"%d\n",n);	//number of elements		

				r = *n % *p;
				if (r > 0)
					adjust = (*p - r);  //adjust the number elements so it is divisible by the number of processes
				*n = *n + adjust;
	
				array = malloc(sizeof(int) * *n);
		
				while (!feof(f) && i < (*n - adjust) && fscanf(f,"%d",&value) == 1){ //reading input to array
					array[i] = value;
					if (i == 0)  //detects the max value in input to adjust the array for scatter operation
						max = value;
					else {
						if (value > max)
							max = value;
					}
					i++;
				}
				max++;

				for (i = 0; i < adjust ; i++) //fill the reserved space with adjustment numbers (max+1)
					array[*n-1-i] = max;			

			}
		} else
			*t = 0;
		message[0] = *n;
		message[1] = *t; //pack both values into one single message
		MPI_Bcast(message,2,MPI_INT,0,MPI_COMM_WORLD);
		
	} else {
		MPI_Bcast(message,2,MPI_INT,0,MPI_COMM_WORLD);	
		*n = message[0]; //receive values from process 0 and unpack
		*t = message[1];
	}
	
	return array; //data read from input, NULL for all processes except process 0
}

int* divide(int *local_n,int *rank,int array[]){ //Scatter elements equally to all processes

	int *local_array;
	local_array = malloc(sizeof(int) * *local_n);
	MPI_Scatter(array,*local_n,MPI_INT,local_array,*local_n,MPI_INT,0,MPI_COMM_WORLD);
	return local_array;	
}

void print_local_array(int *local_n, int local_array[]){ //prints an array

	int i;
	for (i = 0; i < *local_n; i++)
		printf("%d ",local_array[i]);
	printf("\n");
}

int* gather(int local_n,int *p,int *rank,int local_array[]){ //customized reduce function

	int i;
	int divider = 1; //reduction tier
	int *result;
	int *aux = NULL; //auxiliar array for MergeSort
	int *temp = NULL; //receive buffer

	if (*rank % 2 != 1){ //odd processes will not receive messages
		temp = malloc(sizeof(int) * local_n);
		result = malloc(sizeof(int) * local_n*2); //the result from each iteration is the process' data joined to the data received
		myMemCpy(result,local_array,local_n); //copies the process' data to the result array
		aux = malloc(sizeof(int) * local_n*2);    
	} else {
		result = local_array;
	}
	for (i = 0; i < log2(*p) ; i++){ //number of times to iterate based on the number of processes for a tree-like behavior
		if ((*rank / divider) % 2 == 1){ //if the process in the current tier is an odd number it sends its content to the process on its left
			MPI_Send(result,local_n,MPI_INT,*rank - divider,0,MPI_COMM_WORLD);
			break;
		} else { //if the process in the current tier is an even number, it receives from the process on its right
			MPI_Recv(temp,local_n,MPI_INT,(int)(*rank+pow(2,i)),0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			myMemCpy(result+local_n,temp,local_n); //copies to the second half of result array
			//mergeSortPart2(result,0,(local_n*2)-1,aux); //merge both halves
		}
		if (i + 1 < log2(*p)){ //if there is one more step, reallocates memory, doubles data size for next round
			local_n = local_n * 2;
			result = realloc(result,sizeof(int) * local_n*2);
			aux = realloc(aux,sizeof(int) * local_n*2);
			temp = realloc(temp,sizeof(int) * local_n);
			divider = divider * 2; //moves to next tier in reduction operation
		}
	}
	if (*rank % 2 == 0){ //everyone frees allocated memory
		free(aux);
		free(temp);
		if (*rank != 0) 
			free(result); //process 0 keeps the result
	}
	return result;
}

FILE* validation(int* argc, char* argv[]){ //validates several conditions before effectively starting the program

	char filename[50];
	int threads;
	FILE* f;
	if (*argc != 3){ //validates number of arguments passed to executable, currently number of threads and file name
		printf("Usage: %s <number of threads> <file name>\n",argv[0]);
		fflush(stdout);
		exit(0);
	}
	
	threads = strtol(argv[1], NULL, 10);

	if ((threads == 0) || (threads != p_info->p) || (!((threads != 0) && ((threads & (threads - 1)) == 0)))){ //validates if number of threads passed as argument is power of two, which should be appropriate for reduce operation
		printf("Number of processes is not valid or power of two!\n");
		fflush(stdout);
		exit(0);
	}

	p_info->p = threads;

	strcpy(filename,argv[2]);
	f = fopen(filename,"r");
	if (f == NULL){ //check if the file inputted exists
		printf("File not found!");
		fflush(stdout);
		exit(0);
	}
	return f;

}

void calculateLocalArray(long* local_n,long* my_first_i,int* rank){ //calculates local number of elements and starting index for a specific rank based on total number of elements
	long div = p_info->n / p_info->p;
	long r = p_info->n % p_info->p; //divides evenly between all threads, firstmost threads get more elements if remainder is more than zero
	if (*rank < r){
		*local_n = div + 1;
		if (my_first_i != NULL) //allows my_first_i parameter to be NULL instead of an address
			*my_first_i = *rank * *local_n;
	} else {
		*local_n = div;
		if (my_first_i != NULL) //allows my_first_i parameter to be NULL instead of an address
			*my_first_i = *rank * *local_n + r;
	}
}

void 

int main(int argc, char *argv[]){

	FILE *f = NULL;
	char filename[50];
	char print = 'Y';
	int rank,p,t;
	long local_n,my_first_i,n;
	int *array = NULL;
	int *aux;
	int *result;
	int *local_array;
	double start = 0;
	double end = 0;
	double total = 0;
	double max = 0;
	int i;

	MPI_Init(NULL,NULL);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	p_info = (ProgramInfo*) malloc(sizeof(ProgramInfo)); //allocates ProgramInfo structure

	p_info->p = p;

	BMP_HEADER header;
	initialize_header(&header);

	if (rank == 0){
		f = validation(&argc,argv);
		if (f != NULL){
			read_header(&header,f);
			print_header(&header);
			fclose(f);
		}
	}

	MPI_Bcast(&(header.height),1,MPI_LONG,0,MPI_COMM_WORLD);	

	p_info->n = header.height;

	calculateLocalArray(&local_n,&my_first_i,&rank);

	printf("rank: %d, my_first_i: %ld, local_n: %ld\n",rank,my_first_i,local_n);	

	


/*
	if (rank == 0){

		array = initialize(&n,&p,&t,f,&rank); //every process gets t and n, process 0 reads input
		if (rank == 0 && t == 0){
			printf("Either input file was not found or there are no test cases!\n");
		}
		while (t != 0){ //repeat until end signal is reached
			local_n = n/p;
			for (i = 0; i < t; i++){

				MPI_Barrier(MPI_COMM_WORLD);
				start = MPI_Wtime(); //starting benchmark
				local_array = divide(&local_n,&rank,array); //scatter input into processes
				aux = malloc(sizeof(int) * local_n); //auxiliar array for MergeSort
				//MergeSort(local_array, 0, local_n-1,aux); //local MergeSort
				free(aux);
				result = gather(local_n,&p,&rank,local_array); //joins all local arrays into process 0
				free(local_array);
				end = MPI_Wtime(); //ending benchmark
		
				total = end - start;
				MPI_Reduce(&total,&max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD); //maximum time elapsed among all processes 
				if (rank == 0){ //process 0 handles output
					if (i == 0 && print == 'Y') //result is printed only once
						print_local_array(&n,result); 
					free(result);
					printf("%f\n",max); //current test case execution duration (seconds)
				}		
			}
			free(array);
			array = initialize(&n,&p,&t,f,&rank); //loads new test case
		}
		if (rank == 0 && f != NULL)
			fclose(f); //when all test cases are completed, close the file

	} else {
		if (rank == 0)
			printf("Number of processes is not power of two!\n");
	*/
	MPI_Finalize();
	return 0;
}
