#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int read_input(const char *filename, double** mat_a, double** mat_b);

int write_output(char* filename, const double* mat_c, int n);

int main(int argc, char **argv) {

	if (3 != argc) {
		printf("Wrong input, should be: inputfile outputfile\n");
		return 1;
	}

	char *inputfile = argv[1];
	char *outputfile = argv[2];

	// Initialize MPI
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Read input file on process 0
	double *input;
	int n;
	double *mat_a, *mat_b, *mat_c;
	if (rank == 0){
		n = read_input(inputfile, &mat_a, &mat_b);
		mat_c = malloc(n * n * sizeof(double));
	}

	// Broadcast matrix dimension n to all processes
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// create data needed for all processes
	int subsize = n / size; //Number of rows/columns per process
	double *sub_a = (double *)malloc(n * n / size * sizeof(double));
	double *sub_b = (double *)malloc(n * n / size * sizeof(double));
	double *sub_c = (double *)malloc(n * n / size * sizeof(double));
	double *sub_tmp = (double *)malloc(n * n / size * sizeof(double));

	// Initialize new datastructures
	MPI_Datatype row_datatype, col_datatype, tmp_datatype;
	MPI_Type_vector(n, 1, n, MPI_DOUBLE, &tmp_datatype); //Create a vector type for storing columns
	MPI_Type_commit(&tmp_datatype);
	MPI_Type_contiguous(n, MPI_DOUBLE, &row_datatype); //rows can be stored contiguously in memory
	MPI_Type_create_resized(tmp_datatype, 0, sizeof(double), &col_datatype); // columns can not be stored contiguously.
	MPI_Type_commit(&row_datatype);
	MPI_Type_commit(&col_datatype);

	// Needed for communication between processes
	MPI_Status statuses[size];
	MPI_Request requests[size];

	// Start timer
	double start = MPI_Wtime();

	// Scatter data to all processes
	MPI_Scatter(mat_a, subsize, row_datatype, sub_a, subsize*n, MPI_DOUBLE, 0, MPI_COMM_WORLD); //Scatters rows of A
	MPI_Scatter(mat_b, subsize, col_datatype, sub_b, subsize*n, MPI_DOUBLE, 0, MPI_COMM_WORLD); //Scatters columns of B

	// Start non-blocking communication
	for (int i = 0; i < size; i++){
		MPI_Isend(sub_b, subsize*n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &requests[i]);
	}

	// Perform matrix multiplication between A and B and save result in matrix C
	for (int r = 0; r < size; r++){
		int current_rank = (rank + r) % size;

		// OBS: Blocking communication
		MPI_Recv(sub_tmp, subsize*n, MPI_DOUBLE, current_rank, 0, MPI_COMM_WORLD, &statuses[r]);

		// Calculate part of matrix multiplication
		for (int i = 0; i < subsize; i++){
			for (int j = 0; j < subsize; j++){
				int col = current_rank*subsize + j;
				sub_c[i*n + col] = 0;
				for (int k = 0; k < n; k++){
					sub_c[i*n + col] += sub_a[i*n + k] * sub_tmp[j*n + k];
				}
			}
		}
	}

	// Wait for all processes to finish
	MPI_Waitall(size, requests, statuses);
	
	// Process 0 collects result
	MPI_Gather(sub_c, subsize*n, MPI_DOUBLE, mat_c, subsize*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//print time and write results
	if (rank == 0){
		double end = MPI_Wtime();
		double total = end-start;
		printf("%f", total);

		// Write to file
		write_output(outputfile, mat_c, n);

		// Free memory held by process 0
		free(mat_a);
		free(mat_b);
		free(mat_c);
	}
	// Free all local memory on all processes
	free(sub_a);
	free(sub_b);
	free(sub_c);
	free(sub_tmp);

	// Free data types
	MPI_Type_free(&tmp_datatype);
	MPI_Type_free(&col_datatype);
	MPI_Type_free(&row_datatype);
	

	MPI_Finalize();

	return 0;
}

int read_input(const char *filename, double** mat_a, double** mat_b){
	FILE* fp;

	fp = fopen(filename, "r");
	if (fp == NULL) {
		fprintf(stderr, "Could not open input file\n");
		return -1;
	}

	// Read matrix dimension from first number in file
	int n;
	if (EOF == fscanf(fp, "%d", &n)) {
		printf("Could not read input dimension\n");
		return -1;
	}

	// Allocate memory for both matrices
	*mat_a = (double*) malloc(n * n * sizeof(double));
	*mat_b = (double*) malloc(n * n * sizeof(double));

	// Read matrix A
	for (int i = 0; i < n*n; i++){
		if (EOF == fscanf(fp, "%lf", &((*mat_a)[i]))){
			printf("Error reading matrix A elements\n");
			return -1;
		}
	}

	// Read matrix B
	for (int i = 0; i < n*n; i++){
		if (EOF == fscanf(fp, "%lf", &((*mat_b)[i]))){
			printf("Error reading matrix B elements\n");
			return -1;
		}
	}

	fclose(fp);

	return n;
}

int write_output(char* filename, const double* mat_c, int n){
	FILE *fp;
	fp = fopen(filename, "w");

	if (fp == NULL) {
		fprintf(stderr, "Could not open output file\n");
		return -1;
	}

	// Write matrix C to file in row-major order
	for (int i = 0; i < n*n; i++){
		if (0 > fprintf(fp, "%.6f ", mat_c[i])){
			printf("Could not write to output file\n");
		}
	}

	if (0 != fclose(fp)){
		printf("Could not close output file\n");
		return -1;
	}
	return 0;
}

