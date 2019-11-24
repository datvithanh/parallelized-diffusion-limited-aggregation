#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define iter 800
#define N 128
#define tol 0.001
#define omega 1.9

int i, j, k, r, rank, size, Np;
int dx[4] = {-1,0,0,1}, dy[4] = {0,-1,1,0};

float *C, *Cprev, *alpha, *all_alpha;
float *Cabove, *Cbelow;
int *O, *Obelow, *candidates;
float *nutris, *nutri;

char snum[10];
char fpath[40] = "output/";
FILE *fp;
MPI_Status status;

float max(float a, float b) {
    return a>b ? a : b;
}

float r2(){
    return (float)rand() / (float)RAND_MAX;
}

void diffuse() {

    do {
        *alpha = 0;
        for(r = 0; r < 2; ++r) {
            for(i = 0; i < N; ++i) {
                *(Cabove + i) = 0;
                *(Cbelow + i) = 0;
            }

            // send the last row down
            if (rank != size - 1)
                MPI_Send(C + (Np - 1)*N, N, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD);

            // send the first row to above
            if (rank != 0) 
                MPI_Send(C, N, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD); 
            
            // receive from below
            if (rank != size - 1)
                MPI_Recv(Cbelow, N, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &status);
            
            // receive from above
            if (rank != 0)
                MPI_Recv(Cabove, N, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, &status);

            for(i = 0; i < Np; ++i){
                if(rank == 0 && i == 0)
                    continue;

                for(j = 0; j < N; ++j){
                    // general case equation
                    // *(C + i*N + j) = (omega/4)*(*(C + (i-1)*N + j) + *(C + i*N + j-1) + *(C + i*N + j+1) + *(C + (i+1)*N + j)) + (1-omega)* *(C + i*N + j);

                    if(*(O + i*N + j) == 1)
                        continue;

                    if((rank * N + i + j)%2 == r)
                        continue;

                    // Handle first row of rank 0
                    if(rank == 0 && i == 0)
                        continue;
                    
                    // Handle last row of rank size - 1
                    if(rank == size - 1 && i == Np - 1)
                        continue;

                    if(i == 0 && j == 0){
                        *(C + i*N + j) = (omega/4)*(*(Cabove + j) + *(C + i*N + N-1) + *(C + i*N + j+1) + *(C + (i+1)*N + j)) + (1-omega)* *(C + i*N + j);
                    }
                    else {
                        if(i == 0 && j == N-1){
                            *(C + i*N + j) = (omega/4)*(*(Cabove + j) + *(C + i*N + j-1) + *(C + i*N + 0) + *(C + (i+1)*N + j)) + (1-omega)* *(C + i*N + j);
                        }
                        else {
                            if(i == Np-1 && j == 0){
                                *(C + i*N + j) = (omega/4)*(*(C + (i-1)*N + j) + *(C + i*N + N-1) + *(C + i*N + j+1) + *(Cbelow + j)) + (1-omega)* *(C + i*N + j);
                            }
                            else {
                                if(i == Np - 1 && j == N-1){
                                    *(C + i*N + j) = (omega/4)*(*(C + (i-1)*N + j) + *(C + i*N + j-1) + *(C + i*N + 0) + *(Cbelow + j)) + (1-omega)* *(C + i*N + j);
                                }
                                else {
                                    if(i != 0 && i != Np - 1 && j == 0){
                                        *(C + i*N + j) = (omega/4)*(*(C + (i-1)*N + j) + *(C + i*N + N-1) + *(C + i*N + j+1) + *(C + (i+1)*N + j)) + (1-omega)* *(C + i*N + j);  
                                    }  
                                    else{
                                        if(i !=0 && i != Np - 1 && j == N-1){
                                            *(C + i*N + j) = (omega/4)*(*(C + (i-1)*N + j) + *(C + i*N + j-1) + *(C + i*N + 0) + *(C + (i+1)*N + j)) + (1-omega)* *(C + i*N + j);
                                        }
                                        else {
                                            if(i == 0 && j != 0 && j != N-1){
                                                *(C + i*N + j) = (omega/4)*(*(Cabove + j) + *(C + i*N + j-1) + *(C + i*N + j+1) + *(C + (i+1)*N + j)) + (1-omega)* *(C + i*N + j);
                                            }
                                            else {
                                                if(i == Np-1 && j != 0 && j != N-1){
                                                    *(C + i*N + j) = (omega/4)*(*(C + (i-1)*N + j) + *(C + i*N + j-1) + *(C + i*N + j+1) + *(Cbelow + j)) + (1-omega)* *(C + i*N + j);
                                                }
                                                else {
                                                    // general case
                                                    *(C + i*N + j) = (omega/4)*(*(C + (i-1)*N + j) + *(C + i*N + j-1) + *(C + i*N + j+1) + *(C + (i+1)*N + j)) + (1-omega)* *(C + i*N + j);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    *alpha = max(*alpha, fabs(*(C + i*N + j) - *(Cprev + i*N + j)));
                }
            }
        }

        MPI_Allgather(alpha, 1, MPI_FLOAT, all_alpha, 1, MPI_FLOAT, MPI_COMM_WORLD);
        
        *alpha = 0;
        for(i = 0; i < size; ++i)
            *alpha = max(*alpha, *(all_alpha + i));

        for(i=0; i < Np; ++i)
            for(j=0; j < N; ++j)
                *(Cprev + i*N + j) = *(C + i*N + j);
    } while(*alpha > tol);

    return;
}

int main(int argc, char *argv[]) 
{
    srand(time(NULL));

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

    Np = N / size;

    sprintf(snum, "%d", rank);
    strcat(snum, ".txt");
    strcat(fpath, snum);
    fp = fopen(fpath, "w");

    C = (float *) malloc(Np * N * sizeof(float));
    Cprev = (float *) malloc(Np * N * sizeof(float));
    alpha = (float *) malloc(sizeof(float));
    all_alpha = (float *) malloc(size * sizeof(float));

    Cabove = (float *) malloc(N * sizeof(float));
    Cbelow = (float *) malloc(N * sizeof(float));

    O = (int *) malloc(Np * N * sizeof(int));
    candidates = (int *) malloc(Np * N * sizeof(int));
    Obelow = (int *) malloc(N * sizeof(int));
    
    nutris = (float *) malloc(size * sizeof(float));
    nutri = (float *) malloc(sizeof(float));

    // C initialization
    for(i = 0; i < Np; ++i)
        for(j = 0; j < N; ++j){
            if (rank == 0 && i == 0)
                *(C + i*N + j) = 1;
            else
                *(C + i*N + j) = 0;

            *(Cprev + i*N + j) = *(C + i*N +j);
            *(O + i*N + j) = 0;
        }

    // Object initialization
    if(rank == size - 1)
        *(O + (Np - 1)*N + N/2) = 1;

    for(k = 0; k < iter; ++k) {
        /*
        diffuse protein (C)
        */  
        diffuse();

        for(i=0; i < Np; ++i)
            for(j=0; j < N; ++j)
                *(Cprev + i*N + j) = *(C + i*N + j);

        /*
        Grow object
        */

        for(i = 0; i < N; ++i)
            *(Obelow + i) = 0;
        // send the first row of object to above
        if (rank != 0) 
            MPI_Send(O, N, MPI_INT, rank - 1, 0, MPI_COMM_WORLD); 
                
        // receive object from below
        if (rank != size - 1)
            MPI_Recv(Obelow, N, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &status);
        
        for(i = 0; i < Np; ++i)
            for(j = 0; j < N; ++j)
                *(candidates + i*N + j) = 0;
        
        *nutri = 0;

        for(i = 0; i < Np; ++i)
            for(j = 0; j < N; ++j){
                if(*(O + i*N + j) == 1)
                    continue;

                int sum = 0;
                for(r = 0; r < 4; ++r){
                    int u, v;
                    u = i + dx[r];
                    v = j + dy[r];
                    if (u>=0 && u<Np && v>=0 && v<N && *(O + u*N + v) == 1)
                        sum += 1;
                }
                if(i == Np - 1 && *(Obelow + j) == 1)
                    sum += 1;
                
                if(sum > 0){
                    *nutri += *(C + i*N + j);
                    *(candidates + i*N + j) = 1;
                }
            }
        
        MPI_Allgather(nutri, 1, MPI_FLOAT, nutris, 1, MPI_FLOAT, MPI_COMM_WORLD);
        
        float total_nutri = 0.0;

        for(i = 0; i < size; ++i) 
            total_nutri += *(nutris+i);
        
        for(i = 0; i < Np; ++i)
            for(j = 0; j < N; ++j)
                if(*(candidates + i*N + j) == 1 && r2() <= (*(C + i*N +j)/total_nutri)){
                    *(O + i*N + j) = 1;
                    *(C + i*N + j) = 0;
                }
    }

    diffuse();

    for(i = 0; i < Np; ++i){
        for(j = 0; j < N; ++j){
            if(*(O + i*N + j) == 1)
                *(C + i*N + j) = 1;
            fprintf(fp, "%lf\t", *(C + i*N + j));
        }
        fprintf(fp, "\n");
    }

    MPI_Finalize();
	return 0;
}