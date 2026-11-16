

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "stencils.cuh" // the stencils


//header declaration
int load_input(const char *filename, float *input, int N);
int save_output(const char *filename, float *output, int N);


//macro for the error checking
#define CUDA_CHECK(call) cuda_check(call, __FILE__, __LINE__)


//error checking for cuda
void cuda_check(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), file, line);

        exit(EXIT_FAILURE);
    }
}


// main
int main(int argc, char *argv[])
{
    //check if there are arguments
    if (argc != 4)
    {
        printf("Usage: ./stencil <N> <input_file> <stencil_type>\n"
               "<N>: the nubmer of inputs in a row (N x N) ex.(512 x 512) N = 512\n"
               "<input_file>: the text file containing numbers 0-255\n"
               "<stencil_type>: 1 = basic parallel stencil\n"
               "\t\t2 = shared memory stencil\n"
               "\t\t3 = register stencil\n");
        return EXIT_FAILURE;
    }

    int N = atoi(argv[1]);    // the size of the image
    const char *input_file = argv[2]; //the image file for image rending
    int stencil_type = atoi(argv[3]);

    // check if N is atleast 3, this way the smallest would by 3 x 3
    if (N <= 2)
    {
        printf("Image size must be greater than 2\n");
        return EXIT_FAILURE;
    }

    int total = N * N;
    int bytes = total * sizeof(float);


    //use malloc to set up the arrays
    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);


    //check for errors
    if (h_in == NULL || h_out == NULL)
    {
        fprintf(stderr, "Host memory allocation failed\n");

        free(h_in);
        free(h_out);

        return EXIT_FAILURE;
    }


    //load the image pixel values from <image>.txt files
    if (!load_input(input_file, h_in, N))
    {
        free(h_in);
        free(h_out);

        return EXIT_FAILURE;
    }


    //set all values to 0 in the output
    for (int i = 0; i < total; i++)
    {
        h_out[i] = 0;
    }


    float *d_in, *d_out;

    //allocate memory on the gpu
    CUDA_CHECK(cudaMalloc((void**)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, bytes));

    //set the values to be the same from host and device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, h_out, bytes, cudaMemcpyHostToDevice));

    // set up the threads and block amount
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y -1) / threads.y);


    //######################################################################
    //          Edit this by adding your type of kernel
    //  1 = basic kernel,   2 = shared kernel,  3 = register kernel
    //launch the kernel
    if (stencil_type == 1)
        parallel_stencil_kernel<<<blocks, threads>>>(d_in, d_out, N);
    else
        printf("\nNo kernel was launched\n");



    //check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // make sure the code isn't continued untill all threads are done and check errors
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy the results back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));


    //###########################################################################
    //      Add else ifs for the other kernels, you just need to 
    //      change the string in the first arguement.
    // save the newly rendered image into the results folder
    if (stencil_type == 1)
    {
        if (!save_output("result/basic_stencil/stencil_output.txt", h_out, N))
        {
            fprintf(stderr, "Failed to save stencil output\n");
        }
    }


    
    // let the user know that the program is over
    printf("Basic parallel stencil completed!\n");


    //free up memory
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    free(h_in);
    free(h_out);
    h_in = NULL;
    h_out = NULL;


    return 0;
}




// this function will use fscanf to load in the image txt to an array
int load_input(const char *filename, float *input, int N)
{
    FILE *file = fopen(filename, "r");

    if (file == NULL)
    {
        fprintf(stderr, "Could not open input file: %s\n", filename);
        return 0;
    }

    //read the image dimenstion from the first line
    int width, height;

    if (fscanf(file, "%d %d", &width, &height) != 2)
    {
        fprintf(stderr, "Could not read image dimensions\n");
        fclose(file);
        return 0;
    }

    // make sure image dimensions match N
    if (width != N || height != N)
    {
        fprintf(stderr,
                "Image dimensions (%d x %d) do not match N (%d x %d)\n",
                width, height, N, N);

        fclose(file);
        return 0;
    }

    // read pixel values into the input array
    int total = N * N;

    for (int i = 0; i < total; i++)
    {
        if (fscanf(file, "%f", &input[i]) != 1)
        {
            fprintf(stderr, "Error reading pixel value %d\n", i);

            fclose(file);
            return 0;
        }
    }

    fclose(file);

    printf("Successfully loaded image: %d x %d\n", width, height);

    return 1;
}




// this function is used to save the image bluring using stencil
// computation
int save_output(const char *filename, float *output, int N)
{
    FILE *file = fopen(filename, "w");

    //check if file had issues
    if (file == NULL)
    {
        fprintf(stderr, "Could not open input file: %s\n", filename);
        return 0;
    }

    // write into the file the image dimensions at the top
    fprintf(file, "%d %d\n", N, N);

    // write the pixel values
    int total = N * N;
    for (int i = 0; i < total; i++)
    {
        fprintf(file, "%.2f ", output[i]);

        if ((i + 1) % N == 0)   // make a new line for each row
            fprintf(file, "\n");
    }

    fclose(file);

    printf("Output saved to %s\n", filename);

    return 1;//successfuly saved file.
}

