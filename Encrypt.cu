#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// CUDA function to perform encryption using custom rules
__device__ char* CudaCrypt(char* rawPassword) {
    char *newPassword = (char *) malloc(sizeof(char) * 11);

    newPassword[0] = rawPassword[0] + 2;
    newPassword[1] = rawPassword[0] - 2;
    newPassword[2] = rawPassword[0] + 1;
    newPassword[3] = rawPassword[1] + 3;
    newPassword[4] = rawPassword[1] - 3;
    newPassword[5] = rawPassword[1] - 1;
    newPassword[6] = rawPassword[2] + 2;
    newPassword[7] = rawPassword[2] - 2;
    newPassword[8] = rawPassword[3] + 4;
    newPassword[9] = rawPassword[3] - 4;
    newPassword[10] = '\0';

    // Apply limits for lowercase and numbers
    for (int i = 0; i < 10; i++) {
        if (i >= 0 && i < 6) {  // Lowercase letters
            if (newPassword[i] > 122) {
                newPassword[i] = (newPassword[i] - 122) + 97;
            } else if (newPassword[i] < 97) {
                newPassword[i] = (97 - newPassword[i]) + 97;
            }
        } else {  // Digits (0-9)
            if (newPassword[i] > 57) {
                newPassword[i] = (newPassword[i] - 57) + 48;
            } else if (newPassword[i] < 48) {
                newPassword[i] = (48 - newPassword[i]) + 48;
            }
        }
    }

    return newPassword;
}

// CUDA kernel to apply CudaCrypt function on a password
__global__ void encrypt_kernel(char *inputPassword, char *encryptedPassword) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Assume we are only processing 4 characters for simplicity here
    if (index < 1) {
        char rawPassword[4];
        rawPassword[0] = inputPassword[0];
        rawPassword[1] = inputPassword[1];
        rawPassword[2] = inputPassword[2];
        rawPassword[3] = inputPassword[3];

        char* encrypted = CudaCrypt(rawPassword);

        // Copy the encrypted password back to global memory
        for (int i = 0; i < 10; i++) {
            encryptedPassword[i] = encrypted[i];
        }
    }
}

int main(int argc, char *argv[]) {
    // Ensure a password is provided as argument
    if (argc < 2) {
        printf("Usage: %s <password>\n", argv[0]);
        return 1;
    }

    // Allocate memory for the output encrypted password
    char encryptedPassword[11];  // 10 characters + null terminator

    // Allocate device memory
    char *d_inputPassword, *d_encryptedPassword;
    cudaMalloc((void**)&d_inputPassword, sizeof(char) * 4);  // Assuming password is 4 characters
    cudaMalloc((void**)&d_encryptedPassword, sizeof(char) * 11);

    // Copy the input password to device memory
    cudaMemcpy(d_inputPassword, argv[1], sizeof(char) * 4, cudaMemcpyHostToDevice);

    // Launch kernel
    encrypt_kernel<<<1, 1>>>(d_inputPassword, d_encryptedPassword);
    cudaDeviceSynchronize();

    // Copy the result back to host memory
    cudaMemcpy(encryptedPassword, d_encryptedPassword, sizeof(char) * 11, cudaMemcpyDeviceToHost);

    // Print the result (in plain text)
    printf("Encrypted Password: %s\n", encryptedPassword);

    // Free device memory
    cudaFree(d_inputPassword);
    cudaFree(d_encryptedPassword);

    return 0;
}
