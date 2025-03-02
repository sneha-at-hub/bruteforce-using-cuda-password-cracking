#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CUDA functions for string copying and comparison, designed for GPU usage
// We cannot use standard C functions like strcpy() or strcmp() in GPU code

// Function to copy a string from source to destination on the GPU
__device__ char * copyStr(char *dest, const char *src){
  int i = 0;
    // Copy each character from the source to the destination until the null terminator is reached
  do {
    dest[i] = src[i];
  }
  while (src[i++] != 0); // Increment index until null terminator is found
  return dest; // Return pointer to the destination string
}

// Function to compare two strings on the GPU
__device__ bool compareStr(const char *strA, const char *strB, unsigned len = 11){
  unsigned i = 0;
  // Loop through the characters of both strings to compare them
  while (i < len) {
    if (strA[i] != strB[i]) {
      return false;  // Return false immediately if characters don't match
    }
    i++;
  }
  return true;  // Return true if all characters match
}

// Kernel function for encrypting the password
__device__ void cudaCrypt(char *newPassword, const char *rawPassword) {
  // Encrypt the generated password
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

  // Apply range checks to ensure characters stay within certain limits

  for (int i = 0; i < 10; i++) {
    if (i >= 0 && i < 6) {
      // Ensure characters remain in the lowercase 'a' to 'z' range
      if (newPassword[i] > 122) {
        newPassword[i] = (newPassword[i] - 122) + 97;  // Wrap around if greater than 'z'
      } else if (newPassword[i] < 97) {
        newPassword[i] = (97 - newPassword[i]) + 97;   // Wrap around if less than 'a'
      }
    } else {
      // Ensure characters remain in the digit '0' to '9' range
      if (newPassword[i] > 57) {
        newPassword[i] = (newPassword[i] - 57) + 48; // Wrap around if greater than '9'
      } else if (newPassword[i] < 48) {
        newPassword[i] = (48 - newPassword[i]) + 48;  // Wrap around if less than '0'
      }
    }
  }
}

// Kernel function to try all possible combinations for cracking the password
__global__ void findPassword(char *D_chars, char *D_digits, char *D_encPwd, bool *D_passwordFound) {
  char rawPassword[4];
  char newPassword[11];  // To hold the encrypted password

  // Generate a password combination using grid and thread indices
  // The grid's blockIdx.x and blockIdx.y will determine which characters are picked from D_chars
  // The thread's threadIdx.x and threadIdx.y will determine which digits are picked from D_digits
  rawPassword[0] = D_chars[blockIdx.x]; // character based on blockIdx.x
  rawPassword[1] = D_chars[blockIdx.y]; // character based on blockIdx.y
  rawPassword[2] = D_digits[threadIdx.x]; // digit based on threadIdx.x
  rawPassword[3] = D_digits[threadIdx.y]; // digit based on threadIdx.y

  // Encrypt the generated password using the cudaCrypt function
  cudaCrypt(newPassword, rawPassword);

  // Compare the generated encrypted password with the target encrypted password
  if (compareStr(newPassword, D_encPwd)) {
    // Password matched, set flag to true
    *D_passwordFound = true;
    copyStr(D_encPwd, rawPassword);  // Store the decrypted password
  }
}

int main(int argc, char **argv) {
  // Define possible characters and digits for password cracking
  char H_availableChars[26] = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'};
  char H_availableDigits[26] = {'0','1','2','3','4','5','6','7','8','9'};
  char H_newPassword[11];  // For user input
  char * H_decryptedPwd = (char *)malloc(sizeof(char) * 11);  // Ensure we allocate enough space
  bool * H_passwordFound = (bool *)malloc(sizeof(bool));  // Flag for password found

  // Get the encrypted password from the user, ensuring it's not empty
  while (1) {
    printf("Enter the Encrypted password: ");
    fgets(H_newPassword, sizeof(H_newPassword), stdin);
    // Remove newline character if any
    if (H_newPassword[strlen(H_newPassword) - 1] == '\n') {
      H_newPassword[strlen(H_newPassword) - 1] = '\0';
    }

    if (strlen(H_newPassword) > 0) {
      break;  // Exit loop if password is not empty
    } else {
      printf("The encryption cannot be empty. Please write your encrypted password: \n");
    }
  }

  // Allocate memory on GPU (Device memory)
  char * D_chars, * D_digits, * D_encPwd;
  bool * D_passwordFound;
  cudaMalloc((void**)&D_chars, sizeof(char) * 26);
  cudaMemcpy(D_chars, H_availableChars, sizeof(char) * 26, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&D_digits, sizeof(char) * 26);
  cudaMemcpy(D_digits, H_availableDigits, sizeof(char) * 26, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&D_encPwd, sizeof(char) * 11);  
  cudaMemcpy(D_encPwd, H_newPassword, sizeof(char) * 11, cudaMemcpyHostToDevice);

  // Allocate memory for passwordFound flag
  cudaMalloc((void**)&D_passwordFound, sizeof(bool));
  cudaMemcpy(D_passwordFound, H_passwordFound, sizeof(bool), cudaMemcpyHostToDevice);

  // Start measuring time
  cudaEvent_t startTime, endTime;
  float elapsedTime;
  cudaEventCreate(&startTime);
  cudaEventCreate(&endTime);
  cudaEventRecord(startTime);

  // Launch the kernel to crack the password
  findPassword<<<dim3(26, 26, 1), dim3(10, 10, 1)>>>(D_chars, D_digits, D_encPwd, D_passwordFound);
  cudaDeviceSynchronize();  // Wait for the kernel to finish execution

  cudaEventRecord(endTime);
  cudaEventSynchronize(endTime);
  cudaEventElapsedTime(&elapsedTime, startTime, endTime);

  // Copy the flag back to host memory
  cudaMemcpy(H_passwordFound, D_passwordFound, sizeof(bool), cudaMemcpyDeviceToHost);
  cudaMemcpy(H_decryptedPwd, D_encPwd, sizeof(char) * 11, cudaMemcpyDeviceToHost);  

  if (*H_passwordFound) {
    printf("\nPassword Found!");
    printf("\nDecrypted Password: %s\n", H_decryptedPwd);
  } else {
    printf("\nPassword not found.\n");
  }

  printf("Time taken: %fs\n", elapsedTime / 1000);

  // Clean up
  free(H_decryptedPwd);
  free(H_passwordFound);
  cudaFree(D_chars);
  cudaFree(D_digits);
  cudaFree(D_encPwd);
  cudaFree(D_passwordFound);
  cudaEventDestroy(startTime);
  cudaEventDestroy(endTime);

  return 0;
}