# README: Running Encryption and Password Cracking Program

## Introduction
This program encrypts a given password using CUDA and attempts to crack it by brute force. The encryption follows a specific pattern where the password consists of 2 letters followed by 2 digits (e.g., `ab00`).

## Prerequisites
Before running the program, ensure you have the necessary dependencies installed.

### Install CUDA (for NVIDIA GPUs)
1. Download CUDA Toolkit from [NVIDIA's official site](https://developer.nvidia.com/cuda-downloads).
2. Follow the installation instructions specific to your operating system.
3. Verify installation by running:
   ```sh
   nvcc --version
   ```

## Compiling the Program
Navigate to the directory containing the source file and compile it using `nvcc`:
```sh
nvcc Encrypt.cu -o Encrypt
nvcc passwordCrack.cu -o passwordCrack
```
This will generate two executable files: `Encrypt` and `passwordCrack`.

## Running the Encryption Program
To encrypt a password, run:
```sh
./Encrypt <password>
```
Replace `<password>` with a combination of 2 letters and 2 digits (e.g., `ab00`).

Example:
```sh
./Encrypt hp22
```
This will output the encrypted form of `ab00`.

## Running the Password Cracking Program
To attempt to crack an encrypted password, run:
```sh
./passwordCrack
```
You will be prompted to enter the encrypted password. Once entered, the program will try to find the original password.

Example Output:
```sh
Enter the password in encrypted form: <encrypted_string>
Password found: hp22
Time taken: 0.000184 seconds
```

## Notes
- Ensure you have execution permissions for the compiled binaries:
  ```sh
  chmod +x Encrypt passwordCrack
  ```
- The password cracking program uses brute force, so it may take time depending on the complexity of the encryption and available GPU resources.
- Make sure CUDA is properly installed and configured for optimal performance.



