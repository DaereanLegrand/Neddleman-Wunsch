#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

// Scoring constants
const int MATCH = 2;
const int MISMATCH = -1;
const int GAP = -2;

// CUDA kernel for filling the scoring matrix
__global__ void fillMatrix(int* matrix, char* seq1, char* seq2, int len1, int len2, int pitch) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i <= len1 && j <= len2) {
        int idx = i * pitch + j;
        int match = (seq1[i-1] == seq2[j-1]) ? MATCH : MISMATCH;
        int score = max(max(
            matrix[(i-1) * pitch + j] + GAP,
            matrix[i * pitch + (j-1)] + GAP),
            matrix[(i-1) * pitch + (j-1)] + match
        );
        matrix[idx] = score;
    }
}

// Function to perform Needleman-Wunsch alignment
std::vector<std::pair<std::string, std::string>> needlemanWunsch(const std::string& seq1, const std::string& seq2) {
    int len1 = seq1.length();
    int len2 = seq2.length();

    // Allocate and initialize device memory
    thrust::device_vector<char> d_seq1(seq1.begin(), seq1.end());
    thrust::device_vector<char> d_seq2(seq2.begin(), seq2.end());

    int pitch;
    int* d_matrix;
    cudaMallocPitch(&d_matrix, (size_t*)&pitch, (len2 + 1) * sizeof(int), len1 + 1);
    pitch /= sizeof(int);

    // Initialize first row and column
    thrust::device_vector<int> d_init(len2 + 1);
    for (int i = 0; i <= len2; ++i) {
        d_init[i] = i * GAP;
    }
    cudaMemcpy2D(d_matrix, pitch * sizeof(int), thrust::raw_pointer_cast(d_init.data()),
                 (len2 + 1) * sizeof(int), (len2 + 1) * sizeof(int), 1, cudaMemcpyDeviceToDevice);

    for (int i = 1; i <= len1; ++i) {
        cudaMemcpy(&d_matrix[i * pitch], &i, sizeof(int), cudaMemcpyHostToDevice);
    }

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((len2 + blockSize.x - 1) / blockSize.x, (len1 + blockSize.y - 1) / blockSize.y);
    fillMatrix<<<gridSize, blockSize>>>(d_matrix, thrust::raw_pointer_cast(d_seq1.data()),
                                        thrust::raw_pointer_cast(d_seq2.data()),
                                        len1, len2, pitch);

    // Copy results back to host
    std::vector<int> h_matrix((len1 + 1) * (len2 + 1));
    cudaMemcpy2D(h_matrix.data(), (len2 + 1) * sizeof(int), d_matrix,
                 pitch * sizeof(int), (len2 + 1) * sizeof(int), len1 + 1, cudaMemcpyDeviceToHost);

    // Traceback to find alignments
    std::vector<std::pair<std::string, std::string>> alignments;
    std::function<void(int, int, std::string, std::string)> traceback = [&](int i, int j, std::string aligned1, std::string aligned2) {
        if (i == 0 && j == 0) {
            alignments.push_back({aligned1, aligned2});
            return;
        }

        int score = h_matrix[i * (len2 + 1) + j];
        int diag = (i > 0 && j > 0) ? h_matrix[(i-1) * (len2 + 1) + (j-1)] : INT_MIN;
        int up = (i > 0) ? h_matrix[(i-1) * (len2 + 1) + j] : INT_MIN;
        int left = (j > 0) ? h_matrix[i * (len2 + 1) + (j-1)] : INT_MIN;

        if (i > 0 && j > 0 && score == diag + ((seq1[i-1] == seq2[j-1]) ? MATCH : MISMATCH)) {
            traceback(i-1, j-1, seq1[i-1] + aligned1, seq2[j-1] + aligned2);
        }
        if (i > 0 && score == up + GAP) {
            traceback(i-1, j, seq1[i-1] + aligned1, '-' + aligned2);
        }
        if (j > 0 && score == left + GAP) {
            traceback(i, j-1, '-' + aligned1, seq2[j-1] + aligned2);
        }
    };

    traceback(len1, len2, "", "");

    // Clean up
    cudaFree(d_matrix);

    return alignments;
}

// Function to return only one alignment
std::pair<std::string, std::string> needlemanWunschSingle(const std::string& seq1, const std::string& seq2) {
    auto alignments = needlemanWunsch(seq1, seq2);
    return alignments[0];
}

int main() {
    std::string bacteria = "tcaagcgtta gagaagtcat tatgtgataa aaaaattcaa cttggtatca acttaactaa gggtcttggt gctggtgctt tgcctgatgt tggtaaaggt gcagcagaag aatcaattga";
    std::string sars_cov = "attaaaggtt tataccttcc caggtaacaa accaaccaac tttcgatctc ttgtagatct gttctctaaa cgaactttaa aatctgtgtg gctgtcactc ggctgcatgc";
    std::string influenza = "atggaagcaa tatcactgat gactatacta ctggtggtaa caacaagtaa tgcagacaaa atctgcatcg gtcaccaatc aacaaattcc acggaaactg tagacacgct";

    // Perform alignments
    auto alignments1 = needlemanWunsch(bacteria, sars_cov);
    auto alignments2 = needlemanWunsch(sars_cov, influenza);
    auto alignments3 = needlemanWunsch(bacteria, influenza);

    // Print results
    std::cout << "Bacteria vs SARS-CoV alignments: " << alignments1.size() << std::endl;
    std::cout << "SARS-CoV vs Influenza alignments: " << alignments2.size() << std::endl;
    std::cout << "Bacteria vs Influenza alignments: " << alignments3.size() << std::endl;

    // Example of using single alignment function
    auto single_alignment = needlemanWunschSingle(bacteria, sars_cov);
    std::cout << "Single alignment example:" << std::endl;
    std::cout << single_alignment.first << std::endl;
    std::cout << single_alignment.second << std::endl;

    return 0;
}
