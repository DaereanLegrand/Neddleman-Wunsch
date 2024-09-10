#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <omp.h>

#define MAX_SEQ_LENGTH 1100

inline int max(int a, int b) {
    return (a > b) ? a : b;
}

void readSequencesFromFile(const char *filename, std::vector<std::string>& sequences) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        std::cerr << "Error: Cannot open file." << std::endl;
        exit(1);
    }

    char line[256];
    std::string currentSeq;

    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '\n' || line[0] == '\r') continue;
        if (line[0] >= 'A' && line[0] <= 'Z') {
            if (!currentSeq.empty()) {
                sequences.push_back(currentSeq);
                currentSeq.clear();
            }
        } else {
            for (int i = 0; line[i] != '\0'; i++) {
                if (std::isalpha(line[i])) {
                    currentSeq += line[i];
                }
            }
        }
    }

    if (!currentSeq.empty()) {
        sequences.push_back(currentSeq);
    }

    fclose(file);
}

int getAlignmentCount(const std::string& a, const std::string& b) {
    int aSize = a.length() + 1;
    int bSize = b.length() + 1;

    std::vector<std::vector<int>> scoreTable(aSize, std::vector<int>(bSize, 0));

    #pragma omp parallel for
    for (int i = 0; i < aSize; i++) {
        scoreTable[i][0] = i * (-2);
    }

    #pragma omp parallel for
    for (int j = 1; j < bSize; j++) {
        scoreTable[0][j] = j * (-2);
    }

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < aSize; i++) {
        for (int j = 1; j < bSize; j++) {
            int matchScore = a[i - 1] == b[j - 1] ? 1 : -1;
            int diag = scoreTable[i - 1][j - 1] + matchScore;
            int up = scoreTable[i - 1][j] - 2;
            int left = scoreTable[i][j - 1] - 2;
            scoreTable[i][j] = max(max(diag, left), up);
        }
    }

    std::function<int(int, int)> countAlignments = [&](int i, int j) -> int {
        if (i == 0 && j == 0) return 1;
        
        int count = 0;
        int matchScore = (i > 0 && j > 0 && a[i - 1] == b[j - 1]) ? 1 : -1;

        if (i > 0 && j > 0 && scoreTable[i][j] == scoreTable[i - 1][j - 1] + matchScore) {
            count += countAlignments(i - 1, j - 1);
        }
        if (i > 0 && scoreTable[i][j] == scoreTable[i - 1][j] - 2) {
            count += countAlignments(i - 1, j);
        }
        if (j > 0 && scoreTable[i][j] == scoreTable[i][j - 1] - 2) {
            count += countAlignments(i, j - 1);
        }
        return count;
    };

    return countAlignments(aSize - 1, bSize - 1);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: program [FILE CONTAINING SEQUENCES]" << std::endl;
        return 1;
    }

    std::vector<std::string> sequences;
    readSequencesFromFile(argv[1], sequences);

    if (sequences.size() < 2) {
        std::cerr << "Error: At least two sequences are required." << std::endl;
        return 1;
    }

    double start = omp_get_wtime();
    int alignmentCount = getAlignmentCount(sequences[0], sequences[1]);
    double end = omp_get_wtime();

    std::cout << "Count of alignments: " << alignmentCount << std::endl;
    std::cout << "Elapsed time: " << (end - start) << " seconds" << std::endl;

    start = omp_get_wtime();
    alignmentCount = getAlignmentCount(sequences[1], sequences[2]);
    end = omp_get_wtime();

    std::cout << "Count of alignments: " << alignmentCount << std::endl;
    std::cout << "Elapsed time: " << (end - start) << " seconds" << std::endl;

    start = omp_get_wtime();
    alignmentCount = getAlignmentCount(sequences[0], sequences[2]);
    end = omp_get_wtime();

    std::cout << "Count of alignments: " << alignmentCount << std::endl;
    std::cout << "Elapsed time: " << (end - start) << " seconds" << std::endl;

    return 0;
}
