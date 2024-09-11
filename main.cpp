#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

int 
minu(int a, int b) 
{  
    return (a < b) ? a : b;  
} 

void 
readSequencesFromFile(const char *filename, std::vector<std::string>& sequences) {
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

void 
traceBack(const std::vector<std::vector<int>>& scoreTable, const std::vector<std::vector<char>>& arrowTable, const std::string& a, const std::string& b, int i, int j, std::string currentA, std::string currentB, int &count) {
    if (i == 0 && j == 0) {
        std::cout << "Alignment " << ++count << ":\n";
        std::cout << currentA << "\n" << currentB << "\n\n";
        return;
    }

    if (arrowTable[i][j] == 'D' || arrowTable[i][j] == 'X') {
        traceBack(scoreTable, arrowTable, a, b, i - 1, j - 1, a[i - 1] + currentA, b[j - 1] + currentB, count);
    }
    if (arrowTable[i][j] == 'U' || arrowTable[i][j] == 'X') {
        traceBack(scoreTable, arrowTable, a, b, i - 1, j, a[i - 1] + currentA, "-" + currentB, count);
    }
    if (arrowTable[i][j] == 'L' || arrowTable[i][j] == 'X') {
        traceBack(scoreTable, arrowTable, a, b, i, j - 1, "-" + currentA, b[j - 1] + currentB, count);
    }
}

int
getAlignmentCount(const std::string& a, const std::string& b) {
    int aSize = a.length() + 1;
    int bSize = b.length() + 1;

    std::vector<std::vector<int>> scoreTable(aSize, std::vector<int>(bSize, 0));
    std::vector<std::vector<char>> arrowTable(aSize, std::vector<char>(bSize, ' '));

    for (int i = 0; i < aSize; i++) {
        scoreTable[i][0] = i * (-2);
        arrowTable[i][0] = 'U';
    }

    for (int j = 0; j < bSize; j++) {
        scoreTable[0][j] = j * (-2);
        arrowTable[0][j] = 'L';
    }

    int x = aSize - 1;
    int y = bSize - 1;

    for (int line = 1; line <= (x + y - 1); line++) { 
        int start_j = std::max(0, line - x); 
        int count = std::min(line, std::min((y - start_j), x)); 

        for(int k = 0; k < count; k++) {
            int i = std::min(x, line) - k - 1;
            int j = start_j + k;

            if (i == 0 || j == 0) continue;

            int matchScore = (a[i - 1] == b[j - 1]) ? 1 : -1;
            int diag = scoreTable[i - 1][j - 1] + matchScore;
            int up = scoreTable[i - 1][j] - 2;
            int left = scoreTable[i][j - 1] - 2;

            int maxScore = std::max({diag, up, left});
            scoreTable[i][j] = maxScore;

            if (diag == maxScore) {
                arrowTable[i][j] = 'D';
            }
            if (up == maxScore) {
                if (arrowTable[i][j] == 'D') arrowTable[i][j] = 'X';
                else arrowTable[i][j] = 'U';
            }
            if (left == maxScore) {
                if (arrowTable[i][j] == 'D' || arrowTable[i][j] == 'U') arrowTable[i][j] = 'X';
                else arrowTable[i][j] = 'L';
            }
        }
    }

    int count = 0;
    traceBack(scoreTable, arrowTable, a, b, x, y, "", "", count);
    return count;
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

    std::cout << "Sample AAAC and AGC: " << getAlignmentCount("AAAC", "AGC") << std::endl;
    
    std::cout << "Bacteria: " << sequences[0].size() << std::endl;
    std::cout << "Sars-Cov: " << sequences[1].size() << std::endl;
    std::cout << "Influenza: " << sequences[2].size() << std::endl;

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
