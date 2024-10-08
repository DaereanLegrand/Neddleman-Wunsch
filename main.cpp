#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <stack>
#include <omp.h>

int 
minu(int a, int b) 
{  
    return (a < b) ? a : b;  
} 

void
printTable(const std::vector<std::vector<int>>& scoreTable)
{
    for (auto i: scoreTable) {
        for (auto j: i) {
            printf(" %3d \t", j);
        }
        printf("\n");
    }
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
traceback(const std::vector<std::vector<int>>& scoreTable, const std::string& a, const std::string& b, 
               int startI, int startJ, std::vector<std::pair<std::string, std::string>>& alignments) {
    std::stack<std::tuple<int, int, std::string, std::string>> stk;
    stk.push(std::make_tuple(startI, startJ, "", ""));

    while (!stk.empty()) {
        auto [i, j, currentAlignmentA, currentAlignmentB] = stk.top();
        stk.pop();

        if (i == 0 && j == 0) {
            std::string finalAlignmentA = currentAlignmentA;
            std::string finalAlignmentB = currentAlignmentB;
            std::reverse(finalAlignmentA.begin(), finalAlignmentA.end());
            std::reverse(finalAlignmentB.begin(), finalAlignmentB.end());
            alignments.push_back({finalAlignmentA, finalAlignmentB});
            continue;
        }

        if (i > 0 && j > 0 && scoreTable[i][j] == scoreTable[i - 1][j - 1] + ((a[i - 1] == b[j - 1]) ? 1 : -1)) {
            stk.push(std::make_tuple(i - 1, j - 1, currentAlignmentA + a[i - 1], currentAlignmentB + b[j - 1]));
        }

        if (i > 0 && scoreTable[i][j] == scoreTable[i - 1][j] - 2) {
            stk.push(std::make_tuple(i - 1, j, currentAlignmentA + a[i - 1], currentAlignmentB + "-"));
        }

        if (j > 0 && scoreTable[i][j] == scoreTable[i][j - 1] - 2) {
            stk.push(std::make_tuple(i, j - 1, currentAlignmentA + "-", currentAlignmentB + b[j - 1]));
        }
    }
}

int
getAlignmentCount(const std::string& a, const std::string& b, bool bPrinting) {
    int aSize = a.length() + 1;
    int bSize = b.length() + 1;

    std::vector<std::vector<int>> scoreTable(aSize, std::vector<int>(bSize, 0));

    for (int i = 0; i < aSize; i++) {
        scoreTable[i][0] = i * (-2);
    }

    for (int j = 0; j < bSize; j++) {
        scoreTable[0][j] = j * (-2);
    }

    int x = aSize - 1;
    int y = bSize - 1;

    for (int line = 1; line <= (x + y - 1); line++) { 
        int start_j = std::max(0, line - x); 
        int count = std::min(line, std::min((y - start_j), x)); 

        for(int k = 0; k < count; k++) {
            int i = std::min(x, line) - k - 1;
            int j = start_j + k;

            i += 1;
            j += 1;

            if (i == 0 || j == 0) continue;

            int matchScore = (a[i - 1] == b[j - 1]) ? 1 : -1;
            int diag = scoreTable[i - 1][j - 1] + matchScore;
            int up = scoreTable[i - 1][j] - 2;
            int left = scoreTable[i][j - 1] - 2;

            int maxScore = std::max({diag, up, left});
            scoreTable[i][j] = maxScore;

        }
    }


    std::vector<std::pair<std::string, std::string>> alignments;
    traceback(scoreTable, a, b, aSize - 1, bSize - 1, alignments);

    if (bPrinting) {
        printTable(scoreTable);
        for (const auto& alignment : alignments) {
            std::cout << "Alignment 1:\n" << alignment.first << "\n" << alignment.second << "\n";
        }
    }

    return alignments.size();
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

    std::cout << "Sample AAAC and AGC: \n" << getAlignmentCount("AAAC", "AGC", 1) << std::endl;
    
    std::cout << "Bacteria: " << sequences[0].size() << std::endl;
    std::cout << "Sars-Cov: " << sequences[1].size() << std::endl;
    std::cout << "Influenza: " << sequences[2].size() << std::endl;

    double start = omp_get_wtime();
    int alignmentCount = getAlignmentCount(sequences[0], sequences[1], 0);
    double end = omp_get_wtime();

    std::cout << "Count of alignments: " << alignmentCount << std::endl;
    std::cout << "Elapsed time: " << (end - start) << " seconds" << std::endl;

    start = omp_get_wtime();
    alignmentCount = getAlignmentCount(sequences[1], sequences[2], 0);
    end = omp_get_wtime();

    std::cout << "Count of alignments: " << alignmentCount << std::endl;
    std::cout << "Elapsed time: " << (end - start) << " seconds" << std::endl;

    start = omp_get_wtime();
    alignmentCount = getAlignmentCount(sequences[0], sequences[2], 0);
    end = omp_get_wtime();

    std::cout << "Count of alignments: " << alignmentCount << std::endl;
    std::cout << "Elapsed time: " << (end - start) << " seconds" << std::endl;
    return 0;
}
