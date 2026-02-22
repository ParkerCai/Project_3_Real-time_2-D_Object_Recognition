/*
  Jenny Nguyen
  Parker Cai
  February 19, 2026
  CS5330 - Project 3: Real-time 2-D Object Recognition

  Confusion matrix for evaluation of classification results.
*/

#include "or2d.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iomanip>

// adds a new class label if we haven't seen it yet
// needed because we don't know all classes upfront
void addClassToMatrix(ConfusionMatrix& cm, const std::string& name) {
    if(cm.class_index.find(name) == cm.class_index.end()) {
        int idx = cm.classes.size();
        cm.classes.push_back(name);
        cm.class_index[name] = idx;
        
        // have to resize every existing row too, not just add a new one
        cm.matrix.resize(cm.classes.size());
        for(auto& row : cm.matrix) {
            row.resize(cm.classes.size(), 0);
        }
    }
}

void addResultToMatrix(ConfusionMatrix& cm, const std::string& true_label, const std::string& pred_label) {
    // make sure both labels exist before we try to index into anything
    addClassToMatrix(cm, true_label);
    addClassToMatrix(cm, pred_label);
    
    int ti = cm.class_index[true_label];
    int pi = cm.class_index[pred_label];
    cm.matrix[ti][pi]++;
}

void printConfusionMatrix(ConfusionMatrix& cm) {
    std::cout << "\n=== Confusion Matrix ===" << std::endl;
    
    // column headers
    std::cout << std::setw(18) << "";
    for(size_t i = 0; i < cm.classes.size(); i++) {
        std::cout << std::setw(18) << cm.classes[i];
    }
    std::cout << std::endl;
    
    // divider depending on how many classes there are
    int divlen = 18 + cm.classes.size() * 18;
    for(int k = 0; k < divlen; k++) std::cout << "-";
    std::cout << std::endl;
    
    for(size_t i = 0; i < cm.matrix.size(); i++) {
        std::cout << std::setw(18) << cm.classes[i];
        for(size_t j = 0; j < cm.matrix[i].size(); j++) {
            std::cout << std::setw(18) << cm.matrix[i][j];
        }
        std::cout << std::endl;
    }
    
    // overall accuracy
    int correct = 0, total = 0;
    for(size_t i = 0; i < cm.matrix.size(); i++) {
        correct += cm.matrix[i][i];
        for(size_t j = 0; j < cm.matrix[i].size(); j++)
            total += cm.matrix[i][j];
    }
    
    double acc = (total > 0) ? (100.0 * correct) / total : 0.0;
    std::cout << "\nOverall Accuracy: " << std::fixed << std::setprecision(1) << acc << "%" << std::endl;
    
    // break it down per class so we can see which ones are doing well and which ones are struggling
    std::cout << "\nPer-Class Accuracy:" << std::endl;
    for(size_t i = 0; i < cm.matrix.size(); i++) {
        int row_total = 0;
        for(size_t j = 0; j < cm.matrix[i].size(); j++)
            row_total += cm.matrix[i][j];
        
        double class_acc = (row_total > 0) ? (100.0 * cm.matrix[i][i]) / row_total : 0.0;
        std::cout << "  " << cm.classes[i] << ": " 
                  << std::fixed << std::setprecision(1) << class_acc << "%" << std::endl;
    }
    std::cout << "========================\n" << std::endl;
}

// save matrix to csv so we can open it in a spreadsheet if needed
void saveConfusionMatrix(ConfusionMatrix& cm, const std::string& filename) {
    std::ofstream file(filename);
    
    file << "True\\Predicted";
    for(size_t i = 0; i < cm.classes.size(); i++)
        file << "," << cm.classes[i];
    file << "\n";
    
    for(size_t i = 0; i < cm.matrix.size(); i++) {
        file << cm.classes[i];
        for(size_t j = 0; j < cm.matrix[i].size(); j++)
            file << "," << cm.matrix[i][j];
        file << "\n";
    }
    
    file.close();
    std::cout << "Saved confusion matrix to " << filename << std::endl;
}