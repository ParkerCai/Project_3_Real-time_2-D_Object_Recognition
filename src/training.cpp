/*
  Jenny Nguyen
  Parker Cai
  February 20, 2026
  CS5330 - Project 3: Real-time 2-D Object Recognition

  Training mode - collect and store feature vectors
*/

#include "or2d.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <print>

/*
    Save training example to the database file.
    Appends label + features to csv
*/
void saveTrainingExample(const std::string& filename,
  const std::string& label,
  const std::vector<double>& features) {
  std::ofstream file(filename, std::ios::app);

  if (!file.is_open()) {
    std::println("Error: can't open {}", filename);
    return;
  }

  file << label;
  for (const double& f : features) {
    file << "," << f;
  }
  file << "\n";

  std::println("Saved: {}", label);
}

// reads training data back from csv into labels + feature vectors
// returns how many examples were loaded
int loadTrainingData(const std::string& filename,
  std::vector<std::string>& labels,
  std::vector<std::vector<double>>& features) {
  labels.clear();
  features.clear();

  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cout << "No database found at " << filename << std::endl;
    return 0;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::vector<double> fvec;
    std::string label;
    size_t pos;

    pos = line.find(',');
    if (pos == std::string::npos) continue;

    label = line.substr(0, pos);
    line = line.substr(pos + 1);

    while (!line.empty()) {
      pos = line.find(',');
      std::string tok;

      if (pos == std::string::npos) {
        tok = line;
        line = "";
      }
      else {
        tok = line.substr(0, pos);
        line = line.substr(pos + 1);
      }
      fvec.push_back(std::stod(tok));
    }

    labels.push_back(label);
    features.push_back(fvec);
  }

  std::println("Loaded {} examples", labels.size());
  return labels.size();
}

// call this once to set up a fresh database file
void initializeDatabase(const std::string& filename) {
  std::ofstream file(filename);

  if (!file.is_open()) {
    std::println("Error: can't create {}", filename);
    return;
  }

  file << "# Object Recognition Training Database\n";
  file << "# label,percentFilled,bboxRatio,hu0,hu1,...\n";

  std::println("Created database: {}", filename);
}

// Float overloads for CNN embedding data
void saveTrainingExample(const std::string& filename,
  const std::string& label,
  const std::vector<float>& features) {
  std::ofstream file(filename, std::ios::app);
  if (!file.is_open()) {
    std::println("Error: can't open {}", filename);
    return;
  }
  file << label;
  for (const float& f : features) {
    file << "," << f;
  }
  file << "\n";
  std::println("Saved: {}", label);
}

int loadTrainingData(const std::string& filename,
  std::vector<std::string>& labels,
  std::vector<std::vector<float>>& features) {
  labels.clear();
  features.clear();
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::println("No database found at {}", filename);
    return 0;
  }
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::vector<float> fvec;
    std::string label;
    size_t pos;
    pos = line.find(',');
    if (pos == std::string::npos) continue;
    label = line.substr(0, pos);
    line = line.substr(pos + 1);
    while (!line.empty()) {
      pos = line.find(',');
      std::string tok;
      if (pos == std::string::npos) {
        tok = line;
        line = "";
      }
      else {
        tok = line.substr(0, pos);
        line = line.substr(pos + 1);
      }
      fvec.push_back(std::stof(tok));
    }
    labels.push_back(label);
    features.push_back(fvec);
  }
  std::println("Loaded {} examples", labels.size());
  return labels.size();
}