/*
  Jenny Nguyen
  Parker Cai
  February 2026
  CS5330 - Project 3: Real-time 2-D Object Recognition

  OR2D GUI Application using Dear ImGui
*/

#include <iostream>
#include <vector>
#include <string>
#include <print>
#include <filesystem>
#include <chrono>
#include <fstream>
#include <map>
#include <opencv2/opencv.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl2.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <GL/gl.h>
#include <dwmapi.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "dwmapi.lib")
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif
#endif

#include "or2d.h"
#include "utilities.h"

// ============================================================================
// Helpers
// ============================================================================

static double getTime() {
  using namespace std::chrono;
  auto now = high_resolution_clock::now();
  return duration_cast<duration<double>>(now.time_since_epoch()).count();
}

static GLuint matToTexture(const cv::Mat& mat, int& outWidth, int& outHeight) {
  if (mat.empty()) return 0;
  cv::Mat rgb;
  if (mat.channels() == 1)
    cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB);
  else
    cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);

  GLuint textureId;
  glGenTextures(1, &textureId);
  glBindTexture(GL_TEXTURE_2D, textureId);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb.cols, rgb.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
  outWidth = rgb.cols;
  outHeight = rgb.rows;
  return textureId;
}

static void freeTexture(GLuint& textureId) {
  if (textureId != 0) { glDeleteTextures(1, &textureId); textureId = 0; }
}

// Rewrite CSV with remaining rows (for features DB: double; for CNN DB: float)
static void rewriteFeaturesCsv(const std::string& filename,
  const std::vector<std::string>& labels,
  const std::vector<std::vector<double>>& features) {
  std::ofstream file(filename);
  if (!file.is_open()) return;
  for (size_t i = 0; i < labels.size(); i++) {
    file << labels[i];
    for (double v : features[i]) file << "," << v;
    file << "\n";
  }
}

static void rewriteCnnCsv(const std::string& filename,
  const std::vector<std::string>& labels,
  const std::vector<std::vector<float>>& features) {
  std::ofstream file(filename);
  if (!file.is_open()) return;
  for (size_t i = 0; i < labels.size(); i++) {
    file << labels[i];
    for (float v : features[i]) file << "," << v;
    file << "\n";
  }
}

static void clearConfusionMatrix(ConfusionMatrix& cm) {
  cm.classes.clear();
  cm.class_index.clear();
  cm.matrix.clear();
}

// ============================================================================
// App state
// ============================================================================

static const char* displayModeNames[] = {
  "Original (0)", "Threshold (1)", "Cleaned (2)", "Segmented (3)",
  "Features (4)", "Classification (5)", "CNN Classification (6)"
};

struct AppState {
  std::filesystem::path projectRoot;
  std::string db_filename;
  std::string cnn_db_filename;
  std::string cnn_model_path;

  cv::VideoCapture cap;
  int camNum = 0;
  bool auto_mode = true;
  int manual_thresh = 120;
  int display_mode = 2;
  bool training_mode = false;
  bool eval_mode = false;

  std::vector<std::string> train_labels;
  std::vector<std::vector<double>> train_features;
  std::vector<std::string> cnn_train_labels;
  std::vector<std::vector<float>> cnn_train_features;
  cv::dnn::Net cnn_net;

  ConfusionMatrix conf_matrix_features;
  ConfusionMatrix conf_matrix_cnn;

  char objectNameBuf[128] = "";
  char trueLabelBuf[128] = "";

  cv::Mat frame;
  std::vector<RegionInfo> regions;
  cv::Mat segmented;
  cv::Mat labelMap;

  GLuint texOriginal = 0;
  GLuint texResult = 0;
  int texOriginalW = 0, texOriginalH = 0;
  int texResultW = 0, texResultH = 0;

  float dpiScale = 1.0f;
  float splitRatioLeft = 0.375f;
  float splitRatioMid = 0.375f;
  float splitRatioRight = 0.25f;
  float dbSplitRatio = 0.5f;
  float splitterWidth = 6.0f;
  float videoDisplayW = 0.0f;
  float videoDisplayH = 0.0f;
};

static AppState g_app;

// ============================================================================
// Keyboard shortcuts (when not in text input)
// ============================================================================

static void handleKeyboardShortcuts(GLFWwindow* window) {
  ImGuiIO& io = ImGui::GetIO();
  if (io.WantTextInput) return;

  if (ImGui::IsKeyPressed(ImGuiKey_Q))
    glfwSetWindowShouldClose(window, GLFW_TRUE);

  if (ImGui::IsKeyPressed(ImGuiKey_T))
    g_app.training_mode = !g_app.training_mode;
  if (g_app.training_mode && ImGui::IsKeyPressed(ImGuiKey_E))
    g_app.eval_mode = false;
  if (ImGui::IsKeyPressed(ImGuiKey_E)) {
    g_app.eval_mode = !g_app.eval_mode;
    if (g_app.eval_mode) g_app.training_mode = false;
  }

  if (ImGui::IsKeyPressed(ImGuiKey_A))
    g_app.auto_mode = !g_app.auto_mode;

  if (ImGui::IsKeyPressed(ImGuiKey_Equal) || ImGui::IsKeyPressed(ImGuiKey_KeypadAdd))
    g_app.manual_thresh = std::min(g_app.manual_thresh + 5, 255);
  if (ImGui::IsKeyPressed(ImGuiKey_Minus) || ImGui::IsKeyPressed(ImGuiKey_KeypadSubtract))
    g_app.manual_thresh = std::max(g_app.manual_thresh - 5, 0);

  if (ImGui::IsKeyPressed(ImGuiKey_0)) g_app.display_mode = 0;
  if (ImGui::IsKeyPressed(ImGuiKey_1)) g_app.display_mode = 1;
  if (ImGui::IsKeyPressed(ImGuiKey_2)) g_app.display_mode = 2;
  if (ImGui::IsKeyPressed(ImGuiKey_3)) g_app.display_mode = 3;
  if (ImGui::IsKeyPressed(ImGuiKey_4)) g_app.display_mode = 4;
  if (ImGui::IsKeyPressed(ImGuiKey_5)) g_app.display_mode = 5;
  if (ImGui::IsKeyPressed(ImGuiKey_6)) g_app.display_mode = 6;

  if (ImGui::IsKeyPressed(ImGuiKey_N)) {
    if (g_app.training_mode && !g_app.regions.empty()) {
      std::string name(g_app.objectNameBuf);
      if (!name.empty()) {
        saveTrainingExample(g_app.db_filename, name, g_app.regions[0].featureVector);
        loadTrainingData(g_app.db_filename, g_app.train_labels, g_app.train_features);
      }
    }
  }
  if (ImGui::IsKeyPressed(ImGuiKey_C)) {
    if (g_app.training_mode && !g_app.regions.empty() && !g_app.cnn_net.empty() && !g_app.regions[0].embeddingVector.empty()) {
      std::string name(g_app.objectNameBuf);
      if (!name.empty()) {
        saveTrainingExample(g_app.cnn_db_filename, name, g_app.regions[0].embeddingVector);
        loadTrainingData(g_app.cnn_db_filename, g_app.cnn_train_labels, g_app.cnn_train_features);
      }
    }
  }
  if (ImGui::IsKeyPressed(ImGuiKey_R)) {
    if (g_app.eval_mode && !g_app.regions.empty()) {
      double confF;
      std::string predF = classifyObject(g_app.regions[0].featureVector, g_app.train_labels, g_app.train_features, confF);
      float confC;
      std::string predC = "unknown";
      if (!g_app.regions[0].embeddingVector.empty() && !g_app.cnn_train_labels.empty())
        predC = classifyObjectCNN(g_app.regions[0].embeddingVector, g_app.cnn_train_labels, g_app.cnn_train_features, confC);
      std::string trueLabel(g_app.trueLabelBuf);
      if (!trueLabel.empty()) {
        addResultToMatrix(g_app.conf_matrix_features, trueLabel, predF);
        addResultToMatrix(g_app.conf_matrix_cnn, trueLabel, predC);
      }
    }
  }
  if (ImGui::IsKeyPressed(ImGuiKey_P)) {
    saveConfusionMatrix(g_app.conf_matrix_features, (g_app.projectRoot / "data" / "confusion_matrix_features.csv").string());
    saveConfusionMatrix(g_app.conf_matrix_cnn, (g_app.projectRoot / "data" / "confusion_matrix_cnn.csv").string());
  }
  if (ImGui::IsKeyPressed(ImGuiKey_S)) {
    if (!g_app.frame.empty()) {
      std::string ts = std::to_string(getTime());
      std::string base = (g_app.projectRoot / "data").string() + "/" + ts;
      cv::Mat thresh = g_app.auto_mode ? thresholdImage(g_app.frame) : thresholdImage(g_app.frame, g_app.manual_thresh);
      cv::Mat cleaned = cleanupBinary(thresh);
      cv::imwrite(base + "_original.jpg", g_app.frame);
      cv::imwrite(base + "_threshold.jpg", thresh);
      cv::imwrite(base + "_cleaned.jpg", cleaned);
      cv::imwrite(base + "_segmented.jpg", g_app.segmented);
      cv::Mat featImg = colorizeRegions(g_app.labelMap, g_app.regions);
      drawFeatures(featImg, g_app.regions);
      cv::imwrite(base + "_features.jpg", featImg);
      cv::Mat classImg = colorizeRegions(g_app.labelMap, g_app.regions);
      classifyAndLabel(classImg, g_app.regions, g_app.train_labels, g_app.train_features);
      cv::imwrite(base + "_classified.jpg", classImg);
    }
  }
}

// ============================================================================
// Confusion matrix heatmap (ImGui table)
// ============================================================================

static void renderConfusionMatrixTable(ConfusionMatrix& cm, const char* id, bool* pClearClicked) {
  if (cm.classes.empty()) {
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No evaluation data yet. Record results with [R].");
    if (pClearClicked) *pClearClicked = false;
    return;
  }

  int n = (int)cm.classes.size();
  int maxVal = 0;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      if (cm.matrix[i][j] > maxVal) maxVal = cm.matrix[i][j];

  if (ImGui::BeginTable(id, n + 2, ImGuiTableFlags_Borders, ImVec2(-1, 120))) {
    ImGui::TableSetupScrollFreeze(1, 1);
    ImGui::TableSetupColumn("True \\ Pred", ImGuiTableColumnFlags_WidthFixed, 80.0f);
    for (int j = 0; j < n; j++)
      ImGui::TableSetupColumn(cm.classes[j].c_str(), ImGuiTableColumnFlags_WidthFixed, 50.0f);
    ImGui::TableSetupColumn("Acc %", ImGuiTableColumnFlags_WidthFixed, 50.0f);
    ImGui::TableHeadersRow();

    for (int i = 0; i < n; i++) {
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::Text("%s", cm.classes[i].c_str());

      int rowTotal = 0;
      for (int j = 0; j < n; j++) rowTotal += cm.matrix[i][j];

      for (int j = 0; j < n; j++) {
        ImGui::TableSetColumnIndex(j + 1);
        int v = cm.matrix[i][j];
        float t = (maxVal > 0) ? (float)v / maxVal : 0.0f;
        bool diagonal = (i == j);
        ImVec4 bg = diagonal
          ? ImVec4(0.0f, 0.5f + 0.5f * t, 0.0f, 0.6f)
          : ImVec4(0.5f + 0.5f * t, 0.0f, 0.0f, 0.6f);
        ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, ImGui::ColorConvertFloat4ToU32(bg));
        ImGui::Text("%d", v);
      }
      ImGui::TableSetColumnIndex(n + 1);
      double acc = (rowTotal > 0) ? (100.0 * cm.matrix[i][i]) / rowTotal : 0.0;
      ImGui::Text("%.0f%%", acc);
    }
    ImGui::EndTable();
  }

  if (pClearClicked && ImGui::Button("Clear"))
    *pClearClicked = true;
}

// ============================================================================
// Left panel: video + evaluation
// ============================================================================

static void renderLeftPanel(float totalHeight) {
  if (g_app.texOriginal != 0) {
    float w = g_app.videoDisplayW > 0 ? g_app.videoDisplayW : ImGui::GetContentRegionAvail().x;
    float h = g_app.videoDisplayH > 0 ? g_app.videoDisplayH : (w / ((float)g_app.texOriginalW / g_app.texOriginalH));
    ImGui::Text("Original");
    ImGui::Separator();
    ImGui::Image((ImTextureID)(intptr_t)g_app.texOriginal, ImVec2(w, h));
  }

  ImGui::Separator();
  ImGui::Text("Save Images [S]");
  if (ImGui::Button("Save Images [S]")) {
    if (!g_app.frame.empty()) {
      std::string ts = std::to_string(getTime());
      std::string base = (g_app.projectRoot / "data").string() + "/" + ts;
      cv::imwrite(base + "_original.jpg", g_app.frame);
      cv::Mat thresh = g_app.auto_mode ? thresholdImage(g_app.frame) : thresholdImage(g_app.frame, g_app.manual_thresh);
      cv::Mat cleaned = cleanupBinary(thresh);
      cv::imwrite(base + "_threshold.jpg", thresh);
      cv::imwrite(base + "_cleaned.jpg", cleaned);
      cv::imwrite(base + "_segmented.jpg", g_app.segmented);
      cv::Mat featImg = colorizeRegions(g_app.labelMap, g_app.regions);
      drawFeatures(featImg, g_app.regions);
      cv::imwrite(base + "_features.jpg", featImg);
      cv::Mat classImg = colorizeRegions(g_app.labelMap, g_app.regions);
      classifyAndLabel(classImg, g_app.regions, g_app.train_labels, g_app.train_features);
      cv::imwrite(base + "_classified.jpg", classImg);
    }
  }

  ImGui::Separator();
  ImGui::Text("Evaluation");
  if (ImGui::Checkbox("Eval Mode [E]", &g_app.eval_mode)) {
    if (g_app.eval_mode) g_app.training_mode = false;
  }
  if (ImGui::Button("Record Result [R]")) {
    if (g_app.eval_mode && !g_app.regions.empty()) {
      double confF;
      std::string predF = classifyObject(g_app.regions[0].featureVector, g_app.train_labels, g_app.train_features, confF);
      float confC;
      std::string predC = "unknown";
      if (!g_app.regions[0].embeddingVector.empty() && !g_app.cnn_train_labels.empty())
        predC = classifyObjectCNN(g_app.regions[0].embeddingVector, g_app.cnn_train_labels, g_app.cnn_train_features, confC);
      std::string trueLabel(g_app.trueLabelBuf);
      if (!trueLabel.empty()) {
        addResultToMatrix(g_app.conf_matrix_features, trueLabel, predF);
        addResultToMatrix(g_app.conf_matrix_cnn, trueLabel, predC);
      }
    }
  }
  ImGui::SameLine();
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x);
  ImGui::InputTextWithHint("##truelabel", "True label", g_app.trueLabelBuf, sizeof(g_app.trueLabelBuf));

  if (ImGui::Button("Save Matrix to CSV [P]")) {
    saveConfusionMatrix(g_app.conf_matrix_features, (g_app.projectRoot / "data" / "confusion_matrix_features.csv").string());
    saveConfusionMatrix(g_app.conf_matrix_cnn, (g_app.projectRoot / "data" / "confusion_matrix_cnn.csv").string());
  }

  ImGui::Text("Features");
  bool clearF = false;
  renderConfusionMatrixTable(g_app.conf_matrix_features, "cm_features", &clearF);
  if (clearF) clearConfusionMatrix(g_app.conf_matrix_features);

  ImGui::Text("CNN");
  bool clearC = false;
  renderConfusionMatrixTable(g_app.conf_matrix_cnn, "cm_cnn", &clearC);
  if (clearC) clearConfusionMatrix(g_app.conf_matrix_cnn);
}

// ============================================================================
// Mid panel: result video + controls
// ============================================================================

static void renderMidPanel(float totalHeight) {
  (void)totalHeight;
  ImGui::Text("Result");
  ImGui::Separator();
  if (g_app.texResult != 0) {
    float w = g_app.videoDisplayW > 0 ? g_app.videoDisplayW : ImGui::GetContentRegionAvail().x;
    float h = g_app.videoDisplayH > 0 ? g_app.videoDisplayH : (w / ((float)g_app.texResultW / g_app.texResultH));
    ImGui::Image((ImTextureID)(intptr_t)g_app.texResult, ImVec2(w, h));
  }
  ImGui::Separator();

  ImGui::Text("Display Mode");
  ImGui::SetNextItemWidth(-1);
  ImGui::Combo("##display", &g_app.display_mode, displayModeNames, 7);

  ImGui::Separator();
  ImGui::Text("Threshold");
  if (ImGui::Button(g_app.auto_mode ? "Auto [A]" : "Manual [A]"))
    g_app.auto_mode = !g_app.auto_mode;
  if (!g_app.auto_mode) {
    ImGui::SliderInt("##thresh", &g_app.manual_thresh, 0, 255, "%d");
  }

  ImGui::Separator();
  ImGui::Text("Training");
  if (ImGui::Checkbox("Training Mode [T]", &g_app.training_mode)) {
    if (g_app.training_mode) g_app.eval_mode = false;
  }
  ImGui::SetNextItemWidth(-1);
  ImGui::InputTextWithHint("##objname", "Object name", g_app.objectNameBuf, sizeof(g_app.objectNameBuf));

  if (ImGui::Button("Save Features [N]")) {
    if (g_app.training_mode && !g_app.regions.empty()) {
      std::string name(g_app.objectNameBuf);
      if (!name.empty()) {
        saveTrainingExample(g_app.db_filename, name, g_app.regions[0].featureVector);
        loadTrainingData(g_app.db_filename, g_app.train_labels, g_app.train_features);
      }
    }
  }
  if (ImGui::Button("Save CNN [C]")) {
    if (g_app.training_mode && !g_app.regions.empty() && !g_app.cnn_net.empty() && !g_app.regions[0].embeddingVector.empty()) {
      std::string name(g_app.objectNameBuf);
      if (!name.empty()) {
        saveTrainingExample(g_app.cnn_db_filename, name, g_app.regions[0].embeddingVector);
        loadTrainingData(g_app.cnn_db_filename, g_app.cnn_train_labels, g_app.cnn_train_features);
      }
    }
  }
}

// ============================================================================
// Right panel: DB manager (vertical split)
// ============================================================================

static void renderFeaturesDbList(const char* title, const std::string& filename,
  std::vector<std::string>& labels,
  std::vector<std::vector<double>>& features,
  float height) {
  ImGui::Text("%s", title);
  if (ImGui::Button("Reload")) loadTrainingData(filename, labels, features);
  ImGui::SameLine();
  ImGui::Text("(%zu)", labels.size());
  ImGui::BeginChild(title, ImVec2(-1, height), true, ImGuiWindowFlags_NoScrollbar);
  const float delBtnW = 48.0f;
  const float rightMargin = 6.0f;
  float contentW = ImGui::GetWindowContentRegionMax().x - ImGui::GetWindowContentRegionMin().x;
  float labelMaxW = contentW - rightMargin - delBtnW - ImGui::GetStyle().ItemSpacing.x;
  if (labelMaxW < 30.0f) labelMaxW = 30.0f;
  for (size_t i = 0; i < labels.size(); i++) {
    ImGui::PushID((int)i);
    float lineStartX = ImGui::GetCursorPos().x;
    ImGui::PushTextWrapPos(lineStartX + labelMaxW);
    ImGui::TextUnformatted(labels[i].c_str());
    ImGui::PopTextWrapPos();
    ImGui::SameLine((contentW - rightMargin - delBtnW) - lineStartX);
    if (ImGui::Button("Del", ImVec2(delBtnW, 0.0f))) {
      labels.erase(labels.begin() + i);
      features.erase(features.begin() + i);
      rewriteFeaturesCsv(filename, labels, features);
      i--;
    }
    ImGui::PopID();
  }
  ImGui::EndChild();
}

static void renderCnnDbList(const char* title, const std::string& filename, float height) {
  ImGui::Text("%s", title);
  if (ImGui::Button("Reload##cnn")) {
    loadTrainingData(filename, g_app.cnn_train_labels, g_app.cnn_train_features);
  }
  ImGui::SameLine();
  ImGui::Text("(%zu)", g_app.cnn_train_labels.size());
  ImGui::BeginChild(title, ImVec2(-1, height), true, ImGuiWindowFlags_NoScrollbar);
  const float delBtnW = 48.0f;
  const float rightMargin = 6.0f;
  float contentW = ImGui::GetWindowContentRegionMax().x - ImGui::GetWindowContentRegionMin().x;
  float labelMaxW = contentW - rightMargin - delBtnW - ImGui::GetStyle().ItemSpacing.x;
  if (labelMaxW < 30.0f) labelMaxW = 30.0f;
  for (size_t i = 0; i < g_app.cnn_train_labels.size(); i++) {
    ImGui::PushID((int)(i + 10000));
    float lineStartX = ImGui::GetCursorPos().x;
    ImGui::PushTextWrapPos(lineStartX + labelMaxW);
    ImGui::TextUnformatted(g_app.cnn_train_labels[i].c_str());
    ImGui::PopTextWrapPos();
    ImGui::SameLine((contentW - rightMargin - delBtnW) - lineStartX);
    if (ImGui::Button("Del", ImVec2(delBtnW, 0.0f))) {
      g_app.cnn_train_labels.erase(g_app.cnn_train_labels.begin() + i);
      g_app.cnn_train_features.erase(g_app.cnn_train_features.begin() + i);
      rewriteCnnCsv(filename, g_app.cnn_train_labels, g_app.cnn_train_features);
      i--;
    }
    ImGui::PopID();
  }
  ImGui::EndChild();
}

static void renderRightPanel(float totalHeight) {
  float sw = 6.0f * g_app.dpiScale;
  float headerH = ImGui::GetTextLineHeightWithSpacing() + ImGui::GetFrameHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y;
  float availForLists = totalHeight - 2.0f * headerH - sw - 12.0f;
  float topH = availForLists * g_app.dbSplitRatio;
  float botH = availForLists * (1.0f - g_app.dbSplitRatio);
  if (topH < 60.0f) topH = 60.0f;
  if (botH < 60.0f) botH = 60.0f;

  renderFeaturesDbList("Features DB", g_app.db_filename, g_app.train_labels, g_app.train_features, topH);

  ImGui::InvisibleButton("##dbSplitter", ImVec2(-1, sw));
  if (ImGui::IsItemActive())
    g_app.dbSplitRatio += ImGui::GetIO().MouseDelta.y / totalHeight;
  g_app.dbSplitRatio = std::clamp(g_app.dbSplitRatio, 0.2f, 0.8f);
  if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
    ImGui::GetWindowDrawList()->AddRectFilled(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32(100, 150, 255, 120));
  }

  renderCnnDbList("CNN DB", g_app.cnn_db_filename, botH);
}

// ============================================================================
// Main UI: 3 panels + 2 splitters
// ============================================================================

static void renderUI(GLFWwindow* window, float totalWidth, float totalHeight) {
  handleKeyboardShortcuts(window);

  float sw = g_app.splitterWidth * g_app.dpiScale;
  float leftW = totalWidth * g_app.splitRatioLeft - sw * 0.5f;
  float midW = totalWidth * g_app.splitRatioMid - sw * 0.5f;
  float rightW = totalWidth * g_app.splitRatioRight - sw * 0.5f;

  float aspect = (g_app.texOriginalW > 0 && g_app.texOriginalH > 0)
    ? (float)g_app.texOriginalW / g_app.texOriginalH : (4.0f / 3.0f);
  float panelW = std::min(leftW, midW) - 16.0f;
  g_app.videoDisplayW = panelW > 0 ? panelW : 0.0f;
  g_app.videoDisplayH = (g_app.videoDisplayW > 0) ? (g_app.videoDisplayW / aspect) : 0.0f;
  float maxVideoH = totalHeight * 0.42f;
  if (g_app.videoDisplayH > maxVideoH && g_app.videoDisplayH > 0) {
    g_app.videoDisplayH = maxVideoH;
    g_app.videoDisplayW = g_app.videoDisplayH * aspect;
  }

  ImGui::BeginChild("LeftPanel", ImVec2(leftW, totalHeight), true, ImGuiWindowFlags_NoScrollbar);
  renderLeftPanel(totalHeight);
  ImGui::EndChild();

  ImGui::SameLine();
  ImGui::InvisibleButton("##split1", ImVec2(sw, totalHeight));
  if (ImGui::IsItemActive()) {
    float dx = ImGui::GetIO().MouseDelta.x / totalWidth;
    g_app.splitRatioLeft += dx;
    g_app.splitRatioMid -= dx;
    g_app.splitRatioLeft = std::clamp(g_app.splitRatioLeft, 0.25f, 0.55f);
    g_app.splitRatioMid = std::clamp(g_app.splitRatioMid, 0.25f, 0.55f);
    g_app.splitRatioRight = 1.0f - g_app.splitRatioLeft - g_app.splitRatioMid;
  }
  if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    ImGui::GetWindowDrawList()->AddRectFilled(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32(100, 150, 255, 120));
  }
  ImGui::SameLine();

  ImGui::BeginChild("MidPanel", ImVec2(midW, totalHeight), true, ImGuiWindowFlags_NoScrollbar);
  renderMidPanel(totalHeight);
  ImGui::EndChild();

  ImGui::SameLine();
  ImGui::InvisibleButton("##split2", ImVec2(sw, totalHeight));
  if (ImGui::IsItemActive()) {
    float dx = ImGui::GetIO().MouseDelta.x / totalWidth;
    g_app.splitRatioMid += dx;
    g_app.splitRatioRight -= dx;
    g_app.splitRatioMid = std::clamp(g_app.splitRatioMid, 0.25f, 0.55f);
    g_app.splitRatioRight = std::clamp(g_app.splitRatioRight, 0.15f, 0.4f);
    g_app.splitRatioLeft = 1.0f - g_app.splitRatioMid - g_app.splitRatioRight;
  }
  if (ImGui::IsItemHovered() || ImGui::IsItemActive()) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    ImGui::GetWindowDrawList()->AddRectFilled(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32(100, 150, 255, 120));
  }
  ImGui::SameLine();

  ImGui::BeginChild("RightPanel", ImVec2(rightW, totalHeight), true, ImGuiWindowFlags_NoScrollbar);
  renderRightPanel(totalHeight);
  ImGui::EndChild();
}

// ============================================================================
// Process one frame: capture, pipeline, build result image
// ============================================================================

static void processFrame() {
  g_app.cap >> g_app.frame;
  if (g_app.frame.empty()) return;

  cv::Mat thresh = g_app.auto_mode ? thresholdImage(g_app.frame) : thresholdImage(g_app.frame, g_app.manual_thresh);
  cv::Mat cleaned = cleanupBinary(thresh);
  g_app.segmented = segmentRegions(cleaned, g_app.regions, g_app.labelMap);

  for (auto& r : g_app.regions)
    computeRegionFeatures(g_app.labelMap, r);

  if (!g_app.cnn_net.empty()) {
    for (auto& r : g_app.regions) {
      cv::Mat embImg;
      prepEmbeddingImage(g_app.frame, embImg, (int)r.centroid.x, (int)r.centroid.y, r.theta, r.uMin, r.uMax, r.vMin, r.vMax, 0);
      if (!embImg.empty() && embImg.cols > 0 && embImg.rows > 0) {
        cv::Mat embedding;
        getEmbedding(embImg, embedding, g_app.cnn_net, 0);
        r.embeddingVector.clear();
        for (int i = 0; i < embedding.cols; i++)
          r.embeddingVector.push_back(embedding.at<float>(0, i));
      }
    }
  }

  cv::Mat show;
  switch (g_app.display_mode) {
    case 0:
      cv::cvtColor(g_app.frame, show, cv::COLOR_BGR2GRAY);
      cv::cvtColor(show, show, cv::COLOR_GRAY2BGR);
      break;
    case 1:
      cv::cvtColor(thresh, show, cv::COLOR_GRAY2BGR);
      break;
    case 2:
      cv::cvtColor(cleaned, show, cv::COLOR_GRAY2BGR);
      break;
    case 3:
      show = g_app.segmented.clone();
      break;
    case 4:
      show = colorizeRegions(g_app.labelMap, g_app.regions);
      drawFeatures(show, g_app.regions);
      break;
    case 5:
      show = colorizeRegions(g_app.labelMap, g_app.regions);
      drawFeatures(show, g_app.regions);
      classifyAndLabel(show, g_app.regions, g_app.train_labels, g_app.train_features);
      break;
    case 6:
      show = colorizeRegions(g_app.labelMap, g_app.regions);
      drawFeatures(show, g_app.regions);
      classifyAndLabelCNN(show, g_app.regions, g_app.cnn_train_labels, g_app.cnn_train_features);
      break;
    default:
      cv::cvtColor(cleaned, show, cv::COLOR_GRAY2BGR);
      break;
  }

  freeTexture(g_app.texOriginal);
  freeTexture(g_app.texResult);
  g_app.texOriginal = matToTexture(g_app.frame, g_app.texOriginalW, g_app.texOriginalH);
  g_app.texResult = matToTexture(show, g_app.texResultW, g_app.texResultH);
}

// ============================================================================
// Main
// ============================================================================

static void errorCallback(int error, const char* description) {
  std::println(stderr, "GLFW Error {}: {}", error, description);
}

int main(int argc, char* argv[]) {
  glfwSetErrorCallback(errorCallback);
  if (!glfwInit()) { std::println(stderr, "Failed to initialize GLFW"); return 1; }

  g_app.projectRoot = std::filesystem::path(argv[0]).parent_path().parent_path();
  g_app.db_filename = (g_app.projectRoot / "data" / "objects_db.csv").string();
  g_app.cnn_db_filename = (g_app.projectRoot / "data" / "objects_cnn_db.csv").string();
  g_app.cnn_model_path = (g_app.projectRoot / "data" / "CNN" / "resnet18-v2-7.onnx").string();

  g_app.camNum = (argc > 1) ? atoi(argv[1]) : 0;
  g_app.cap.open(g_app.camNum);
  if (!g_app.cap.isOpened()) {
    std::println(stderr, "Can't open camera");
    glfwTerminate();
    return -1;
  }
  g_app.cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  g_app.cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

  loadTrainingData(g_app.db_filename, g_app.train_labels, g_app.train_features);
  loadTrainingData(g_app.cnn_db_filename, g_app.cnn_train_labels, g_app.cnn_train_features);

  try {
    g_app.cnn_net = cv::dnn::readNetFromONNX(g_app.cnn_model_path);
  } catch (const cv::Exception&) {
    g_app.cnn_net = cv::dnn::Net();
  }

  float xscale = 1.0f, yscale = 1.0f;
  if (GLFWmonitor* mon = glfwGetPrimaryMonitor())
    glfwGetMonitorContentScale(mon, &xscale, &yscale);
  g_app.dpiScale = xscale;

  int winW = (int)(1400 * xscale);
  int winH = (int)(900 * xscale);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

  GLFWwindow* window = glfwCreateWindow(winW, winH, "OR2D - Real-time 2-D Object Recognition", nullptr, nullptr);
  if (!window) { std::println(stderr, "Failed to create window"); glfwTerminate(); return 1; }

  if (GLFWmonitor* mon = glfwGetPrimaryMonitor()) {
    const GLFWvidmode* mode = glfwGetVideoMode(mon);
    glfwSetWindowPos(window, (mode->width - winW) / 2, (mode->height - winH) / 2);
  }
  glfwShowWindow(window);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

#ifdef _WIN32
  {
    HWND hwnd = glfwGetWin32Window(window);
    BOOL useDarkMode = TRUE;
    DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, &useDarkMode, sizeof(useDarkMode));
    COLORREF captionColor = RGB(25, 25, 30);
    DwmSetWindowAttribute(hwnd, 35, &captionColor, sizeof(captionColor));
  }
#endif

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.Fonts->AddFontDefault();
  io.FontGlobalScale = xscale;

  ImGui::StyleColorsDark();
  ImGuiStyle& style = ImGui::GetStyle();
  style.ScaleAllSizes(xscale);
  style.WindowRounding = 0.0f;
  style.FrameRounding = 4.0f;
  style.GrabRounding = 4.0f;
  style.Colors[ImGuiCol_Header] = ImVec4(0.26f, 0.59f, 0.98f, 0.31f);
  style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
  style.Colors[ImGuiCol_Button] = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
  style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
  style.Colors[ImGuiCol_FrameBg] = ImVec4(0.16f, 0.29f, 0.48f, 0.54f);

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL2_Init();

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    processFrame();

    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(io.DisplaySize);
    ImGui::Begin("OR2D", nullptr,
      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
      ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollbar);

    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    float totalWidth = (float)display_w;
    float totalHeight = (float)display_h;
    renderUI(window, totalWidth, totalHeight);

    ImGui::End();
    ImGui::Render();

    glViewport(0, 0, display_w, display_h);
    glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  freeTexture(g_app.texOriginal);
  freeTexture(g_app.texResult);
  g_app.cap.release();
  ImGui_ImplOpenGL2_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
