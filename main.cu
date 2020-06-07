#include <iostream>
#include <ostream>
#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <vector>
#include <fstream>

// TODO: impletemnt PFM to GPU
__global__ void cuda_hello() {
  printf("Hello Cuda!\n");
}

const unsigned int x_length = 53;
const unsigned int y_length = 53;

bool save(std::vector<std::vector<double>> phase, unsigned int n) {
  try {
    std::ofstream file;
    std::ostringstream filename;
    filename << "datas/step_" << std::setfill('0') << std::right << std::setw(2) << n << ".dat";
    file.open(filename.str(), std::ios_base::app);

    file << "#x #y #phase" << std::endl;
    // remove boundaries
    for (unsigned int y_i = 1; y_i < y_length - 1; y_i++) {
      for (unsigned int x_i = 1; x_i < x_length - 1; x_i++) {
        file << y_i << ' ' <<  x_i << ' ' << phase.at(y_i).at(x_i) << std::endl;
      }
      file << std::endl;
    }
    file.close();
  } catch(char *str) {
    std::cout << str << std::endl;
    return false;
  }
  return true;
}

int main() {
  std::vector<std::vector<double>> phase(y_length, std::vector<double>(x_length, 0));
  std::vector<std::vector<double>> phase_tmp(y_length, std::vector<double>(x_length, 0));

  // 格子サイズ
  float dx = 5e-7;
  float dy = 5e-7;
  // 界面エネルギー
  float gamma = 1.;
  // 界面幅
  float delta = 4. * dx;
  // 界面モビリティ
  float M = 4e-14;
  // 界面領域
  float lambda = .1;
  // 勾配計数
  float b = 2. * std::atanh(1.-2.*lambda);
  float a = std::sqrt(3. * delta * gamma / b);
  // エネルギー障壁
  float w = 6. * gamma * b / delta;
  // フェーズフィールドモビリティ
  float M_phi = M * std::sqrt(2. * w) / (6. * a);
  // パラメータ
  float beta = 0.5;
  // 時間ステップ
  float dt = dx * dx / (5. * M_phi * a * a);
  printf("Time Step: %.3e[s]\n", dt);
  // 計算ステップ
  unsigned int step = 100;
  // 固相初期半径
  float r0 = .5 * (x_length - 1) * dx;


  // 初期条件セット
  // メインループ
  for (unsigned int y_i = 1; y_i < y_length - 1; y_i++) {
    for (unsigned int x_i = 1; x_i < x_length - 1; x_i++) {
      float y = (y_i - 1) * dy;
      float x = (x_i - 1) * dx;

      float r = std::sqrt(x*x + y*y) - r0;
      phase.at(y_i).at(x_i) = .5 * (1. - std::tanh(std::sqrt(2. * w) / (2. * a) * r));
    }
  }
  // 境界条件
  for (unsigned int x_i = 0; x_i < x_length; x_i++) {
    phase.at(0).at(x_i) = phase.at(1).at(x_i);
    phase.at(y_length - 1).at(x_i) = phase.at(y_length - 2).at(x_i);
  }
  for (unsigned int y_i = 0; y_i < y_length; y_i++) {
    phase.at(y_i).at(0) = phase.at(1).at(y_i);
    phase.at(y_i).at(x_length - 1) = phase.at(y_i).at(x_length - 2);
  }

  // メインループ
  for (unsigned int n = 0; n < step; n++) {
    printf("step: %d\n", n);
    for (unsigned int y_i = 1; y_i < y_length - 1; y_i++) {
      for (unsigned int x_i = 1; x_i < x_length - 1; x_i++) {
        double rpx = (phase.at(y_i).at(x_i + 1) - 2. * phase.at(y_i).at(x_i) + phase.at(y_i).at(x_i - 1)) / (dx * dx);
        double rpy = (phase.at(y_i + 1).at(x_i) - 2. * phase.at(y_i).at(x_i) + phase.at(y_i - 1).at(x_i)) / (dy * dy);

        double dpi1 = a * a * (rpx + rpy);
        double dpi2 = 4. * w * phase.at(y_i).at(x_i) * (1 - phase.at(y_i).at(x_i)) * (phase.at(y_i).at(x_i) - .5 + beta);
        double dpi = dpi1 + dpi2;
        phase_tmp.at(y_i).at(x_i) = phase.at(y_i).at(x_i) + M_phi * dpi * dt;
      }
    }

    save(phase, n);

    // Swap
    for (unsigned int y_i = 1; y_i < y_length - 1; y_i++) {
      for (unsigned int x_i = 1; x_i < x_length - 1; x_i++) {
        phase.at(y_i).at(x_i) = phase_tmp.at(y_i).at(x_i);
      }
    }

    // 境界条件
    for (unsigned int x_i = 0; x_i < x_length; x_i++) {
      phase.at(0).at(x_i) = phase.at(1).at(x_i);
      phase.at(y_length - 1).at(x_i) = phase.at(y_length - 2).at(x_i);
    }
    for (unsigned int y_i = 0; y_i < y_length; y_i++) {
      phase.at(y_i).at(0) = phase.at(1).at(y_i);
      phase.at(y_i).at(x_length - 1) = phase.at(y_i).at(x_length - 2);
    }
  }

  return 0;
}
