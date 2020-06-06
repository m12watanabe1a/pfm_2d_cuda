#include <stdio.h>
#include <math.h>

using namespace std;

// TODO: impletemnt PFM to GPU
__global__ void cuda_hello() {
  printf("Hello Cuda!\n");
}


int main() {
  const unsigned int x_max = 52;
  const unsigned int y_max = 52;

  // 格子サイズ
  float dx = 5e-6;
  float dy = 5e-6;
  // 界面エネルギー
  float gamma = 1.0;
  // 界面幅
  float delta = 4. * dx;
  // 界面モビリティ
  float M = 4e-14;
  // 界面領域
  float lambda = 0.1;
  // 勾配計数
  float tmp = 2. * std::atanh(1.-2.*lambda);
  float a = std::sqrt(3. * delta * gamma / tmp);
  // エネルギー障壁
  float w = 6. * gamma * tmp / delta;
  // フェーズフィールドモビリティ
  float M_phi = M * std::sqrt(2. * w) / 6. /a;
  // パラメータ
  float beta = 0.5;
  // 時間ステップ
  float dt = dx * dx / (5. * M_phi * a * a);
  printf("Time Step: %.3e[s]\n", dt);
  // 計算ステップ
  unsigned int n = 100;
  // 固相初期半径
  float r0 = .5 * x_max * dx;

  return 0;
}
