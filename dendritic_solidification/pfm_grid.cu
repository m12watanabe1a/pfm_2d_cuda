#include <cstdlib>
#include <iostream>
#include <ostream>
#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>

const unsigned int field_size = 750;
const unsigned int step = 100;

__device__ unsigned int d_field_size;
// TODO: Decrare variables for device

__global__ void calc_step(double *d_phase, double *d_T, double *d_phase_tmp, double *d_T_tmp) {
  int x_i = blockIdx.x * blockDim.x + threadIdx.x;
  int y_i = blockIdx.y * blockDim.y + threadIdx.y;
  if (x_i <= 0 || x_i >= d_field_size - 1 || y_i <= 0 || y_i >= d_field_size - 1) return;

  int i = y_i * d_field_size + x_i;
  float ddx = d_dx * d_dx;
  double rpx = (d_phase[i + 1] - 2.* d_phase[i] + d_phase[i - 1]) / ddx;
  double rpy = (d_phase[i + d_field_size] - 2. * d_phase[i] + d_phase[i - d_field_size]) / ddx;

  double dpi1 = d_a * d_a * (rpx + rpy);
  double dpi2 = 4. * d_w * d_phase[i] * (1 - d_phase[i]) * (d_phase[i] - .5 + d_beta);
  double dpi = dpi1 + dpi2;
  d_phase_tmp[i] = d_phase[i] + d_tau * dpi;
}

// 零ノイマン境界条件
void set_bc(double *phase, double *T) {
  for (unsigned int x_i = 0; x_i < field_size; x_i++) {
    phase[x_i] = phase[field_size + x_i];
    phase[(field_size -1) * field_size + x_i] = phase[(field_size - 2) * field_size + x_i];

    T[x_i] = T[field_size + x_i];
    T[(field_size -1) * field_size + x_i] = T[(field_size - 2) * field_size + x_i];
  }
  for (unsigned int y_i = 0; y_i < field_size; y_i++) {
    phase[y_i * field_size] = phase[y_i * field_size + 1];
    phase[(y_i + 1)*field_size - 1] = phase[(y_i + 1)*field_size - 2];

    T[y_i * field_size] = T[y_i * field_size + 1];
    T[(y_i + 1)*field_size - 1] = T[(y_i + 1)*field_size - 2];
  }
  return;
}

bool save(double *phase, double *T, unsigned int n) {
  try {
    std::ofstream file;
    std::ostringstream filename;
    filename << "datas/step_" << std::setfill('0') << std::right << std::setw(2) << n << ".dat";
    file.open(filename.str(), std::ios_base::app);

    file << "#x #y #phase #temperature" << std::endl;
    // remove boundaries
    for (unsigned int y_i = 1; y_i < field_size - 1; y_i++) {
      for (unsigned int x_i = 1; x_i < field_size - 1; x_i++) {
        file << y_i << ' ' <<  x_i << ' ' << phase[y_i * field_size + x_i] << ' ' << T[y_i * field_size + x_i] << std::endl;
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
  unsigned int N = field_size * field_size;
  double *phase, *T; // phase field for host
  phase = (double *)malloc(N * sizeof(double));
  T = (double *)malloc(N * sizeof(double));
  double *d_phase, *d_phase_tmp, *d_T, *d_T_tmp; // phase field for device

  float dx = 20e-9;
  // 熱伝導率
  float K = 84.01;
  // 比熱
  float c = 5.42e+6;
  // 潜熱
  float L = 2.350e+9;
  // 融点
  float Tm = 1728.;
  // 界面キネティック係数
  float mu = 2.;
  // ゆらぎ
  float kai = .1;
  // 優先成長方向
  float theta_0 = 0.;
  // 界面エネルギー
  float gamma = 0.37;
  // 異方性モード
  float k = 4.;
  // 界面幅
  float delta = 4. * dx;
  // 界面領域
  float lambda = .1;
  // 勾配計数
  float b = 2. * std::atanh(1.-2.*lambda);
  float a = std::sqrt(3. * delta * gamma / b);
  // エネルギー障壁
  float W = 6. * gamma * b / delta;
  // フェーズフィールドモビリティ
  float M_phi = b * Tm * mu / 3. / delta / L;
  // 熱拡散係数
  float kappa = K / c;
  // 時間ステップ
  float dt1 = dx * dx / 5. / M_phi / a / a;
  float dt2 = dx * dx / 5. / kappa;
  float dt = std::min(dt1, dt2);
  printf("Time Step: %.3e[s]\n", dt);
  // 固相初期半径
  float r0 = 2. * dx;
  // 無次元過冷却度
  float Delta = .5;
  // 無次元過冷却温度
  float T_0 = Tm - Delta * L / c;

  // TODO: set necessary variables for simulation
  cudaMemcpyToSymbol(d_field_size, &field_size, sizeof(unsigned int));

  // 初期条件セット
  for (unsigned int y_i = 0; y_i < field_size; y_i++) {
    for (unsigned int x_i = 0; x_i < field_size; x_i++) {
      int i = y_i * field_size + x_i;
      if (x_i <= 0 || x_i >= field_size - 1 || y_i <= 0 || y_i >= field_size - 1) {
        phase[i] = 0.0;
        T[i] = T_0;
        continue;
      }
      float y = (y_i - 1) * dx;
      float x = (x_i - 1) * dx;

      float r = std::sqrt(x*x + y*y) - r0;
      phase[i] = .5 * (1. - std::tanh(std::sqrt(2. * W) / (2. * a) * r));
      T[i] = T_0 + phase[i] * (Tm - T_0);
    }
  }

  set_bc(phase, T);

  // allocate memory to GPU
  cudaMalloc((void**)&d_phase, N * sizeof(double));
  cudaMalloc((void**)&d_phase, N * sizeof(double));
  cudaMalloc((void**)&d_T, N * sizeof(double));
  cudaMalloc((void**)&d_T_tmp, N * sizeof(double));

  // calc blocks
  int threadsPerBlock = 32;
  int blocksInGrid = (field_size + threadsPerBlock -1)/threadsPerBlock;
  dim3 blocks(threadsPerBlock, threadsPerBlock);
  dim3 grid(blocksInGrid, blocksInGrid);

  // メインループ
  for (unsigned int n = 0; n < step; n++) {
    printf("step: %d\n", n);

    // copy memory on GPU
    cudaMemcpy(d_phase, phase, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T, T, N * sizeof(double), cudaMemcpyHostToDevice);

    calc_step<<<grid, blocks>>>(d_phase, d_T, d_phase_tmp, d_T);
    cudaDeviceSynchronize();
    save(phase, T, n);

    // Swap
    cudaMemcpy(phase, d_phase_tmp, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(T, d_T_tmp, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Boundary Condition
    set_bc(phase, T);

  }

  free(phase);
  cudaFree(d_phase);
  cudaFree(d_phase_tmp);

  free(T);
  cudaFree(d_T);
  cudaFree(d_T_tmp);

  return 0;
}
