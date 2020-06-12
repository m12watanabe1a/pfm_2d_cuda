#include <cstdlib>
#include <iostream>
#include <ostream>
#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <vector>
#include <fstream>

const unsigned int field_size = 53;
const unsigned int step = 100;

__device__ unsigned int d_field_size;
__device__ float d_dx;
__device__ float d_a;
__device__ float d_w;
__device__ float d_beta;
__device__ float d_tau;

__global__ void calc_step(double *d_phase, double *d_phase_tmp) {
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


__global__ void set_bc_device(double *field) {
  int x_i = blockIdx.x * blockDim.x + threadIdx.x;
  if ( x_i >= field_size - 2)  return;
  int i = x_i + 1;
  // top
  field[i] = field[i+field_size];
  // bottom
  field[field_size * (field_size - 1) + i] = field[field_size * (field_size - 2) + i];
  // left
  field[field_size * i] = field[field_size * i + 1];
  // right
  field[field_size * (i + 1) - 1] = field[field_size * (i + 1) - 2];
  return;
}


// 境界条件
double* set_bc(double *phase) {
  for (unsigned int x_i = 0; x_i < field_size; x_i++) {
    phase[x_i] = phase[field_size + x_i];
    phase[(field_size -1) * field_size + x_i] = phase[(field_size - 2) * field_size + x_i];
  }
  for (unsigned int y_i = 0; y_i < field_size; y_i++) {
    phase[y_i * field_size] = phase[y_i * field_size + 1];
    phase[(y_i + 1)*field_size - 1] = phase[(y_i + 1)*field_size - 2];
  }
  return phase;
}

bool save(double *phase, unsigned int n) {
  try {
    std::ofstream file;
    std::ostringstream filename;
    filename << "datas/step_" << std::setfill('0') << std::right << std::setw(2) << n << ".dat";
    file.open(filename.str(), std::ios_base::app);

    file << "#x #y #phase" << std::endl;
    // remove boundaries
    for (unsigned int y_i = 1; y_i < field_size - 1; y_i++) {
      for (unsigned int x_i = 1; x_i < field_size - 1; x_i++) {
        file << y_i << ' ' <<  x_i << ' ' << phase[y_i * field_size + x_i] << std::endl;
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
  double *phase; // phase field for host
  double *d_phase, *d_phase_tmp; // phase field for device

  phase = (double *)malloc(N * sizeof(double));

  const float dx = 5e-7;
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
  // 時間ステップ
  float dt = dx * dx / (5. * M_phi * a * a);
  printf("Time Step: %.3e[s]\n", dt);
  // 固相初期半径
  float r0 = .5 * (field_size - 1) * dx;

  float beta = .5;

  float tau = M_phi * dt;

  cudaMemcpyToSymbol(d_field_size, &field_size, sizeof(unsigned int));
  cudaMemcpyToSymbol(d_dx, &dx, sizeof(float));
  cudaMemcpyToSymbol(d_a, &a, sizeof(float));
  cudaMemcpyToSymbol(d_w, &w, sizeof(float));
  cudaMemcpyToSymbol(d_beta, &beta, sizeof(float));
  cudaMemcpyToSymbol(d_tau, &tau, sizeof(float));

  // 初期条件セット
  for (unsigned int y_i = 0; y_i < field_size; y_i++) {
    for (unsigned int x_i = 0; x_i < field_size; x_i++) {
      int i = y_i * field_size + x_i;
      if (x_i <= 0 || x_i >= field_size - 1 || y_i <= 0 || y_i >= field_size - 1) {
        phase[i] = 0.0;
        continue;
      }
      float y = (y_i - 1) * dx;
      float x = (x_i - 1) * dx;

      float r = std::sqrt(x*x + y*y) - r0;
      phase[i] = .5 * (1. - std::tanh(std::sqrt(2. * w) / (2. * a) * r));
    }
  }

  set_bc(phase);

  // allocate memory to GPU
  cudaMalloc((void**)&d_phase, N * sizeof(double));
  cudaMalloc((void**)&d_phase_tmp, N * sizeof(double));


  int threadsPerBlock = 32;
  int blocksInGrid = (field_size + threadsPerBlock -1)/threadsPerBlock;
  dim3 blocks(threadsPerBlock, threadsPerBlock);
  dim3 grid(blocksInGrid, blocksInGrid);

  cudaMemcpy(d_phase, phase, N * sizeof(double), cudaMemcpyHostToDevice);
  // メインループ
  for (unsigned int n = 0; n < step; n++) {
    printf("step: %d\n", n);

    // copy memory on GPU

    calc_step<<<grid, blocks>>>(d_phase, d_phase_tmp);
    cudaDeviceSynchronize();
    save(phase, n);

    // Swap
    cudaMemcpy(d_phase, d_phase_tmp, N * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(phase, d_phase_tmp, N * sizeof(double), cudaMemcpyDeviceToHost);

    set_bc_device<<<1, field_size>>>(d_phase);
  }

  free(phase);
  cudaFree(d_phase);
  cudaFree(d_phase_tmp);

  return 0;
}
