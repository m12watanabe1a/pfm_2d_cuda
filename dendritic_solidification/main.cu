#include <iostream>
#include <ostream>
#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

const unsigned int field_size = 100;
const unsigned int step = 100000;

__device__ unsigned int d_field_size;
__device__ float d_dx;
__device__ float d_a;
__device__ float d_xi;
__device__ float d_k;
__device__ float d_theta_0;
__device__ float d_a_bar;
__device__ float d_W;
__device__ float d_Tm;
__device__ float d_L;
__device__ float d_chi;
__device__ float d_M_phi;
__device__ float d_dt;
__device__ float d_c;
__device__ float d_kappa;

__device__ float get_a(float theta) {
  return d_a_bar * ( 1 + d_xi * cos(d_k * (theta - d_theta_0)) );
}

__device__ float get_rat(float theta) {
  return d_a_bar * d_xi * d_k * sin(d_k * (theta - d_theta_0));
}

__global__ void setCurand(unsigned long long seed, curandState *state){
  int x_i = blockIdx.x * blockDim.x + threadIdx.x;
  int y_i = blockIdx.y * blockDim.y + threadIdx.y;
  int i = y_i * d_field_size + x_i;
  curand_init(seed, i, 0, &state[i]);
}

__global__ void init_field(float *phase, float *T, float r_0, float T_0) {
  int x_i = blockIdx.x * blockDim.x + threadIdx.x;
  int y_i = blockIdx.y * blockDim.y + threadIdx.y;
  if (x_i <= 0 || x_i >= d_field_size - 1 || y_i <= 0 || y_i >= d_field_size - 1) return;
  int i = y_i * d_field_size + x_i;

  float y = (y_i - 1) * d_dx;
  float x = (x_i - 1) * d_dx;
  float r = sqrt(x*x + y*y) - r_0;
  phase[i] = .5 * (1. - tanh(sqrt(2. * d_W) / (2. * d_a_bar) * r));
  T[i] = T_0 + phase[i] * (d_Tm - T_0);
  return;
}

// 零ノイマン境界条件
__global__ void set_bc(float *field) {
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

__global__ void calc_phase_term_1(float *d_phase_term_1_tmp, float *d_phase_term_1) {
  int x_i = blockIdx.x * blockDim.x + threadIdx.x;
  int y_i = blockIdx.y * blockDim.y + threadIdx.y;
  if (x_i <= 0 || x_i >= d_field_size - 1 || y_i <= 0 || y_i >= d_field_size - 1) return;
  int i = y_i * d_field_size + x_i;

  float rtx = (d_phase_term_1_tmp[i + 1] - d_phase_term_1_tmp[i + 1]) / d_dx;
  float rty = (d_phase_term_1_tmp[i + d_field_size] - d_phase_term_1_tmp[i - d_field_size]) / d_dx;
  d_phase_term_1[i] = rtx + rty;
  return;
}

__global__ void calc_phase_term_1_tmp(float *d_rpx, float *d_rpy, float *d_theta, float *d_phase_term_1_tmp) {
  int x_i = blockIdx.x * blockDim.x + threadIdx.x;
  int y_i = blockIdx.y * blockDim.y + threadIdx.y;
  if (x_i <= 0 || x_i >= d_field_size - 1 || y_i <= 0 || y_i >= d_field_size - 1) return;
  int i = y_i * d_field_size + x_i;

  float nabla_phi = d_rpx[i] + d_rpy[i];
  float a = get_a(d_theta[i]);

  d_phase_term_1_tmp[i] = a * a * nabla_phi;
  return;
}

__global__ void calc_phase_term_2(float *d_phase_term_2_tmp_x, float *d_phase_term_2_tmp_y, float *d_phase_term_2){
  int x_i = blockIdx.x * blockDim.x + threadIdx.x;
  int y_i = blockIdx.y * blockDim.y + threadIdx.y;
  if (x_i <= 0 || x_i >= d_field_size - 1 || y_i <= 0 || y_i >= d_field_size - 1) return;
  int i = y_i * d_field_size + x_i;

  float rtx = (d_phase_term_2_tmp_x[i + 1] - d_phase_term_2_tmp_x[i - 1]) / d_dx;
  float rty = (d_phase_term_2_tmp_y[i + field_size] - d_phase_term_2_tmp_y[i - field_size]) / d_dx;

  d_phase_term_2[i] = -rtx + rty;
  return;
}

__global__ void calc_phase_term_2_tmp(float *d_rpx, float *d_rpy, float *d_theta, float *d_phase_term_2_tmp_x, float *d_phase_term_2_tmp_y) {
  int x_i = blockIdx.x * blockDim.x + threadIdx.x;
  int y_i = blockIdx.y * blockDim.y + threadIdx.y;
  if (x_i <= 0 || x_i >= d_field_size - 1 || y_i <= 0 || y_i >= d_field_size - 1) return;
  int i = y_i * d_field_size + x_i;

  float a = get_a(d_theta[i]);
  float rat = get_rat(d_theta[i]);

  d_phase_term_2_tmp_x[i] = a * rat * d_rpy[i];
  d_phase_term_2_tmp_y[i] = a * rat * d_rpx[i];
  return;
}

__global__ void calc_phase_term_3(float *d_phase, float *d_T,  curandState *state, float *d_phase_term_3) {
  int x_i = blockIdx.x * blockDim.x + threadIdx.x;
  int y_i = blockIdx.y * blockDim.y + threadIdx.y;
  if (x_i <= 0 || x_i >= d_field_size - 1 || y_i <= 0 || y_i >= d_field_size - 1) return;
  int i = y_i * d_field_size + x_i;

  float chi =  2. * d_chi * curand_normal(&state[i]) - d_chi;
  d_phase_term_3[i] = 4. * d_W * d_phase[i] * (1. - d_phase[i])
    * (d_phase[i] - .5 - 15. / 2. / d_W * d_L * (d_T[i] - d_Tm) / d_Tm * d_phase[i] * (1. - d_phase[i]) + chi);
}

__global__ void calc_phase_func(float *d_phase_term_1, float *d_phase_term_2, float *d_phase_term_3, float *d_phase_tmp) {
  int x_i = blockIdx.x * blockDim.x + threadIdx.x;
  int y_i = blockIdx.y * blockDim.y + threadIdx.y;
  if (x_i <= 0 || x_i >= d_field_size - 1 || y_i <= 0 || y_i >= d_field_size - 1) return;
  int i = y_i * d_field_size + x_i;

  d_phase_tmp[i] = d_M_phi * (d_phase_term_1[i] + d_phase_term_2[i] + d_phase_term_3[i]);
  return;
}

__global__ void calc_next_phase(float *d_phase_func, float *d_phase, float *d_phase_tmp) {
  int x_i = blockIdx.x * blockDim.x + threadIdx.x;
  int y_i = blockIdx.y * blockDim.y + threadIdx.y;
  if (x_i <= 0 || x_i >= d_field_size - 1 || y_i <= 0 || y_i >= d_field_size - 1) return;
  int i = y_i * d_field_size + x_i;

  d_phase_tmp[i] = d_phase[i] + d_phase_func[i] * d_dt;
  return;
}

__global__ void calc_next_T(float *d_T, float *d_phase, float *d_phase_d_t, float *d_T_tmp) {
  int x_i = blockIdx.x * blockDim.x + threadIdx.x;
  int y_i = blockIdx.y * blockDim.y + threadIdx.y;
  if (x_i <= 0 || x_i >= d_field_size - 1 || y_i <= 0 || y_i >= d_field_size - 1) return;
  int i = y_i * d_field_size + x_i;

  float rTx = (d_T[i + 1] - 2. * d_T[i] + d_T[i - 1]) / d_dx / d_dx;
  float rTy = (d_T[i + d_field_size] - 2. * d_T[i] + d_T[i - d_field_size]) / d_dx / d_dx;
  float term_1 = d_kappa * (rTx + rTy);
  float term_2 = 30.* pow(d_phase[i], 2.) * pow((1. - d_phase[i]), 2.) * d_L / d_c * d_phase_d_t[i];
  d_T_tmp[i] = d_T[i] + (term_1 + term_2) * d_dt;
  return;
}

__global__ void calc_phase_nabla(float *d_phase, float *d_rpx, float *d_rpy) {
  int x_i = blockIdx.x * blockDim.x + threadIdx.x;
  int y_i = blockIdx.y * blockDim.y + threadIdx.y;
  if (x_i <= 0 || x_i >= d_field_size - 1 || y_i <= 0 || y_i >= d_field_size - 1) return;
  int i = y_i * d_field_size + x_i;

  d_rpx[i] = (d_phase[i + 1] - d_phase[i - 1]) / d_dx;
  d_rpy[i] = (d_phase[i + d_field_size] - d_phase[i - d_field_size]) / d_dx;
  return;
}

__global__ void calc_theta(float *d_rpx, float *d_rpy, float *d_theta) {
  int x_i = blockIdx.x * blockDim.x + threadIdx.x;
  int y_i = blockIdx.y * blockDim.y + threadIdx.y;
  if (x_i <= 0 || x_i >= d_field_size - 1 || y_i <= 0 || y_i >= d_field_size - 1) return;
  int i = y_i * d_field_size + x_i;

  d_theta[i] = atan2(d_rpy[i], d_rpx[i]);
  return;
}


bool save(float *phase, float *T, unsigned int n) {
  try {
    std::ofstream file;
    std::ostringstream filename;
    filename << "datas/step_" << std::setfill('0') << std::right << std::setw(std::log10(step)+1) << n << ".dat";
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
  const unsigned int N = field_size * field_size;
  size_t size_field = N * sizeof(float);

  float *phase, *T; // phase field for host
  phase = (float *)malloc(size_field);
  T = (float *)malloc(size_field);

  // phase field for device
  float *d_phase, *d_phase_tmp, *d_T, *d_T_tmp;
  float *d_rpx, *d_rpy, *d_theta;
  float *d_phase_term_1;
  float *d_phase_term_2;
  float *d_phase_term_3;

  float *d_tmp_1, *d_tmp_2;

  // allocate memory to GPU
  cudaMalloc((void**)&d_phase, size_field);
  cudaMalloc((void**)&d_phase, size_field);
  cudaMalloc((void**)&d_T, size_field);
  cudaMalloc((void**)&d_T_tmp, size_field);
  cudaMalloc((void**)&d_rpx, size_field);
  cudaMalloc((void**)&d_rpy, size_field);
  cudaMalloc((void**)&d_theta, size_field);
  cudaMalloc((void**)&d_phase_term_1, size_field);
  cudaMalloc((void**)&d_phase_term_2, size_field);
  cudaMalloc((void**)&d_phase_term_3, size_field);

  cudaMalloc((void**)&d_tmp_1, size_field);
  cudaMalloc((void**)&d_tmp_2, size_field);


  // 異方性強度
  float xi = .01;
  // 無次元過冷却度
  float Delta = .5;

  const float dx = 20e-9;
  // 熱伝導率
  const float K = 84.01;
  // 比熱
  const float c = 5.42e+6;
  // 潜熱
  const float L = 2.350e+9;
  // 融点
  const float Tm = 1728.;
  // 界面キネティック係数
  const float mu = 2.;
  // ゆらぎ
  const float chi = .1;
  // 優先成長方向
  const float theta_0 = 0.;
  // 界面エネルギー
  const float gamma = 0.37;
  // 異方性モード
  const float k = 4.;
  // 界面幅
  const float delta = 4. * dx;
  // 界面領域
  const float lambda = .1;
  // 勾配計数
  const float b = 2. * std::atanh(1.-2.*lambda);
  const float a_bar = std::sqrt(3. * delta * gamma / b);
  // エネルギー障壁
  const float W = 6. * gamma * b / delta;
  // フェーズフィールドモビリティ
  const float M_phi = b * Tm * mu / 3. / delta / L;
  // 熱拡散係数
  const float kappa = K / c;
  // 時間ステップ
  const float dt1 = dx * dx / 5. / M_phi / a_bar / a_bar;
  const float dt2 = dx * dx / 5. / kappa;
  const float dt = std::min(dt1, dt2);
  printf("Time Step: %.3e[s]\n", dt);
  // 固相初期半径
  const float r_0 = 2. * dx;
  // 無次元過冷却温度
  const float T_0 = Tm - Delta * L / c;


  cudaMemcpyToSymbol(d_field_size, &field_size, sizeof(unsigned int));

  size_t size_val = sizeof(float);
  cudaMemcpyToSymbol(d_dx, &dx, size_val);
  cudaMemcpyToSymbol(d_a_bar, &a_bar, size_val);
  cudaMemcpyToSymbol(d_xi, &xi, size_val);
  cudaMemcpyToSymbol(d_k, &k, size_val);
  cudaMemcpyToSymbol(d_theta_0, &theta_0, size_val);
  cudaMemcpyToSymbol(d_a_bar, &a_bar, size_val);
  cudaMemcpyToSymbol(d_W, &W, size_val);
  cudaMemcpyToSymbol(d_Tm, &Tm, size_val);
  cudaMemcpyToSymbol(d_L, &L, size_val);
  cudaMemcpyToSymbol(d_chi, &chi, size_val);
  cudaMemcpyToSymbol(d_M_phi, &M_phi, size_val);
  cudaMemcpyToSymbol(d_dt, &dt, size_val);
  cudaMemcpyToSymbol(d_c, &c, size_val);
  cudaMemcpyToSymbol(d_kappa, &kappa, size_val);

  // calc blocks
  int threadsPerBlock = 32;
  int blocksInGrid = (field_size + threadsPerBlock -1)/threadsPerBlock;
  dim3 blocks(threadsPerBlock, threadsPerBlock);
  dim3 grid(blocksInGrid, blocksInGrid);

  // set randam seed
  curandState *state;
  cudaMalloc((void**)&state, N * sizeof(curandState));
  setCurand<<<grid, blocks>>>(time(NULL), state);

  // set initial conditions
  init_field<<<grid, blocks>>>(d_phase, d_T, r_0, T_0);
  set_bc<<<1, field_size -2>>>(d_phase);
  set_bc<<<1, field_size -2>>>(d_T);

  // メインループ
  for (unsigned int n = 0; n < step; n++) {
    printf("step: %d\n", n);

    // Copy Phase field from Device
    cudaMemcpy(phase, d_phase, size_field, cudaMemcpyDeviceToHost);
    cudaMemcpy(T, d_T, size_field, cudaMemcpyDeviceToHost);
    if ( n == step - 1 ) {
      save(phase, T, n);
    }

    calc_phase_nabla<<<grid, blocks>>>(d_phase, d_rpx, d_rpy);
    calc_theta<<<grid, blocks>>>(d_rpx, d_rpy, d_theta);

    calc_phase_term_1_tmp<<<grid, blocks>>>(d_rpx, d_rpy, d_phase, d_tmp_1);
    set_bc<<<1, field_size -2>>>(d_tmp_1);
    calc_phase_term_1<<<grid, blocks>>>(d_tmp_1, d_phase_term_1);

    calc_phase_term_2_tmp<<<grid, blocks>>>(d_rpx, d_rpy, d_theta, d_tmp_1, d_tmp_2);
    set_bc<<<1, field_size -2>>>(d_tmp_1);
    set_bc<<<1, field_size -2>>>(d_tmp_2);
    calc_phase_term_2<<<grid, blocks>>>(d_tmp_1, d_tmp_2, d_phase_term_2);

    calc_phase_term_3<<<grid, blocks>>>(d_phase, d_T, state, d_phase_term_3);
    calc_phase_func<<<grid, blocks>>>(d_phase_term_1, d_phase_term_2, d_phase_term_3, d_tmp_1);

    calc_next_phase<<<grid, blocks>>>(d_tmp_1, d_phase, d_phase_tmp);
    calc_next_T<<<grid, blocks>>>(d_T, d_phase, d_tmp_1, d_T_tmp);

    // Swap
    cudaMemcpy(d_phase, d_phase_tmp, size_field, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_T, d_T_tmp, size_field, cudaMemcpyDeviceToDevice);

    // Boundary Condition
    set_bc<<<1, field_size -2>>>(d_phase);
    set_bc<<<1, field_size -2>>>(d_T);

  }

  free(phase);
  free(T);

  cudaFree(d_phase);
  cudaFree(d_phase_tmp);
  cudaFree(d_T);
  cudaFree(d_T_tmp);
  cudaFree(d_rpx);
  cudaFree(d_rpy);
  cudaFree(d_theta);
  cudaFree(d_phase_term_1);
  cudaFree(d_phase_term_2);
  cudaFree(d_phase_term_3);
  cudaFree(d_tmp_1);
  cudaFree(d_tmp_2);
  cudaFree(state);

  return 0;
}
