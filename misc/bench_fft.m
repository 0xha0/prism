function results = bench_fft(varargin)
% bench_fft  与 bench_fft.cpp 类似的 FFT 变换微基准
%
% 用法:
%   bench_fft
%   bench_fft('Iterations',200)
%   bench_fft('UseGPU',true)
%   bench_fft('Sizes',[256 1024 4096 16384 65536 262144 1048576])
%   bench_fft('BatchSizes',[8 16 32 64])
%
% 输出:
%   results: 包含不同精度与设备结果表的结构体

% ----------------- 参数解析 -----------------
p = inputParser;
p.addParameter('Sizes',      [256 1024 4096 16384 65536 262144]);
p.addParameter('BatchSizes', [8 16 32 64]);
p.addParameter('Iterations', 256);
p.addParameter('UseGPU',     false);
p.parse(varargin{:});
opt = p.Results;

sizes      = opt.Sizes(:).';
batchSizes = opt.BatchSizes(:).';
iters      = opt.Iterations;
useGPU     = logical(opt.UseGPU);

% ----------------- 后端信息 -----------------
fprintf('\n==================== FFT Bench ====================\n');
fprintf('MATLAB: %s\n', version);
fprintf('CPU threads (max): %d\n', maxNumCompThreads);
if useGPU
  try
    g = gpuDevice;
    fprintf('GPU: %s (ComputeCapability: %s)\n', g.Name, g.ComputeCapability);
  catch
    fprintf('GPU: unavailable (disable UseGPU or check toolbox/device)\n');
    useGPU = false;
  end
end
fprintf('Iterations per case: %d\n', iters);
fprintf('Sizes: %s\n', mat2str(sizes));
fprintf('BatchSizes: %s\n', mat2str(batchSizes));
fprintf('===================================================\n');

results = struct();
% CPU 单精度/双精度
results.cpu.single  = run_one('cpu', 'single',  sizes, batchSizes, iters, false);
results.cpu.double  = run_one('cpu', 'double',  sizes, batchSizes, iters, false);
if useGPU
  % GPU 单精度/双精度
  results.gpu.single = run_one('gpu', 'single', sizes, batchSizes, iters, true);
  results.gpu.double = run_one('gpu', 'double', sizes, batchSizes, iters, true);
end

end

% ============================================================
function T = run_one(where, precision, sizes, batchSizes, iters, useGPU)
fprintf('\n--- FFT %s (%s) ---\n', upper(precision), where);
print_header();

rows = {};
for N = sizes
  % 准备数据（列向量或批量矩阵）
  if strcmpi(precision,'single')
    realT = 'single';
  else
    realT = 'double';
  end

  % C2C (一维复数到复数)
  xC = complex(randn(N,1,realT), randn(N,1,realT));
  if useGPU
    xC = gpuArray(xC);
  end
  c2c_ms = bench_ms(@() fft(xC), iters, useGPU);
  print_row(N, "C2C", c2c_ms, double(N));

  rows(end+1,:) = {where, precision, N, 1, "C2C", c2c_ms, throughput_msps(double(N), c2c_ms)}; %#ok<AGROW>

  % 批量 C2C
  for B = batchSizes
    if B <= 0, continue; end
    X = complex(randn(N,B,realT), randn(N,B,realT));
    if useGPU
      X = gpuArray(X);
    end
    b_ms = bench_ms(@() fft(X, [], 1), iters, useGPU);
    label = "C2C x" + string(B);
    print_row(N, label, b_ms, double(N)*double(B));
    rows(end+1,:) = {where, precision, N, B, "C2C_BATCH", b_ms, throughput_msps(double(N)*double(B), b_ms)}; %#ok<AGROW>
  end

  % R2C（实输入 FFT，保留半谱）
  xR = randn(N,1,realT);
  if useGPU
    xR = gpuArray(xR);
  end
  r2c_ms = bench_ms(@() r2c_half(xR), iters, useGPU);
  print_row(N, "R2C", r2c_ms, double(N));
  rows(end+1,:) = {where, precision, N, 1, "R2C", r2c_ms, throughput_msps(double(N), r2c_ms)}; %#ok<AGROW>

  % C2R（从半谱重建全谱，再 ifft 取实部）
  % 复用已有半谱缓冲区，降低分配开销对计时的影响
  h = r2c_half(xR); % 半谱
  c2r_ms = bench_ms(@() c2r_from_half(h, N), iters, useGPU);
  print_row(N, "C2R", c2r_ms, double(N));
  rows(end+1,:) = {where, precision, N, 1, "C2R", c2r_ms, throughput_msps(double(N), c2r_ms)}; %#ok<AGROW>
end

T = cell2table(rows, 'VariableNames', ...
  {'where','precision','N','batch','op','time_ms','throughput_MSps'});
end

% ============================================================
function print_header()
kSizeWidth = 12;
kOpWidth = 12;
kTimeWidth = 15;
kThrWidth = 15;
% 列标题与分隔线
fprintf('%*s%*s%*s%*s\n', kSizeWidth,'Size', kOpWidth,'Op', kTimeWidth,'Time (ms)', kThrWidth,'Throughput');
fprintf('%s\n', repmat('-', 1, kSizeWidth+kOpWidth+kTimeWidth+kThrWidth));
end

function print_row(N, op, time_ms, throughputSize)
kSizeWidth = 12;
kOpWidth = 12;
kTimeWidth = 15;
kThrWidth = 15;

fprintf('%*d%*s', kSizeWidth, N, kOpWidth, op);

if isnan(time_ms)
  % 计时失败时输出占位
  fprintf('%*s', kTimeWidth, 'n/a');
  fprintf('%*s\n', kThrWidth, 'n/a');
  return;
end

fprintf('%*s', kTimeWidth, sprintf('%.3fms', time_ms));

thr = throughput_msps(double(throughputSize), time_ms);
if isnan(thr)
  fprintf('%*s\n', kThrWidth, 'n/a');
else
  fprintf('%*s\n', kThrWidth, sprintf('%.1f Mpt/s', thr));
end
end

% ============================================================
function ms = bench_ms(fun, iters, useGPU)
% 热身运行，触发 JIT/GPU 初始化
try
  fun();
  if useGPU
    wait(gpuDevice);
  end
catch
  ms = NaN;
  return;
end

% 计时：固定迭代次数取平均
try
  if useGPU
    wait(gpuDevice);
  end
  t0 = tic;
  for i = 1:iters
    fun();
  end
  if useGPU
    wait(gpuDevice);
  end
  sec = toc(t0) / iters;
  ms = sec * 1e3;
catch
  ms = NaN;
end
end

function thr = throughput_msps(sizePoints, time_ms)
% Mpt/s = (点数 / 秒) / 1e6
if time_ms <= 0 || sizePoints <= 0 || isnan(time_ms)
  thr = NaN; return;
end
thr = (sizePoints * 1e-6) / (time_ms * 1e-3);
end

% ============================================================
function h = r2c_half(xR)
% R2C：对实数输入做 FFT，并保留半谱
X = fft(xR);
N = size(xR,1);
L = floor(N/2) + 1;
h = X(1:L, :);
end

function y = c2r_from_half(h, N)
% C2R：重建共轭对称全谱，再 ifft 取实部
Xfull = half_to_full(h, N);
y = real(ifft(Xfull));
end

function X = half_to_full(H, N)
% H： (floor(N/2)+1) x B
% 返回满足共轭对称的 N x B 全谱
L = floor(N/2) + 1;
B = size(H,2);
if size(H,1) ~= L
  error('half_to_full: invalid half length');
end

X = zeros(N, B, 'like', H);
X(1:L, :) = H;

if mod(N,2) == 0
  % 偶数 N：频点 2..L-1 镜像到 N..L+1
  if L > 2
    X(L+1:N, :) = conj(H(L-1:-1:2, :));
  end
else
  % 奇数 N：频点 2..L 镜像到 N..L+1
  if L > 1
    X(L+1:N, :) = conj(H(L:-1:2, :));
  end
end
end
