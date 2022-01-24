#pragma once
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"
#include "edges.hxx"
#include "csr.hxx"
#include "pagerank.hxx"
#include "pagerankSeq.hxx"

using std::array;
using std::vector;
using std::sqrt;
using std::partition;
using std::swap;
using std::min;
using std::max;




// PAGERANK-PARTITON
// -----------------

template <class H, class K>
void pagerankPartition(const H& xt, vector<K>& ks, K i, K n) {
  auto ib = ks.begin()+i, ie = ib+n;
  partition(ib, ie, [&](auto u) { return xt.degree(u) < SWITCH_DEGREE_PRC(); });
}

template <class H, class K>
void pagerankPartition(const H& xt, vector<K>& ks) {
  pagerankPartition(xt, ks, 0, ks.size());
}




// PAGERANK-COMPONENTS
// -------------------

template <class G, class H, class T>
auto pagerankCudaComponents(const G& x, const H& xt, const PagerankOptions<T>& o, const PagerankData<G> *D=nullptr) {
  auto cs = pagerankComponents(x, xt, o, D);
  auto a  = joinUntilSizeVector(cs, o.minCompute);
  for (auto& ks : a)
    pagerankPartition(xt, ks);
  return a;
}


template <class G, class H, class T>
auto pagerankCudaDynamicComponents(const G& x, const H& xt, const G& y, const H& yt, const PagerankOptions<T>& o, const PagerankData<G> *D=nullptr) {
  auto [cs, n] = pagerankDynamicComponents(x, xt, y, yt, o, D);
  auto a = joinUntilSizeVector(sliceIter(cs, 0, n), o.minCompute);
  for (auto& ks : a)
    pagerankPartition(xt, ks);
  a.push_back(joinValuesVector(sliceIterable(cs, n)));
  return make_pair(a, a.size()-1);
}




// PAGERANK-FACTOR
// ---------------
// For contribution factors of vertices (unchanging).

template <class T, class K>
__global__ void pagerankFactorKernel(T *a, const K *vdata, K i, K n, T p) {
  DEFINE(t, b, B, G);
  for (K v=i+B*b+t; v<i+n; v+=G*B) {
    K d = vdata[v];
    a[v] = d>0? p/d : 0;
  }
}

template <class T, class K>
void pagerankFactorCu(T *a, const K *vdata, K i, K n, T p) {
  int B = BLOCK_DIM_M<T>();
  int G = min(ceilDiv(n, B), GRID_DIM_M<T>());
  pagerankFactorKernel<<<G, B>>>(a, vdata, i, n, p);
}




// PAGERANK-BLOCK
// --------------

template <class T, class O, class K, int S=BLOCK_LIMIT>
__global__ void pagerankBlockKernel(T *a, const T *c, const O *vfrom, const K *efrom, K i, K n, T c0) {
  DEFINE(t, b, B, G);
  __shared__ T cache[S];
  for (K v=i+b; v<i+n; v+=G) {
    K ebgn = vfrom[v];
    K ideg = vfrom[v+1]-vfrom[v];
    cache[t] = sumAtKernelLoop(c, efrom+ebgn, ideg, t, B);
    sumKernelReduce(cache, B, t);
    if (t==0) a[v] = c0 + cache[0];
  }
}

template <class T, class O, class K>
void pagerankBlockCu(T *a, const T *c, const O *vfrom, const K *efrom, K i, K n, T c0) {
  int B = BLOCK_DIM_PRCB<T>();
  int G = min(n, GRID_DIM_PRCB<T>());
  pagerankBlockKernel<<<G, B>>>(a, c, vfrom, efrom, i, n, c0);
}




// PAGERANK-THREAD
// ---------------

template <class T, class O, class K>
__global__ void pagerankThreadKernel(T *a, const T *c, const O *vfrom, const K *efrom, K i, K n, T c0) {
  DEFINE(t, b, B, G);
  for (K v=i+B*b+t; v<i+n; v+=G*B) {
    K ebgn = vfrom[v];
    K ideg = vfrom[v+1]-vfrom[v];
    a[v] = c0 + sumAtKernelLoop(c, efrom+ebgn, ideg, 0, 1);
  }
}

template <class T, class O, class K>
void pagerankThreadCu(T *a, const T *c, const O *vfrom, const K *efrom, K i, K n, T c0) {
  int B = BLOCK_DIM_PRCT<T>();
  int G = min(ceilDiv(n, B), GRID_DIM_PRCT<T>());
  pagerankThreadKernel<<<G, B>>>(a, c, vfrom, efrom, i, n, c0);
}




// PAGERANK-SWITCHED
// -----------------

template <class T, class O, class K, class J>
void pagerankSwitchedCu(T *a, const T *c, const O *vfrom, const K *efrom, K i, const J& ns, T c0) {
  for (auto n : ns) {
    if (n>0) pagerankBlockCu (a, c, vfrom, efrom, i,  n, c0);
    else     pagerankThreadCu(a, c, vfrom, efrom, i, -n, c0);
    i += abs(n);
  }
}




// PAGERANK-SWITCH-POINT
// ---------------------

template <class H, class J>
auto pagerankSwitchPoint(const H& xt, const J& ks) {
  using K = typename H::key_type;
  K a = countIf(ks, [&](auto u) { return xt.degree(u) < SWITCH_DEGREE_PRC(); });
  K L = SWITCH_LIMIT_PRC(), N = ks.size();
  return a<L? 0 : (N-a<L? N : a);
}




// PAGERANK-WAVE
// -------------

template <class H, class K>
void pagerankPairWave(vector<array<K, 2>>& a, const H& xt, const vector2d<K>& cs) {
  for (const auto& ks : cs) {
    K N = ks.size();
    K s = pagerankSwitchPoint(xt, ks);
    a.push_back({s, N-s});
  }
}

template <class H, class K>
auto pagerankPairWave(const H& xt, const vector2d<K>& cs) {
  vector<array<K, 2>> a; pagerankPairWave(a, xt, cs);
  return a;
}


template <class K>
void pagerankAddStep(vector<K>& a, K n) {
  if (n==0) return;
  if (a.empty() || sgn(a.back())!=sgn(n)) a.push_back(n);
  else a.back() += n;
}

template <class H, class J, class K>
void pagerankWave(vector<K>& a, const H& xt, const J& ks) {
  auto N = ks.size();
  auto s = pagerankSwitchPoint(xt, ks);
  pagerankAddStep(a,  -s);
  pagerankAddStep(a, N-s);
}

template <class H, class J>
auto pagerankWave(const H& xt, const J& ks) {
  using K = typename H::key_type;
  vector<K> a; pagerankWave(a, xt, ks);
  return a;
}

template <class H, class J, class K>
void pagerankComponentWave(vector<K>& a, const H& xt, const J& cs) {
  for (const auto& ks : cs)
    pagerankWave(a, xt, ks);
}
template <class H, class J>
auto pagerankComponentWave(const H& xt, const J& cs) {
  using K = typename H::key_type;
  vector<K> a; pagerankComponentWave(a, xt, cs);
  return a;
}




// PAGERANK-ERROR
// --------------
// For convergence check.

template <class T, class K>
void pagerankErrorCu(T *a, const T *x, const T *y, K N, int EF) {
  switch (EF) {
    case 1:  l1NormCu(a, x, y, N); break;
    case 2:  l2NormCu(a, x, y, N); break;
    default: liNormCu(a, x, y, N); break;
  }
}

template <class T, class K>
T pagerankErrorReduce(const T *x, K N, int EF) {
  switch (EF) {
    case 1:  return sumValues(x, N);
    case 2:  return sqrt(sumValues(x, N));
    default: return maxValue(x, N);
  }
}




// PAGERANK
// --------
// For Monolithic / Componentwise PageRank.

template <class H, class J, class K, class M, class FL, class T=float>
PagerankResult<T> pagerankCuda(const H& xt, const J& ks, K i, const M& ns, FL fl, const vector<T> *q, const PagerankOptions<T>& o) {
  K    N  = xt.order();
  T    p  = o.damping;
  T    E  = o.tolerance;
  int  L  = o.maxIterations, l = 0;
  int  EF = o.toleranceNorm;
  int  R  = reduceSizeCu<T>(N);
  auto vfrom = sourceOffsets(xt, ks);
  auto efrom = destinationIndices(xt, ks);
  auto vdata = vertexData(xt, ks);
  size_t VFROM1 = vfrom.size() * sizeof(size_t);
  size_t EFROM1 = efrom.size() * sizeof(K);
  size_t VDATA1 = vdata.size() * sizeof(K);
  size_t N1 = N * sizeof(T);
  size_t R1 = R * sizeof(T);
  vector<T> a(N), r(N), qc;
  if (q) qc = compressContainer(xt, *q, ks);

  T *e,  *r0;
  T *eD, *r0D, *fD, *rD, *cD, *aD;
  size_t *vfromD; K *efromD, *vdataD;
  // TRY( cudaProfilerStart() );
  TRY( cudaSetDeviceFlags(cudaDeviceMapHost) );
  TRY( cudaHostAlloc(&e,  R1, cudaHostAllocDefault) );
  TRY( cudaHostAlloc(&r0, R1, cudaHostAllocDefault) );
  TRY( cudaMalloc(&vfromD, VFROM1) );
  TRY( cudaMalloc(&efromD, EFROM1) );
  TRY( cudaMalloc(&vdataD, VDATA1) );
  TRY( cudaMalloc(&aD, N1) );
  TRY( cudaMalloc(&rD, N1) );
  TRY( cudaMalloc(&cD, N1) );
  TRY( cudaMalloc(&fD, N1) );
  TRY( cudaMalloc(&eD,  R1) );
  TRY( cudaMalloc(&r0D, R1) );
  TRY( cudaMemcpy(vfromD, vfrom.data(), VFROM1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(efromD, efrom.data(), EFROM1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(vdataD, vdata.data(), VDATA1, cudaMemcpyHostToDevice) );

  float t = measureDurationMarked([&](auto mark) {
    if (q) copy(r, qc);    // copy old ranks (qc), if given
    else fill(r, T(1)/N);
    TRY( cudaMemcpy(aD, r.data(), N1, cudaMemcpyHostToDevice) );
    TRY( cudaMemcpy(rD, r.data(), N1, cudaMemcpyHostToDevice) );
    mark([&] { pagerankFactorCu(fD, vdataD, 0, N, p); multiplyCu(cD, aD, fD, N); });                       // calculate factors (fD) and contributions (cD)
    mark([&] { l = fl(e, r0, eD, r0D, aD, rD, cD, fD, vfromD, efromD, i, ns, N, p, E, L, EF); });  // calculate ranks of vertices
  }, o.repeat);
  TRY( cudaMemcpy(a.data(), aD, N1, cudaMemcpyDeviceToHost) );

  TRY( cudaFreeHost(e) );
  TRY( cudaFreeHost(r0) );
  TRY( cudaFree(eD) );
  TRY( cudaFree(r0D) );
  TRY( cudaFree(aD) );
  TRY( cudaFree(rD) );
  TRY( cudaFree(cD) );
  TRY( cudaFree(fD) );
  TRY( cudaFree(vfromD) );
  TRY( cudaFree(efromD) );
  TRY( cudaFree(vdataD) );
  // TRY( cudaProfilerStop() );
  return {decompressContainer(xt, a, ks), l, t};
}
