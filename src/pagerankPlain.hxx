#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"
#include "edges.hxx"
#include "csr.hxx"
#include "transpose.hxx"
#include "dynamic.hxx"
#include "pagerank.hxx"

using std::vector;
using std::swap;




template <class T, class K>
void pagerankFactor(vector<T>& a, const vector<K>& vdata, K i, K n, T p) {
  for (K u=i; u<i+n; u++) {
    K d = vdata[u];
    a[u] = d>0? p/d : 0;
  }
}


template <class T, class K>
void pagerankCalculate(vector<T>& a, const vector<T>& c, const vector<size_t>& vfrom, const vector<K>& efrom, K i, K n, T c0) {
  for (K v=i; v<i+n; v++)
    a[v] = c0 + sumValuesAt(c, sliceIterable(efrom, vfrom[v], vfrom[v+1]));
}


template <class T, class K>
int pagerankPlainLoop(vector<T>& a, vector<T>& r, vector<T>& c, const vector<T>& f, const vector<size_t>& vfrom, const vector<K>& efrom, const vector<K>& vdata, K i, K n, K N, T p, T E, int L) {
  T c0 = (1-p)/N;
  int l = 1;
  for (; l<L; l++) {
    if (l==1) multiplyValues(r, f, c, 0, N);  // 1st time, find contrib for all
    else      multiplyValues(r, f, c, i, n);  // nth time, only those that changed
    pagerankCalculate(a, c, vfrom, efrom, i, n, c0);  // only changed
    T el = l1Norm(a, r, 0, N);  // full error check, partial can be done too (i, n)
    if (el < E) break;
    swap(a, r);
  }
  return l;
}


template <class H, class J, class K, class FL, class T=float>
PagerankResult<T> pagerankPlainCore(const H& xt, const J& ks, K i, K n, FL fl, const vector<T> *q, PagerankOptions<T> o) {
  K    N = xt.order();
  T    p = o.damping;
  T    E = o.tolerance;
  int  L = o.maxIterations, l = 0;
  auto vfrom = sourceOffsets(xt, ks);
  auto efrom = destinationIndices(xt, ks);
  auto vdata = vertexData(xt, ks);
  vector<T> a(N), r(N), c(N), f(N);
  float t = measureDurationMarked([&](auto mark) {
    if (q) r = compressContainer(xt, *q, ks);
    else fillValue(r, T(1)/N);
    copyValues(r, a);  // copy old ranks
    if (N==0 || n==0) return;  // skip if nothing to do!
    mark([&] { pagerankFactor(f, vdata, 0, N, p); });
    mark([&] { l = fl(a, r, c, f, vfrom, efrom, vdata, i, n, N, p, E, L); });  // with full error check, partial can be done too (E*n/N)
  }, o.repeat);
  return {decompressContainer(xt, a, ks), l, t};
}




// Find pagerank using a single thread (pull, CSR).
// @param x original graph
// @param q initial ranks (optional)
// @param o options {damping=0.85, tolerance=1e-6, maxIterations=500}
// @returns {ranks, iterations, time}
template <class G, class FL, class T=float>
PagerankResult<T> pagerankPlain(const G& x, FL fl, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  auto N  = x.order();
  auto xt = transposeWithDegree(x);
  return pagerankPlainCore(xt, xt.vertexKeys(), 0, N, fl, q, o);
}

template <class G, class T=float>
PagerankResult<T> pagerankPlain(const G& x, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  return pagerankPlain(x, pagerankPlainLoop<T>, q, o);
}


template <class G, class FD, class FL, class T=float>
PagerankResult<T> pagerankPlainDynamic(const G& x, const G& y, FD fd, FL fl, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  auto [ks, n] = fd(x, y);
  auto yt = transposeWithDegree(y);
  return pagerankPlainCore(yt, ks, 0, n, fl, q, o);
}

template <class G, class T=float>
PagerankResult<T> pagerankPlainDynamic(const G& x, const G& y, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  auto xt = transposeWithDegree(x);
  auto yt = transposeWithDegree(y);
  auto [ks, n] = dynamicVertices(x, xt, y, yt);
  return pagerankPlainCore(yt, ks, 0, n, pagerankPlainLoop<T>, q, o);
}
