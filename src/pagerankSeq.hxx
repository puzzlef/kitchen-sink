#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "vertices.hxx"
#include "edges.hxx"
#include "csr.hxx"
#include "pagerank.hxx"

using std::vector;
using std::swap;




// PAGERANK-VERTICES
// -----------------

template <class G, class H, class T>
auto pagerankVertices(const G& x, const H& xt, const PagerankOptions<T>& o, const PagerankData<G> *D=nullptr) {
  if (!o.splitComponents) return vertices(xt);
  return joinValuesVector(componentsD(x, xt, D));
}


template <class G, class H, class T>
auto pagerankDynamicVertices(const G& x, const H& xt, const G& y, const H& yt, const PagerankOptions<T>& o, const PagerankData<G> *D=nullptr) {
  if (!o.splitComponents) return dynamicVertices(x, xt, y, yt);
  const auto& cs = componentsD(y, yt, D);
  const auto& b  = blockgraphD(y, cs, D);
  auto [is, n] = dynamicComponentIndices(x, xt, y, yt, cs, b);
  auto ks = joinAtVector(cs, sliceIterable(is, 0, n)); size_t nv = ks.size();
  joinAt(cs, sliceIterable(is, n), ks);
  return make_pair(ks, nv);
}




// PAGERANK-COMPONENTS
// -------------------

template <class G, class H, class T>
auto pagerankComponents(const G& x, const H& xt, const PagerankOptions<T>& o, const PagerankData<G> *D=nullptr) {
  using K = typename G::key_type;
  if (!o.splitComponents) return vector2d<K> {vertices(xt)};
  return componentsD(x, xt, D);
}


template <class G, class H>
auto pagerankDynamicComponentsDefault(const G& x, const H& xt, const G& y, const H& yt) {
  using K = typename G::key_type; vector2d<K> a;
  auto [ks, n] = dynamicVertices(x, xt, y, yt);
  a.push_back(vector<K>(ks.begin(), ks.begin()+n));
  a.push_back(vector<K>(ks.begin()+n, ks.end()));
  return make_pair(a, size_t(1));
}

template <class G, class H, class T>
auto pagerankDynamicComponentsSplit(const G& x, const H& xt, const G& y, const H& yt, const PagerankOptions<T>& o, const PagerankData<G> *D=nullptr) {
  using K = typename G::key_type; vector2d<K> a;
  const auto& cs = componentsD(y, yt, D);
  const auto& b  = blockgraphD(y, cs, D);
  auto [is, n] = dynamicComponentIndices(x, xt, y, yt, cs, b);
  for (auto i : is)
    a.push_back(cs[i]);
  return make_pair(a, n);
}

template <class G, class H, class T>
auto pagerankDynamicComponents(const G& x, const H& xt, const G& y, const H& yt, const PagerankOptions<T>& o, const PagerankData<G> *D=nullptr) {
  if (o.splitComponents) return pagerankDynamicComponentsSplit(x, xt, y, yt, o, D);
  return pagerankDynamicComponentsDefault(x, xt, y, yt);
}




// PAGERANK-FACTOR
// ---------------
// For contribution factors of vertices (unchanging).

template <class T, class K>
void pagerankFactor(vector<T>& a, const vector<K>& vdata, K i, K n, T p) {
  for (K u=i; u<i+n; u++) {
    K d = vdata[u];
    a[u] = d>0? p/d : 0;
  }
}




// PAGERANK-CALCULATE
// ------------------
// For rank calculation from in-edges.

template <class T, class K>
void pagerankCalculate(vector<T>& a, const vector<T>& c, const vector<size_t>& vfrom, const vector<K>& efrom, K i, K n, T c0) {
  for (K v=i; v<i+n; v++)
    a[v] = c0 + sumValuesAt(c, sliceIterable(efrom, vfrom[v], vfrom[v+1]));
}




// PAGERANK-ERROR
// --------------
// For convergence check.

template <class T, class K>
T pagerankError(const vector<T>& x, const vector<T>& y, K i, K N, int EF) {
  switch (EF) {
    case 1:  return l1Norm(x, y, i, N);
    case 2:  return l2Norm(x, y, i, N);
    default: return liNorm(x, y, i, N);
  }
}




// PAGERANK
// --------
// For Monolithic / Componentwise PageRank.

template <class H, class J, class K, class M, class FL, class T=float>
PagerankResult<T> pagerankSeq(const H& xt, const J& ks, K i, const M& ns, FL fl, const vector<T> *q, const PagerankOptions<T>& o) {
  K    N  = xt.order();
  T    p  = o.damping;
  T    E  = o.tolerance;
  int  L  = o.maxIterations, l = 0;
  int  EF = o.toleranceNorm;
  auto vfrom = sourceOffsets(xt, ks);
  auto efrom = destinationIndices(xt, ks);
  auto vdata = vertexData(xt, ks);
  vector<T> a(N), r(N), c(N), f(N), qc;
  if (q) qc = compressContainer(xt, *q, ks);
  float t = measureDurationMarked([&](auto mark) {
    if (q) copyValues(qc, r);   // copy old ranks (qc), if given
    else fillValue(r, T(1)/N);
    copyValues(r, a);
    mark([&] { pagerankFactor(f, vdata, 0, N, p); multiplyValues(a, f, c, 0, N); });  // calculate factors (f) and contributions (c)
    mark([&] { l = fl(a, r, c, f, vfrom, efrom, i, ns, N, p, E, L, EF); });           // calculate ranks of vertices
  }, o.repeat);
  return {decompressContainer(xt, a, ks), l, t};
}
