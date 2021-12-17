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
auto pagerankVertices(const G& x, const H& xt, const PagerankOptions<T>& o) {
  if (!o.splitComponents) return vertices(xt);
  if (!o.sortComponents)  return join(components(x, xt));
  return join(topologicalComponents(x, xt));
}


template <class G, class H, class T>
auto pagerankDynamicVertices(const G& x, const H& xt, const G& y, const H& yt, const PagerankOptions<T>& o) {
  if (!o.splitComponents) return dynamicVertices(x, xt, y, yt);
  auto cs = components(y, yt);
  auto b  = blockgraph(y, cs);
  if (o.sortComponents) topologicalComponentsTo(cs, b);
  auto [is, n] = dynamicComponentIndices(x, xt, y, yt, cs, b);
  auto ks = joinAt(cs, sliceIter(is, 0, n)); int nv = ks.size();
  joinAt(ks, cs, sliceIter(is, n));
  return make_pair(ks, nv);
}




// PAGERANK-COMPONENTS
// -------------------

template <class G, class H, class T>
auto pagerankComponents(const G& x, const H& xt, const PagerankOptions<T>& o) {
  if (!o.splitComponents) return vector2d<int> {vertices(xt)};
  if (!o.sortComponents)  return components(x, xt);
  return topologicalComponents(x, xt);
}


template <class G, class H>
auto pagerankDynamicComponentsDefault(const G& x, const H& xt, const G& y, const H& yt) {
  vector2d<int> a;
  auto [ks, n] = dynamicVertices(x, xt, y, yt);
  a.push_back(vector<int>(ks.begin(), ks.begin()+n));
  a.push_back(vector<int>(ks.begin()+n, ks.end()));
  return make_pair(a, 1);
}

template <class G, class H, class T>
auto pagerankDynamicComponentsSplit(const G& x, const H& xt, const G& y, const H& yt, const PagerankOptions<T>& o) {
  auto cs = components(y, yt);
  auto b  = blockgraph(y, cs);
  if (o.sortComponents) topologicalComponentsTo(cs, b);
  auto [is, n] = dynamicComponentIndices(x, xt, y, yt, cs, b);
  vector2d<int> a;
  for (int i : is)
    a.push_back(move(cs[i]));
  return make_pair(a, n);
}

template <class G, class H, class T>
auto pagerankDynamicComponents(const G& x, const H& xt, const G& y, const H& yt, const PagerankOptions<T>& o) {
  if (o.splitComponents) return pagerankDynamicComponentsSplit(x, xt, y, yt, o);
  return pagerankDynamicComponentsDefault(x, xt, y, yt);
}




// PAGERANK-FACTOR
// ---------------
// For contribution factors of vertices (unchanging).

template <class T>
void pagerankFactor(vector<T>& a, const vector<int>& vdata, int i, int n, T p) {
  for (int u=i; u<i+n; u++) {
    int d = vdata[u];
    a[u] = d>0? p/d : 0;
  }
}




// PAGERANK-CALCULATE
// ------------------
// For rank calculation from in-edges.

template <class T>
void pagerankCalculate(vector<T>& a, const vector<T>& c, const vector<int>& vfrom, const vector<int>& efrom, int i, int n, T c0) {
  for (int v=i; v<i+n; v++)
    a[v] = c0 + sumAt(c, sliceIter(efrom, vfrom[v], vfrom[v+1]));
}




// PAGERANK-ERROR
// --------------
// For convergence check.

template <class T>
T pagerankError(const vector<T>& x, const vector<T>& y, int i, int N, int EF) {
  switch (EF) {
    case 1:  return l1Norm(x, y, i, N);
    case 2:  return l2Norm(x, y, i, N);
    default: return liNorm(x, y, i, N);
  }
}




// PAGERANK
// --------
// For Monolithic / Componentwise PageRank.

template <class H, class J, class M, class FL, class T=float>
PagerankResult<T> pagerankSeq(const H& xt, const J& ks, int i, const M& ns, FL fl, const vector<T> *q, const PagerankOptions<T>& o) {
  int  N  = xt.order();
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
    if (q) copy(r, qc);    // copy old ranks (qc), if given
    else fill(r, T(1)/N);
    copy(a, r);
    mark([&] { pagerankFactor(f, vdata, 0, N, p); multiply(c, a, f, 0, N); });  // calculate factors (f) and contributions (c)
    mark([&] { l = fl(a, r, c, f, vfrom, efrom, i, ns, N, p, E, L, EF); });     // calculate ranks of vertices
  }, o.repeat);
  return {decompressContainer(xt, a, ks), l, t};
}
