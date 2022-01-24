#pragma once
#include <vector>
#include <unordered_set>
#include "_main.hxx"
#include "duplicate.hxx"
#include "transpose.hxx"
#include "deadEnds.hxx"
#include "pagerank.hxx"
#include "pagerankPlain.hxx"

using std::vector;
using std::unordered_set;




template <class G, class H, class J, class T>
void pagerankRemoveCalculate(vector<T>& a, const G& xr, const H& xt, const J& ks, T p) {
  using K = typename G::key_type;
  a.resize(xt.span());            // ensure bounds!
  K N = coalesce(xr.order(), 1);  // can be empty!
  for (auto u : ks) {
    a[u] = (1-p)/N;
    for (auto v : xt.edgeKeys(u))
      a[u] += (p/coalesce(xr.degree(v), 1)) * a[v];  // degree can be 0!
  }
}




// Find pagerank by removing dead ends initially, and calculating their ranks after convergence (pull, CSR).
// @param x original graph
// @param q initial ranks (optional)
// @param o options {damping=0.85, tolerance=1e-6, maxIterations=500}
// @returns {ranks, iterations, time}
template <class G, class T=float>
PagerankResult<T> pagerankRemove(const G& x, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  T p = o.damping;
  auto xr = duplicate(x, [&](auto u) { return !isDeadEnd(x, u); });
  auto a  = pagerankPlain(xr, q, o);
  auto xt = transposeWithDegree(x);
  auto ks = deadEnds(x);
  a.time += measureDuration([&] { pagerankRemoveCalculate(a.ranks, xr, xt, ks, p); }, o.repeat);
  multiplyValue(a.ranks, a.ranks, T(1)/coalesce(sum(a.ranks), T(1)));
  return a;
}


template <class G, class T=float>
PagerankResult<T> pagerankRemoveDynamic(const G& x, const G& y, const vector<T> *q=nullptr, PagerankOptions<T> o={}) {
  T p = o.damping;
  auto yr = duplicate(y, [&](auto u) { return !isDeadEnd(y, u); });
  auto a  = pagerankPlainDynamic(x, yr, q, o);
  auto yt = transposeWithDegree(y);
  auto ks = deadEnds(y);
  a.time += measureDuration([&] { pagerankRemoveCalculate(a.ranks, yr, yt, ks, p); }, o.repeat);
  multiplyValue(a.ranks, a.ranks, T(1)/coalesce(sum(a.ranks), T(1)));
  return a;
}
