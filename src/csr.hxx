#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"

using std::vector;
using std::transform;




// SOURCE-OFFSETS
// --------------

template <class G, class J, class T>
auto sourceOffsetsAs(const G& x, const J& ks, T _) {
  T i = 0;
  vector<T> a;
  a.reserve(x.order()+1);
  for (auto u : ks) {
    a.push_back(i);
    i += x.degree(u);
  }
  a.push_back(i);
  return a;
}
template <class G, class T>
auto sourceOffsetsAs(const G& x, T _) {
  return sourceOffsetsAs(x, x.vertices(), T());
}

template <class G, class J>
auto sourceOffsets(const G& x, const J& ks) {
  return sourceOffsetsAs(x, ks, size_t());
}
template <class G>
auto sourceOffsets(const G& x) {
  return sourceOffsetsAs(x, size_t());
}




// DESTINATION-INDICES
// -------------------

template <class G, class J, class F>
auto destinationIndices(const G& x, const J& ks, F fp) {
  vector<int> a;
  auto ids = indices(ks);
  for (int u : ks) {
    append(a, x.edges(u));
    auto ie = a.end(), ib = ie-x.degree(u);
    fp(ib, ie); transform(ib, ie, ib, [&](int v) { return ids[v]; });
  }
  return a;
}

template <class G, class J>
auto destinationIndices(const G& x, const J& ks) {
  return destinationIndices(x, ks, [](auto ib, auto ie) {});
}

template <class G>
auto destinationIndices(const G& x) {
  return destinationIndices(x, x.vertices());
}
