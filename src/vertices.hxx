#pragma once
#include <type_traits>
#include <algorithm>
#include <vector>
#include <iterator>
#include "_main.hxx"

using std::remove_reference_t;
using std::vector;
using std::transform;
using std::back_inserter;
using std::equal;




// VERTICES
// --------

template <class G, class F, class D>
auto vertices(const G& x, F fm, D fp) {
  using K = typename G::key_type; vector<K> a;
  copyAppend(x.vertexKeys(), a);
  auto ie = a.end(), ib = a.begin();
  fp(ib, ie); transform(ib, ie, ib, fm);
  return a;
}
template <class G, class F>
inline auto vertices(const G& x, F fm) {
  return vertices(x, fm, [](auto ib, auto ie) {});
}
template <class G>
inline auto vertices(const G& x) {
  return vertices(x, [](auto u) { return u; });
}




// VERTEX-DATA
// -----------

template <class G, class J, class F, class D>
auto vertexData(const G& x, const J& ks, F fm, D fp) {
  using K = typename G::key_type;
  using V = decltype(fm(0));
  vector<V> a; vector<K> b;
  copyAppend(ks, b);
  auto ie = b.end(), ib = b.begin();
  fp(ib, ie); transform(ib, ie, back_inserter(a), fm);
  return a;
}
template <class G, class J, class F>
inline auto vertexData(const G& x, const J& ks, F fm) {
  return vertexData(x, ks, fm, [](auto ib, auto ie) {});
}
template <class G, class J>
inline auto vertexData(const G& x, const J& ks) {
  return vertexData(x, ks, [&](auto u) { return x.vertexValue(u); });
}
template <class G>
inline auto vertexData(const G& x) {
  return vertexData(x, x.vertexKeys());
}




// CONTAINER
// ---------

template <class G, class T>
inline auto createContainer(const G& x, const T& _) {
  return vector<T>(x.span());
}
template <class G, class T>
inline auto createCompressedContainer(const G& x, const T& _) {
  return vector<T>(x.order());
}


template <class G, class T, class J>
inline void decompressContainer(vector<T>& a, const G& x, const vector<T>& vs, const J& ks) {
  scatterValues(vs, ks, a);
}
template <class G, class T>
inline void decompressContainer(vector<T>& a, const G& x, const vector<T>& vs) {
  decompressContainerTo(a, x, vs, x.vertexKeys());
}
template <class G, class T, class J>
inline auto decompressContainer(const G& x, const vector<T>& vs, const J& ks) {
  auto a = createContainer(x, T());
  decompressContainerTo(a, x, vs, ks);
  return a;
}
template <class G, class T>
inline auto decompressContainer(const G& x, const vector<T>& vs) {
  return decompressContainer(x, vs, x.vertexKeys());
}


template <class G, class T, class J>
inline void compressContainerTo(vector<T>& a, const G& x, const vector<T>& vs, const J& ks) {
  gatherValues(vs, ks, a);
}
template <class G, class T>
inline void compressContainerTo(vector<T>& a, const G& x, const vector<T>& vs) {
  return compressContainerTo(a, x, vs, x.vertexKeys());
}
template <class G, class T, class J>
inline auto compressContainer(const G& x, const vector<T>& vs, const J& ks) {
  auto a = createCompressedContainer(x, T());
  compressContainerTo(a, x, vs, ks);
  return a;
}
template <class G, class T>
inline auto compressContainer(const G& x, const vector<T>& vs) {
  return compressContainer(x, vs, x.vertexKeys());
}




// VERTICES-EQUAL
// --------------

template <class G, class K>
inline bool verticesEqual(const G& x, K u, const G& y, K v) {
  if (x.degree(u) != y.degree(v)) return false;
  return equalValues(x.edgeKeys(u), y.edgeKeys(v));
}
template <class G, class H, class K>
inline bool verticesEqual(const G& x, const H& xt, K u, const G& y, const H& yt, K v) {
  return verticesEqual(x, u, y, u) && verticesEqual(xt, u, yt, u);
}
