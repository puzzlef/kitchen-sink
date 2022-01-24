#pragma once
#include <cmath>
#include <type_traits>
#include <iterator>
#include <array>
#include <vector>
#include <map>
#include <algorithm>
#include <utility>
#include "_openmp.hxx"

using std::remove_reference_t;
using std::iterator_traits;
using std::array;
using std::vector;
using std::map;
using std::abs;
using std::max;
using std::sqrt;
using std::swap;
using std::move;
using std::copy;
using std::fill;




// 2D/3D
// -----

template <class T>
using vector2d = vector<vector<T>>;

template <class T>
using vector3d = vector<vector<vector<T>>>;




// SIZE
// ----

template <class T>
inline size_t size(const vector<T>& x) {
  return x.size();
}
template <class T>
size_t size2d(const vector2d<T>& x) {
  size_t a = 0;
  for (const auto& v : x)
    a += size(v);
  return a;
}

template <class T>
size_t size3d(const vector3d<T>& x) {
  size_t a = 0;
  for (const auto& v : x)
    a += size2d(v);
  return a;
}




// REORDER
// -------
// Ref: https://stackoverflow.com/a/22183350/1413259

template <class T, class K>
void reorderDirty(vector<T>& x, vector<K>& is) {
  for(size_t i=0, N=x.size(); i<N; ++i) {
    while(is[i] != is[is[i]]) {
      swap(x[is[i]], x[is[is[i]]]);
      swap(  is[i],    is[is[i]]);
    }
  }
}
template <class T, class K>
inline void reorder(vector<T>& x, vector<K> is) {
  reorderDirty(x, is);
}




// ERASE
// -----

template <class T>
inline void eraseIndex(vector<T>& a, size_t i) {
  a.erase(a.begin()+i);
}
template <class T>
inline void eraseIndex(vector<T>& a, size_t i, size_t I) {
  a.erase(a.begin()+i, a.begin()+I);
}




// INSERT-VALUE-AT
// ---------------

template <class T>
inline void insertValueAt(vector<T>& a, size_t i, const T& v) {
  a.insert(a.begin()+i, v);
}
template <class T>
inline void insertValueAt(vector<T>& a, size_t i, size_t n, const T& v) {
  a.insert(a.begin()+i, n, v);
}




// PARITITON-VALUES
// ----------------

template <class J, class T, class F>
void partitionValues(const J& x, vector2d<T>& a, F fn) {
  for (const auto& v : x) {
    auto& b = a.back();
    if (a.empty() || !fn(b, v)) a.push_back({v});
    else b.push_back(v);
  }
}
template <class J, class F>
inline auto partitionValuesVector(const J& x, F fn) {
  using I = decltype(x.begin());
  using T = typename iterator_traits<I>::value_type;
  vector2d<T> a; partitionValues(x, a, fn);
  return a;
}




// JOIN
// ----

template <class J, class T, class F>
void joinIf(const J& xs, vector2d<T>& a, F fn) {
  for (const auto& x : xs) {
    auto& b = a.back();
    if (a.empty() || !fn(b, x)) a.push_back(x);
    else b.insert(b.end(), x.begin(), x.end());
  }
}
template <class J, class F>
inline auto joinIfVector(const J& xs, F fn) {
  using I = decltype(xs.begin());
  using X = typename iterator_traits<I>::value_type;
  using T = typename iterator_traits<X>::value_type;
  vector2d<T> a; joinIf(xs, a, fn);
  return a;
}


template <class T, class J>
void joinUntilSize(const J& xs, vector2d<T>& a, size_t S) {
  auto fn = [&](const auto& b, const auto& x) { return b.size()<S; };
  joinIf(xs, a, fn);
}
template <class T, class J>
inline auto joinUntilSizeVector(const J& xs, size_t S) {
  vector2d<T> a; joinUntilSize(xs, a, S);
  return a;
}


template <class T, class J>
void joinValues(const J& xs, vector<T>& a) {
  for (const auto& x : xs)
    a.insert(a.end(), x.begin(), x.end());
}
template <class T, class J>
auto joinValuesVector(const J& xs) {
  vector<T> a; join(xs, a);
  return a;
}




// JOIN-AT-*
// ---------

template <class T, class J, class F>
void joinAtIf(const vector2d<T>& xs, const J& is, vector2d<T>& a, F fn) {
  for (int i : is) {
    auto& b = a.back();
    if (a.empty() || !fn(b, xs[i])) a.push_back(xs[i]);
    else b.insert(b.end(), xs[i].begin(), xs[i].end());
  }
}
template <class T, class J, class F>
inline auto joinAtIfVector(const vector2d<T>& xs, const J& is, F fn) {
  vector2d<T> a; joinAtIf(xs, is, a, fn);
  return a;
}


template <class T, class J>
void joinAtUntilSize(const vector2d<T>& xs, const J& is, vector2d<T>& a, size_t N) {
  auto fn = [&](const auto& b, const auto& x) { return b.size()<N; };
  joinAtIf(xs, is, a, fn);
}
template <class T, class J>
inline auto joinAtUntilSizeVector(const vector2d<T>& xs, const J& is, size_t N) {
  vector2d<T> a; joinAtUntilSize(xs, is, a, N);
  return a;
}


template <class T, class J>
void joinAt(const vector2d<T>& xs, const J& is, vector<T>& a) {
  for (auto i : is)
    a.insert(a.end(), xs[i].begin(), xs[i].end());
}
template <class T, class J>
inline auto joinAtVector(const vector2d<T>& xs, const J& is) {
  vector<T> a; joinAt(xs, is, a);
  return a;
}


template <class T, class J>
void joinAt2d(const vector2d<T>& xs, const J& ig, vector2d<T>& a) {
  for (const auto& is : ig)
    a.push_back(joinAtVector(xs, is));
}
template <class T, class J>
inline auto joinAt2d(const vector2d<T>& xs, const J& ig) {
  vector2d<T> a; joinAt2d(xs, ig, a);
  return a;
}




// GATHER-VALUES
// -------------

template <class T, class J, class TA>
void gatherValues(const T *x, const J& is, TA *a) {
  size_t j = 0;
  for (auto i : is)
    a[j++] = x[i];
}
template <class T, class J, class TA>
inline void gatherValues(const vector<T>& x, const J& is, vector<TA>& a) {
  gatherValues(x.data(), is, a.data());
}




// SCATTER-VALUES
// --------------

template <class T, class J, class TA>
void scatterValues(const T *x, const J& is, TA *a) {
  size_t j = 0;
  for (auto i : is)
    a[i] = x[j++];
}
template <class T, class J, class TA>
inline void scatterValues(const vector<T>& x, const J& is, vector<TA>& a) {
  scatterValues(x.data(), is, a.data());
}




// COPY-VALUES
// -----------

template <class T, class TA>
size_t copyValues(const T *x, TA *a, size_t N) {
  for (size_t i=0; i<N; ++i)
    a[i] = x[i];
  return N;
}
template <class T, class TA>
inline size_t copyValues(const vector<T>& x, vector<TA>& a) {
  return copyValues(x.data(), a.data(), x.size());
}
template <class T, class TA>
inline size_t copyValues(const vector<T>& x, vector<TA>& a, size_t i, size_t N) {
  return copyValues(x.data()+i, a.data()+i, N);
}




// FILL-VALUE
// ----------

template <class T, class V>
void fillValue(T *a, size_t N, const V& v) {
  fill(a, a+N, v);
}
template <class T, class V>
inline void fillValue(vector<T>& a, const V& v) {
  fill(a.begin(), a.end(), v);
}
template <class T, class V>
inline void fillValue(vector<T>& a, size_t i, size_t N, const V& v) {
  fill(a.begin()+i, a.begin()+i+N, v);
}




// FILL-VALUE-AT
// -------------

template <class T, class J, class V>
void fillValueAt(T *a, const J& is, const V& v) {
  for (auto i : is)
    a[i] = v;
}
template <class T, class J, class V>
inline void fillValueAt(vector<T>& a, const J& is, const V& v) {
  fillValueAt(a.data(), is, v);
}
template <class T, class J, class V>
inline void fillValueAt(vector<T>& a, size_t i, const J& is, const V& v) {
  fillValueAt(a.data()+i, is, v);
}




// SUM-VALUES
// ----------

template <class T, class V=T>
V sumValues(const T *x, size_t N, V a=V()) {
  for (size_t i=0; i<N; ++i)
    a += x[i];
  return a;
}
template <class T, class V=T>
inline V sumValues(const vector<T>& x, V a=V()) {
  return sumValues(x.data(), x.size(), a);
}
template <class T, class V=T>
inline V sumValues(const vector<T>& x, size_t i, size_t N, V a=V()) {
  return sumValues(x.data()+i, N, a);
}




// SUM-ABS-VALUES
// --------------

template <class T, class V=T>
V sumAbsValues(const T *x, size_t N, V a=V()) {
  for (size_t i=0; i<N; ++i)
    a += abs(x[i]);
  return a;
}
template <class T, class V=T>
inline V sumAbsValues(const vector<T>& x, V a=V()) {
  return sumAbsValues(x.data(), x.size(), a);
}
template <class T, class V=T>
inline V sumAbsValues(const vector<T>& x, size_t i, size_t N, V a=V()) {
  return sumAbsValues(x.data()+i, N, a);
}
// NOTE: ADDITIONAL HELPER
template <class T, size_t N, class V=T>
inline V sumAbsValues(const array<T, N>& x, V a=V()) {
  return sumAbsValues(x.data(), N, a);
}



// SUM-SQR-VALUES
// --------------

template <class T, class V=T>
V sumSqrValues(const T *x, size_t N, V a=V()) {
  for (size_t i=0; i<N; ++i)
    a += x[i]*x[i];
  return a;
}
template <class T, class V=T>
inline V sumSqrValues(const vector<T>& x, V a=V()) {
  return sumSqrValues(x.data(), x.size(), a);
}
template <class T, class V=T>
inline V sumSqrValues(const vector<T>& x, size_t i, size_t N, V a=V()) {
  return sumSqrValues(x.data()+i, N, a);
}




// SUM-VALUES-AT
// -------------

template <class T, class J, class V=T>
V sumValuesAt(const T *x, const J& is, V a=V()) {
  for (auto i : is)
    a += x[i];
  return a;
}
template <class T, class J, class V=T>
inline V sumValuesAt(const vector<T>& x, const J& is, V a=V()) {
  return sumValuesAt(x.data(), is, a);
}
template <class T, class J, class V=T>
inline V sumValuesAt(const vector<T>& x, size_t i, const J& is, V a=V()) {
  return sumValuesAt(x.data()+i, is, a);
}




// ADD-VALUE
// ---------

template <class T, class V>
void addValue(T *a, size_t N, const V& v) {
  for (size_t i=0; i<N; ++i)
    a[i] += v;
}
template <class T, class V>
inline void addValue(vector<T>& a, const V& v) {
  addValue(a.data(), a.size(), v);
}
template <class T, class V>
inline void addValue(vector<T>& a, size_t i, size_t N, const V& v) {
  addValue(a.data()+i, N, v);
}




// ADD-VALUE-AT
// ------------

template <class T, class J, class U>
void addValueAt(T *a, const J& is, const U& v) {
  for (auto i : is)
    a[i] += v;
}
template <class T, class J, class U>
inline void addValueAt(vector<T>& a, const J& is, const U& v) {
  addValueAt(a.data(), is, v);
}
template <class T, class J, class U>
inline void addValueAt(vector<T>& a, int i, const J& is, const U& v) {
  addValueAt(a.data()+i, is, v);
}




// MAX-VALUE
// ---------

template <class T, class V=T>
V maxValue(const T *x, size_t N, V a=V()) {
  for (size_t i=0; i<N; ++i)
    a = max(a, x[i]);
  return a;
}
template <class T, class V=T>
inline V maxValue(const vector<T>& x, V a=V()) {
  return maxValue(x.data(), x.size(), a);
}
template <class T, class V=T>
inline V maxValue(const vector<T>& x, size_t i, size_t N, V a=V()) {
  return maxValue(x.data()+i, N, a);
}




// MAX-ABS-VALUE
// -------------

template <class T, class V=T>
V maxAbsValue(const T *x, size_t N, V a=V()) {
  for (size_t i=0; i<N; ++i)
    a = max(a, abs(x[i]));
  return a;
}
template <class T, class V=T>
inline V maxAbsValue(const vector<T>& x, V a=V()) {
  return maxAbsValue(x.data(), x.size(), a);
}
template <class T, class V=T>
inline V maxAbsValue(const vector<T>& x, size_t i, size_t N, V a=V()) {
  return maxAbsValue(x.data()+i, N, a);
}




// MAX-VALUE-AT
// ------------

template <class T, class J, class V=T>
V maxAt(const T *x, const J& is, V a=V()) {
  for (auto i : is)
    a = max(a, x[i]);
  return a;
}
template <class T, class J, class V=T>
inline V maxAt(const vector<T>& x, const J& is, V a=V()) {
  return maxAt(x.data(), is, a);
}
template <class T, class J, class V=T>
inline V maxAt(const vector<T>& x, size_t i, const J& is, V a=V()) {
  return maxAt(x.data()+i, is, a);
}




// CONSTRAIN-MAX
// -------------

template <class T, class V>
void constrainMax(T *a, size_t N, const V& v) {
  for (size_t i=0; i<N; ++i)
    a[i] = max(a[i], v);
}
template <class T, class V>
inline void constrainMax(vector<T>& a, const V& v) {
  constrainMax(a.data(), a.size(), v);
}
template <class T, class V>
inline void constrainMax(vector<T>& a, size_t i, size_t N, const V& v) {
  constrainMax(a.data()+i, N, v);
}




// CONSTRAIN-MAX-AT
// ----------------

template <class T, class J, class V>
void constrainMaxAt(T *a, const J& is, const V& v) {
  for (auto i : is)
    a[i] = max(a[i], v);
}
template <class T, class J, class V>
inline void constrainMaxAt(vector<T>& a, const J& is, const V& v) {
  constrainMaxAt(a.data(), is, v);
}
template <class T, class J, class V>
inline void constrainMaxAt(vector<T>& a, size_t i, const J& is, const V& v) {
  constrainMaxAt(a.data()+i, is, v);
}




// MIN-VALUE
// ---------

template <class T, class V=T>
V minValue(const T *x, size_t N, V a=V()) {
  for (size_t i=0; i<N; ++i)
    a = min(a, x[i]);
  return a;
}
template <class T, class V=T>
inline V minValue(const vector<T>& x, V a=V()) {
  return minValue(x.data(), x.size(), a);
}
template <class T, class V=T>
inline V minValue(const vector<T>& x, size_t i, size_t N, V a=V()) {
  return minValue(x.data()+i, N, a);
}




// MIN-ABS-VALUE
// -------------

template <class T, class V=T>
V minAbsValue(const T *x, size_t N, V a=V()) {
  for (size_t i=0; i<N; ++i)
    a = min(a, abs(x[i]));
  return a;
}
template <class T, class V=T>
inline V minAbsValue(const vector<T>& x, V a=V()) {
  return minAbsValue(x.data(), x.size(), a);
}
template <class T, class V=T>
inline V minAbsValue(const vector<T>& x, size_t i, size_t N, V a=V()) {
  return minAbsValue(x.data()+i, N, a);
}




// MIN-VALUE-AT
// ------------

template <class T, class J, class V=T>
V minValueAt(const T *x, const J& is, V a=V()) {
  for (auto i : is)
    a = min(a, x[i]);
  return a;
}
template <class T, class J, class V=T>
inline V minValueAt(const vector<T>& x, const J& is, V a=V()) {
  return minValueAt(x.data(), is, a);
}
template <class T, class J, class V=T>
inline V minValueAt(const vector<T>& x, size_t i, const J& is, V a=V()) {
  return minValueAt(x.data()+i, is, a);
}




// CONSTRAIN-MIN
// -------------

template <class T, class V>
void constrainMin(T *a, size_t N, const V& v) {
  for (size_t i=0; i<N; ++i)
    a[i] = min(a[i], v);
}
template <class T, class V>
inline void constrainMin(vector<T>& a, const V& v) {
  constrainMin(a.data(), a.size(), v);
}
template <class T, class V>
inline void constrainMin(vector<T>& a, size_t i, size_t N, const V& v) {
  constrainMin(a.data()+i, N, v);
}




// CONSTRAIN-MIN-AT
// ----------------

template <class T, class J, class V>
void constrainMinAt(T *a, const J& is, const V& v) {
  for (auto i : is)
    a[i] = min(a[i], v);
}
template <class T, class J, class V>
inline void constrainMinAt(vector<T>& a, const J& is, const V& v) {
  constrainMinAt(a.data(), is, v);
}
template <class T, class J, class V>
inline void constrainMinAt(vector<T>& a, size_t i, const J& is, const V& v) {
  constrainMinAt(a.data()+i, is, v);
}




// L1-NORM
// -------

template <class TX, class TY, class V=TX>
V l1Norm(const TX *x, const TY *y, size_t N, const V& a=V()) {
  for (size_t i=0; i<N; i++)
    a += abs(x[i] - y[i]);
  return a;
}
template <class TX, class TY, class V=TX>
inline V l1Norm(const vector<TX>& x, const vector<TY>& y, const V& a=V()) {
  return l1Norm(x.data(), y.data(), x.size(), a);
}
template <class TX, class TY, class V=TX>
inline V l1Norm(const vector<TX>& x, const vector<TY>& y, size_t i, size_t N, const V& a=V()) {
  return l1Norm(x.data()+i, y.data()+i, N, a);
}




// L2-NORM
// -------

template <class TX, class TY, class V=TX>
V l2Norm(const TX *x, const TY *y, size_t N, const V& a=V()) {
  for (size_t i=0; i<N; i++)
    a += (x[i] - y[i]) * (x[i] - y[i]);
  return sqrt(a);
}
template <class TX, class TY, class V=TX>
inline V l2Norm(const vector<TX>& x, const vector<TY>& y, const V& a=V()) {
  return l2Norm(x.data(), y.data(), x.size(), a);
}
template <class TX, class TY, class V=TX>
inline V l2Norm(const vector<TX>& x, const vector<TY>& y, size_t i, size_t N, const V& a=V()) {
  return l2Norm(x.data()+i, y.data()+i, N, a);
}




// LI-NORM (INFINITY)
// ------------------

template <class TX, class TY, class V=TX>
V liNorm(const TX *x, const TY *y, size_t N, const V& a=V()) {
  for (size_t i=0; i<N; i++)
    a = max(a, abs(x[i] - y[i]));
  return a;
}
template <class TX, class TY, class V=TX>
inline V liNorm(const vector<TX>& x, const vector<TY>& y, const V& a=V()) {
  return liNorm(x.data(), y.data(), x.size(), a);
}
template <class TX, class TY, class V=TX>
inline V liNorm(const vector<TX>& x, const vector<TY>& y, size_t i, size_t N, const V& a=V()) {
  return liNorm(x.data()+i, y.data()+i, N, a);
}




// MULTIPLY-VALUES
// ---------------

template <class TX, class TY, class TA>
void multiplyValues(const TX *x, const TY *y, TA *a, size_t N) {
  for (size_t i=0; i<N; i++)
    a[i] = x[i] * y[i];
}
template <class TX, class TY, class TA>
inline void multiplyValues(const vector<TX>& x, const vector<TY>& y, vector<TA>& a) {
  multiplyValues(x.data(), y.data(), a.data(), x.size());
}
template <class TX, class TY, class TA>
inline void multiplyValues(const vector<TX>& x, const vector<TY>& y, vector<TA>& a, size_t i, size_t N) {
  multiplyValues(x.data()+i, y.data()+i, a.data()+i, N);
}




// MULTIPLY-VALUE
// --------------

template <class T, class TA, class V>
void multiplyValue(const T *x, const TA *a, size_t N, const V& v) {
  for (size_t i=0; i<N; i++)
    a[i] = TA(x[i] * v);
}
template <class T, class TA, class V>
inline void multiplyValue(const vector<T>& x, vector<TA>& a, const V& v) {
  multiplyValue(x.data(), a.data(), x.size(), v);
}
template <class T, class TA, class V>
inline void multiplyValue(const vector<T>& x, vector<TA>& a, size_t i, size_t N, const V& v) {
  multiplyValue(x.data()+i, a.data()+i, N, v);
}
