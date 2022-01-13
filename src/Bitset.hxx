#pragma once
#include <utility>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include "_main.hxx"

using std::pair;
using std::vector;
using std::out_of_range;
using std::iter_swap;
using std::find_if;
using std::lower_bound;
using std::sort;
using std::inplace_merge;




// BITSET-*
// --------
// Helps create bitsets.

#ifndef BITSET_TYPES
#define BITSET_TYPES(K, V, data) \
  using key_type   = K; \
  using value_type = V; \
  using pair_type  = pair<K, V>; \
  using iterator       = decltype(data.begin()); \
  using const_iterator = decltype(data.cbegin());
#endif


#ifndef BITSET_ITERATOR
#define BITSET_ITERATOR(K, V, data) \
  ITERABLE_ITERATOR(inline, noexcept, data.begin(), data.end())

#define BITSET_CITERATOR(K, V, data) \
  ITERABLE_CITERATOR(inline, noexcept, data.begin(), data.end())
#endif


#ifndef BITSET_SIZE
#define BITSET_SIZE(K, V, data)  \
  inline size_t size() const noexcept { return data.size(); }

#define BITSET_EMPTY(K, V) \
  inline bool empty()  const noexcept { return size() == 0; }
#endif


#ifndef BITSET_FIND
#define BITSET_FIND(K, V, data) \
  inline auto find(const K& k)        noexcept { return locateMatch(k); } \
  inline auto cfind(const K& k) const noexcept { return locateMatch(k); } \
  inline auto find(const K& k)  const noexcept { return cfind(k); }

#define BITSET_ENTRIES(K, V, data) \
  /* dont change the keys! */ \
  inline auto values() noexcept { \
    return staticTransformIterable(data, PairSecond<K, V>()); \
  } \
  inline auto pairs() noexcept { \
    return iterable(data); \
  } \
  inline auto ckeys() const noexcept { \
    return staticTransformIterable(data, ConstPairFirst<K, V>()); \
  } \
  inline auto cvalues() const noexcept { \
    return staticTransformIterable(data, ConstPairSecond<K, V>()); \
  } \
  inline auto cpairs() const noexcept { \
    return iterable(data); \
  } \
  inline auto keys()   const noexcept { return ckeys(); } \
  inline auto values() const noexcept { return cvalues(); } \
  inline auto pairs()  const noexcept { return cpairs(); }

#define BITSET_FOREACH(K, V, data) \
  /* dont change the keys! */ \
  template <class F> \
  inline void forEachValue(F fn) { \
    for (pair<K, V>& p : data) fn(p.second); \
  } \
  template <class F> \
  inline void forEachPair(F fn) { \
    for (pair<K, V>& p : data) fn(p); \
  } \
  template <class F> \
  inline void forEach(F fn) { \
    for (pair<K, V>& p : data) fn(p.first, p.second); \
  } \
  template <class F> \
  inline void cforEachKey(F fn) const { \
    for (const pair<K, V>& p : data) fn(p.first); \
  } \
  template <class F> \
  inline void cforEachValue(F fn) const { \
    for (const pair<K, V>& p : data) fn(p.second); \
  } \
  template <class F> \
  inline void cforEachPair(F fn) const { \
    for (const pair<K, V>& p : data) fn(p); \
  } \
  template <class F> \
  inline void cforEach(F fn) const { \
    for (const pair<K, V>& p : data) fn(p.first, p.second); \
  } \
  template <class F> \
  inline void forEachKey(F fn)   const { cforEachKey(fn); } \
  template <class F> \
  inline void forEachValue(F fn) const { cforEachValue(fn); } \
  template <class F> \
  inline void forEachPair(F fn)  const { cforEachPair(fn); } \
  template <class F> \
  inline void forEach(F fn)      const { cforEach(fn); }
#endif


#ifndef BITSET_HAS
#define BITSET_HAS(K, V) \
  inline bool has(const K& k) const noexcept { \
    return locateMatch(k) != end(); \
  }

#define BITSET_GET(K, V) \
  inline V get(const K& k) const noexcept { \
    auto it = locateMatch(k); \
    return it == end()? V() : (*it).second; \
  }

#define BITSET_SET(K, V) \
  inline bool set(const K& k, const V& v) noexcept { \
    auto it = locateMatch(k); \
    if (it == end()) return false; \
    (*it).second = v; \
    return true; \
  }

#define BITSET_SUBSCRIPT(K, V) \
  inline V& operator[](const K& k) noexcept { \
    auto it = locateMatch(k); \
    return (*it).second; \
  } \
  inline const V& operator[](const K& k) const noexcept { \
    auto it = locateMatch(k); \
    return (*it).second; \
  }

#define BITSET_AT(K, V) \
  inline V& at(const K& k) { \
    auto it = locateMatch(k); \
    if (it == end()) throw out_of_range("bitset key not present"); \
    return (*it).second; \
  } \
  inline const V& at(const K& k) const { \
    auto it = locateMatch(k); \
    if (it == end()) throw out_of_range("bitset key not present"); \
    return (*it).second; \
  }
#endif




// BITSET (UNSORTED)
// -----------------
// An integer set that constantly checks duplicates.
// It maintains integers in insertion order.

#define BITSET_UNSORTED_LOCATE(K, V, f0, f1) \
  f0 auto locateMatch(const K& k) f1 { \
    auto fe = [&](const pair<K, V>& p) { return p.first == k; }; \
    return find_if(begin(), end(), fe); \
  }


template <class K=int, class V=NONE>
class BitsetUnsorted {
  // Data.
  protected:
  vector<pair<K, V>> data;

  // Types.
  public:
  BITSET_TYPES(K, V, data)


  // Iterator operations.
  public:
  BITSET_ITERATOR(K, V, data)
  BITSET_CITERATOR(K, V, data)


  // Size operations.
  public:
  BITSET_SIZE(K, V, data)
  BITSET_EMPTY(K, V)


  // Search operations.
  protected:
  BITSET_UNSORTED_LOCATE(K, V, inline, noexcept)
  BITSET_UNSORTED_LOCATE(K, V, inline, const noexcept)
  public:
  BITSET_FIND(K, V, data)


  // Access operations.
  public:
  BITSET_ENTRIES(K, V, data)
  BITSET_FOREACH(K, V, data)
  BITSET_HAS(K, V)
  BITSET_GET(K, V)
  BITSET_SET(K, V)
  BITSET_SUBSCRIPT(K, V)
  BITSET_AT(K, V)


  // Update operations.
  public:
  inline bool correct()  { return false; }
  inline bool optimize() { return correct(); }

  inline bool clear() noexcept {
    if (empty()) return false;
    data.clear();
    return true;
  }

  inline bool add(const K& k, const V& v=V()) {
    if (has(k)) return false;
    data.push_back({k, v});
    return true;
  }

  inline bool remove(const K& k) {
    auto it = locateMatch(k);
    if (it == end()) return false;
    iter_swap(it, end()-1);
    data.pop_back();
    return true;
  }
};




// BITSET (SORTED)
// ---------------
// An integer set that constantly checks duplicates.
// It maintains integers in ascending value order.

#define BITSET_SORTED_LOCATE(K, V, f0, f1) \
  f0 auto locateSpot(const K& k) f1 { \
    auto fl = [](const pair<K, V>& p, const K& k) { return p.first < k; }; \
    return lower_bound(begin(), end(), k, fl); \
  } \
  f0 auto locateMatch(const K& k) f1 { \
    auto it = locateSpot(k); \
    return it == end() || (*it).first != k? end() : it; \
  }


template <class K=int, class V=NONE>
class BitsetSorted {
  // Data.
  protected:
  vector<pair<K, V>> data;

  // Types.
  public:
  BITSET_TYPES(K, V, data)


  // Iterator operations.
  public:
  BITSET_ITERATOR(K, V, data)
  BITSET_CITERATOR(K, V, data)


  // Size operations.
  public:
  BITSET_SIZE(K, V, data)
  BITSET_EMPTY(K, V)


  // Search operations.
  protected:
  BITSET_SORTED_LOCATE(K, V, inline, noexcept)
  BITSET_SORTED_LOCATE(K, V, inline, const noexcept)
  public:
  BITSET_FIND(K, V, data)


  // Access operations.
  public:
  BITSET_ENTRIES(K, V, data)
  BITSET_FOREACH(K, V, data)
  BITSET_HAS(K, V)
  BITSET_GET(K, V)
  BITSET_SET(K, V)
  BITSET_SUBSCRIPT(K, V)
  BITSET_AT(K, V)


  // Update operations.
  public:
  inline bool correct()  { return false; }
  inline bool optimize() { return correct(); }

  inline bool clear() noexcept {
    if (empty()) return false;
    data.clear();
    return true;
  }

  inline bool add(const K& k, const V& v=V()) {
    auto it = locateSpot(k);
    if (it != end() && (*it).first == k) return false;
    data.insert(it, {k, v});
    return true;
  }

  inline bool remove(const K& k) {
    auto it = locateMatch(k);
    if (it == end()) return false;
    data.erase(it);
    return true;
  }
};




// BITSET (PARTIALLY-SORTED)
// -------------------------
// An integer set that constantly checks duplicates.
// It maintains a portion of integers in ascending value order.

#define BITSET_PSORTED_LOCATE(K, V, f0, f1) \
  f0 auto locateMatchSorted(const K& k) f1 { \
    auto fl = [](const pair<K, V>& p, const K& k) { return p.first < k; }; \
    auto it = lower_bound(begin(), middle(), k, fl); \
    return it == middle() || (*it).first != k? end() : it; \
  } \
  f0 auto locateMatchUnsorted(const K& k) f1 { \
    auto fe = [&](const pair<K, V>& p) { return p.first == k; }; \
    return find_if(middle(), end(), fe); \
  } \
  f0 auto locateMatch(const K& k) f1 { \
    auto it = locateMatchSorted(k); \
    return it != end()? it : locateMatchUnsorted(k); \
  }


template <class K=int, class V=NONE, size_t LIMIT=64>
class BitsetPsorted {
  // Data.
  protected:
  vector<pair<K, V>> data;
  size_t sorted = 0;

  // Types.
  public:
  BITSET_TYPES(K, V, data)


  // Iterator operations.
  public:
  BITSET_ITERATOR(K, V, data)
  BITSET_CITERATOR(K, V, data)
  protected:
  ITERABLE_NAMES(inline, noexcept, middle, begin() + sorted)


  // Size operations.
  public:
  BITSET_SIZE(K, V, data)
  BITSET_EMPTY(K, V)
  protected:
  inline size_t unsorted() const noexcept { return size() - sorted; }


  // Search operations.
  protected:
  BITSET_PSORTED_LOCATE(K, V, inline, noexcept)
  BITSET_PSORTED_LOCATE(K, V, inline, const noexcept)
  public:
  BITSET_FIND(K, V, data)


  // Ordering opertions.
  protected:
  inline void mergePartitions() {
    auto fl = [](const pair<K, V>& p, const pair<K, V>& q) { return p.first < q.first; };
    sort(middle(), end(), fl);
    inplace_merge(begin(), middle(), end(), fl);
    sorted = size();
  }


  // Access operations.
  public:
  BITSET_ENTRIES(K, V, data)
  BITSET_FOREACH(K, V, data)
  BITSET_HAS(K, V)
  BITSET_GET(K, V)
  BITSET_SET(K, V)
  BITSET_SUBSCRIPT(K, V)
  BITSET_AT(K, V)


  // Update operations.
  public:
  inline bool correct()  { return false; }

  inline bool optimize() {
    bool c = correct();
    if (unsorted() == 0) return c;
    mergePartitions();
    return true;
  }

  inline bool clear() noexcept {
    if (empty()) return false;
    data.clear();
    sorted = 0;
    return true;
  }

  inline bool add(const K& k, const V& v=V()) {
    auto it = locateMatch(k);
    if (it != end()) return false;
    data.push_back({k, v});
    if (unsorted() <= LIMIT) mergePartitions();
    return true;
  }

  inline bool remove(const K& k) {
    auto it = locateMatch(k);
    if (it == end()) return false;
    if (it < middle()) --sorted;
    data.erase(it);
    return true;
  }
};




// UNCHECKED-BITSET (SORTED)
// -------------------------
// An integer set that does not check duplicates.
// Removing duplicates can be done manually, with correct().
// It maintains integers in ascending value order (after correct()).

#define UBITSET_SORTED_LOCATE(K, V, f0, f1) \
  BITSET_PSORTED_LOCATE(K, V, f0, f1)


template <class K=int, class V=NONE>
class UBitsetSorted {
  // Data.
  protected:
  vector<pair<K, V>> data;
  size_t sorted = 0;

  // Types.
  public:
  BITSET_TYPES(K, V, data)


  // Iterator operations.
  public:
  BITSET_ITERATOR(K, V, data)
  BITSET_CITERATOR(K, V, data)
  protected:
  ITERABLE_NAMES(inline, noexcept, middle, begin() + sorted)


  // Size operations.
  public:
  BITSET_SIZE(K, V, data)
  BITSET_EMPTY(K, V)


  // Search operations.
  protected:
  UBITSET_SORTED_LOCATE(K, V, inline, noexcept)
  UBITSET_SORTED_LOCATE(K, V, inline, const noexcept)
  public:
  BITSET_FIND(K, V, data)


  // Access operations.
  public:
  BITSET_ENTRIES(K, V, data)
  BITSET_FOREACH(K, V, data)
  BITSET_HAS(K, V)
  BITSET_GET(K, V)
  BITSET_SET(K, V)
  BITSET_SUBSCRIPT(K, V)
  BITSET_AT(K, V)


  // Update operations.
  public:
  inline bool correct() {
    auto fl = [](const pair<K, V>& a, const pair<K, V>& b) { return a.first <  b.first; };
    auto fe = [](const pair<K, V>& a, const pair<K, V>& b) { return a.first == b.first; };
    if (sorted == size()) return false;
    if (sorted <= size()/2) sort(begin(), end(), fl);
    else { sort(middle(), end(), fl); inplace_merge(begin(), middle(), end(), fl); }
    auto it = unique(begin(), end(), fe);  // TODO: optimize
    data.resize(it - begin());
    sorted = size();
    return true;
  }

  inline bool optimize() { return correct(); }

  inline bool clear() noexcept {
    if (empty()) return false;
    data.clear();
    sorted = 0;
    return true;
  }

  inline bool add(const K& k, const V& v=V()) {
    data.push_back({k, v});
    return true;
  }

  inline bool remove(const K& k) {
    auto it = locateMatch(k);
    if (it == end()) return false;
    if (it < middle()) --sorted;
    data.erase(it);
    return true;
  }
};
