#pragma once
#include <utility>
#include <vector>
#include <algorithm>
#include "_main.hxx"

using std::pair;
using std::vector;
using std::iter_swap;
using std::find_if;
using std::lower_bound;




// BITSET (UNSORTED)
// -----------------
// An integer set that constantly checks duplicates.
// It maintains integers in insertion order.

template <class K=int, class V=NONE>
class BitsetUnsorted {
  // Data.
  private:
  vector<pair<K, V>> pairs;


  // Types.
  public:
  using key_type   = K;
  using value_type = V;


  // Iterators.
  public:
  auto cbegin()  const { return pairs.cbegin(); }
  auto cend()    const { return pairs.cend(); }
  auto begin()   const { return pairs.begin(); }
  auto end()     const { return pairs.end(); }
  auto begin()         { return pairs.begin(); }
  auto end()           { return pairs.end(); }


  // Iterables.
  public:
  auto entries() const { return transformIter(pairs, [](const auto& e) { return e; }); }
  auto keys()    const { return transformIter(pairs, [](const auto& e) { return e.first; }); }
  auto values()  const { return transformIter(pairs, [](const auto& e) { return e.second; }); }


  // Size.
  size_t size()  const { return pairs.size(); }


  // Search.
  private:
  auto clookup(K k) const {
    auto fe = [&](const auto& e) { return e.first == k; };
    return find_if(pairs.begin(), pairs.end(), fe);
  }

  auto lookup(K k) const { return clookup(k); }
  auto lookup(K k)       { return begin() + (clookup() - cbegin()); }


  // Read operations.
  public:
  bool has(K k) const { return clookup(k) != cend(); }
  V    get(K k) const { auto it = clookup(k); return it != cend()? (*it).second : V(); }


  // Write operations.
  public:
  void clear() {
    pairs.clear();
  }

  void correct() {}

  void set(K k, V v) {
    auto it = lookup(k);
    if (it == end()) return;
    (*it).second = v;
  }

  void add(K k, V v=V()) {
    if (has(k)) return;
    ids.push_back({k, v});
  }

  void remove(K k) {
    auto it = lookup(k);
    if (it == end()) return;
    iter_swap(it, end()-1);
    pairs.pop_back();
  }
};




// BITSET (SORTED)
// ---------------
// An integer set that constantly checks duplicates.
// It maintains integers in ascending value order.

template <class K=int, class V=NONE>
class BitsetSorted {
  // Data.
  private:
  vector<pair<K, V>> pairs;


  // Types.
  public:
  using key_type   = K;
  using value_type = V;


  // Iterators.
  public:
  auto cbegin()  const { return pairs.cbegin(); }
  auto cend()    const { return pairs.cend(); }
  auto begin()   const { return pairs.begin(); }
  auto end()     const { return pairs.end(); }
  auto begin()         { return pairs.begin(); }
  auto end()           { return pairs.end(); }


  // Iterables.
  public:
  auto entries() const { return transformIter(pairs, [](const auto& e) { return e; }); }
  auto keys()    const { return transformIter(pairs, [](const auto& e) { return e.first; }); }
  auto values()  const { return transformIter(pairs, [](const auto& e) { return e.second; }); }


  // Size.
  size_t size()  const { return pairs.size(); }


  // Search.
  private:
  auto cwhere(K k) const {
    auto fl = [](const auto& e, K k) { return e.first < k; };
    return lower_bound(cbegin(), cend(), k, fl);
  }

  auto clookup(K k) const {
    auto it = cwhere(k);
    return it != cend() && (*it).first == k? it : cend();
  }

  auto where(K k)  const { return cwhere(k); }
  auto where(K k)        { return begin() + (cwhere() - cbegin()); }
  auto lookup(K k) const { return clookup(k); }
  auto lookup(K k)       { return begin() + (clookup() - cbegin()); }


  // Read operations.
  public:
  bool has(K k) const { return clookup(k) != cend(); }
  V    get(K k) const { auto it = clookup(k); return it != cend()? (*it).second : V(); }


  // Write operations.
  public:
  void clear() {
    pairs.clear();
  }

  void correct() {}

  void set(K k, V v) {
    auto it = lookup(k);
    if (it == end()) return;
    (*it).second = v;
  }

  void add(K k, V v=V()) {
    auto it = cwhere(k);
    if (it != cend() && (*it).first == k) return;
    pairs.insert(it, {k, v});
  }

  void remove(K k) {
    auto it = clookup(k);
    if (it == cend()) return;
    pairs.erase(it);
  }
};




// BITSET (PARTIALLY-SORTED)
// -------------------------
// An integer set that does not check duplicates.
// Removing duplicates can be done manually, with correct().
// It maintains integers in ascending value order.

template <class K=int, class V=NONE>
class BitsetPsorted {
  // Data.
  private:
  vector<pair<K, V>> pairs;
  size_t sorted;


  // Types.
  public:
  using key_type   = K;
  using value_type = V;


  // Iterators.
  public:
  auto cbegin()  const { return pairs.cbegin(); }
  auto cend()    const { return pairs.cend(); }
  auto begin()   const { return pairs.cbegin(); }
  auto end()     const { return pairs.cend(); }
  auto begin()         { return pairs.begin(); }
  auto end()           { return pairs.end(); }
  private:
  auto cmiddle() const { return cbegin() + sorted; }
  auto middle()  const { return cbegin() + sorted; }
  auto middle()        { return begin()  + sorted; }


  // Iterables.
  public:
  auto entries() const { return transformIter(pairs, [](const auto& e) { return e; }); }
  auto keys()    const { return transformIter(pairs, [](const auto& e) { return e.first; }); }
  auto values()  const { return transformIter(pairs, [](const auto& e) { return e.second; }); }


  // Size.
  size_t size()  const { return pairs.size(); }


  // Search.
  private:
  auto clookup(K k) const {
    auto fl =  [](const auto& e, K k) { return e.first <  k; };
    auto fe = [&](const auto& e)        { return e.first == k; };
    auto it = lower_bound(cbegin(), cmiddle(), k, fl);
    if (it != cmiddle() && (*it).first == k) return it;
    return sorted == size()? cend() : find_if(cmiddle(), cend(), fe);
  }

  auto lookup(K k) const { return clookup(k); }
  auto lookup(K k)       { return begin() + (clookup() - cbegin()); }


  // Read operations.
  public:
  bool has(K k) const { return clookup(k) != cend(); }
  V    get(K k) const { auto it = clookup(k); return it != cend()? (*it).second : V(); }


  // Write operations.
  public:
  void clear() {
    pairs.clear();
    sorted = 0;
  }

  void correct() {
    auto fe = [](const auto& ea, const auto& eb) { return ea.first == eb.first; };
    if (sorted == size())   return;
    if (sorted <= size()/2) sort(begin(), end());
    else { sort(middle(), end()); inplace_merge(begin(), middle(), end()); }
    auto it = unique(begin(), end(), fe);  // TODO: optimize
    pairs.resize(it - cbegin());
    sorted = size();
  }

  void set(K k, V v) {
    auto it = lookup(k);
    if (it == end()) return;
    (*it).second = v;
  }

  void add(K k, V v=V()) {
    pairs.push_back({k, v});
  }

  void remove(K k) {
    auto it = clookup(k);
    if (it == cend()) return;
    if (it <  cmiddle()) --sorted;
    pairs.erase(it);
  }
};
