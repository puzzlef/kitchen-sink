#pragma once
#include <vector>
#include <chrono>
#include "_main.hxx"
#include "vertices.hxx"
#include "dfs.hxx"
#include "topologicalSort.hxx"

using std::vector;
using std::chrono::high_resolution_clock;



// COMPONENTS
// ----------
// Finds Strongly Connected Components (SCC) using Kosaraju's algorithm.

template <class G, class H>
auto components(const G& x, const H& xt) {
  vector2d<int> a;
  vector<int> vs;
  // original dfs
  auto vis = createContainer(x, bool());
  for (int u : x.vertices())
    if (!vis[u]) dfsEndLoop(vs, vis, x, u);
  // transpose dfs
  fill(vis, false);
  while (!vs.empty()) {
    int u = vs.back(); vs.pop_back();
    if (vis[u]) continue;
    a.push_back(vector<int>());
    dfsLoop(a.back(), vis, xt, u);
  }
  return a;
}




// COMPONENTS-IDS
// --------------
// Get component id of each vertex.

template <class G>
auto componentIds(const G& x, const vector2d<int>& cs) {
  auto a = createContainer(x, int()); int i = 0;
  for (const auto& c : cs) {
    for (int u : c)
      a[u] = i;
    i++;
  }
  return a;
}




// BLOCKGRAPH
// ----------

template <class H, class G>
void blockgraph(H& a, const G& x, const vector2d<int>& cs) {
  auto t0 = high_resolution_clock::now();
  auto c = componentIds(x, cs);
  auto t1 = high_resolution_clock::now();
  for (int u : x.vertices()) {
    a.addVertex(c[u]);
    for (int v : x.edges(u))
      if (c[u] != c[v]) a.addEdge(c[u], c[v]);
  }
  auto t2 = high_resolution_clock::now();
  printf("[%09.3f ms] auto c = componentIds(x, cs);\n", durationMilliseconds(t0, t1));
  printf("[%09.3f ms] ...\n", durationMilliseconds(t1, t2));
}

template <class G>
auto blockgraph(const G& x, const vector2d<int>& cs) {
  G a; blockgraph(a, x, cs);
  return a;
}




// SORTED-COMPONENTS
// -----------------

template <class G>
auto sortedComponents(const G& x, vector2d<int> cs) {
  auto b = blockgraph(x, cs);
  auto bks = topologicalSort(b);
  reorder(cs, bks);
  return cs;
}

template <class G, class H>
auto sortedComponents(const G& x, const H& xt) {
  auto cs = components(x, xt);
  return sortedComponents(x, cs);
}




// COMPONENTS-EQUAL
// ----------------

template <class G>
bool componentsEqual(const G& x, const vector<int>& xc, const G& y, const vector<int>& yc) {
  if (xc != yc) return false;
  for (int i=0, I=xc.size(); i<I; i++)
    if (!verticesEqual(x, xc[i], y, yc[i])) return false;
  return true;
}

template <class G, class H>
bool componentsEqual(const G& x, const H& xt, const vector<int>& xc, const G& y, const H& yt, const vector<int>& yc) {
  return componentsEqual(x, xc, y, yc) && componentsEqual(xt, xc, yt, yc);
}




// COMPONENTS-HASH
// ---------------

auto componentsHash(const vector2d<int>& cs) {
  vector<size_t> a;
  for (const auto& c : cs)
    a.push_back(hashValue(c));
  return a;
}
