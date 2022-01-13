#pragma once




// COPY
// ----

template <class H, class G, class FV, class FE>
void copyTo(H& a, const G& x, FV fv, FE fe) {
  x.forEachVertex([&](auto u, auto d) { if (fv(u)) a.addVertex(u, d); });
  x.forEachVertex([&](auto u, auto _) {
    if (fv(u)) x.forEachEdge(u, [&](auto v, auto w) { if (fv(v) && fe(u, v)) a.addEdge(u, v, w); });
  });
}

template <class H, class G, class FV>
void copyTo(H& a, const G& x, FV fv) {
  copyTo(a, x, fv, [](int u, int v) { return true; });
}

template <class H, class G>
void copyTo(H& a, const G& x) {
  copyTo(a, x, [](int u) { return true; });
}

template <class G, class FV, class FE>
auto copy(const G& x, FV fv, FE fe) {
  G a; copyTo(a, x, fv, fe);
  return a;
}

template <class G, class FV>
auto copy(const G& x, FV fv) {
  G a; copyTo(a, x, fv);
  return a;
}

template <class G>
auto copy(const G& x) {
  G a; copyTo(a, x);
  return a;
}
