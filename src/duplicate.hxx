#pragma once




// DUPLICATE
// ---------

template <class H, class G, class FV, class FE>
void duplicateTo(H& a, const G& x, FV fv, FE fe, bool unq=false) {
  x.forEachVertex([&](auto u, auto d) { if (fv(u)) a.addVertex(u, d); });
  x.forEachVertex([&](auto u, auto d) {
    if (fv(u)) x.forEachEdge(u, [&](auto v, auto w) {
      if (fv(v) && fe(u, v)) a.addEdge(u, v, w);
    });
  });
  a.correct(unq);
}
template <class H, class G, class FV>
inline void duplicateTo(H& a, const G& x, FV fv, bool unq=false) {
  duplicateTo(a, x, fv, [](auto u, auto v) { return true; }, unq);
}
template <class H, class G>
inline void duplicateTo(H& a, const G& x, bool unq=false) {
  duplicateTo(a, x, [](auto u) { return true; }, unq);
}
template <class G, class FV, class FE>
inline auto duplicate(const G& x, FV fv, FE fe) {
  G a; duplicateTo(a, x, fv, fe, true);
  return a;
}
template <class G, class FV>
inline auto duplicate(const G& x, FV fv) {
  G a; duplicateTo(a, x, fv, true);
  return a;
}
template <class G>
inline auto duplicate(const G& x) {
  G a; duplicateTo(a, x, true);
  return a;
}
