#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include <cstdio>
#include <iostream>
#include <utility>
#include "src/main.hxx"

using namespace std;




template <class G, class T>
void printRow(float t, const G& x, const PagerankResult<T>& a, const PagerankResult<T>& b, const char *tec) {
  auto e = l1Norm(b.ranks, a.ranks);
  int repeat = 5; float tp = t - repeat*b.time;
  print(x); printf(" [%09.3f ms + %09.3f ms; %03d iters.] [%.4e err.] %s\n", tp, b.time, b.iterations, e, tec);
}

void runPagerankBatch(const string& data, int repeat, int skip, int batch) {
  using T = float;
  enum NormFunction { L0=0, L1=1, L2=2, Li=3 };
  vector<T> r0, s0, r1, s1;
  vector<T> *init = nullptr;
  PagerankOptions<T> o = {repeat, Li, true};
  PagerankResult<T> a0, b0, c0, e0, f0;
  PagerankResult<T> b1, c1, d1, b2, c2, d2, b3, c3, d3;
  PagerankResult<T> b4, c4, d4, b5, c5, d5, b6, c6, d6;
  PagerankResult<T> e1, f1, g1, e2, f2, g2, e3, f3, g3;
  PagerankResult<T> e4, f4, g4, e5, f5, g5, e6, f6, g6;

  DiGraph<> xo;
  stringstream s(data);
  while (true) {
    // Skip some edges (to speed up execution)
    if (skip>0 && !readSnapTemporal(xo, s, skip)) break;
    auto x  = selfLoop(xo, [&](int u) { return isDeadEnd(xo, u); });
    auto xt = transposeWithDegree(x);
    auto ksOld = vertices(x);
    a0 = pagerankNvgraph(x, xt, init, o);
    auto r0 = a0.ranks;

    // Read edges for this batch.
    auto yo = copy(xo);
    if (!readSnapTemporal(yo, s, batch)) break;
    auto y  = selfLoop(yo, [&](int u) { return isDeadEnd(yo, u); });
    auto yt = transposeWithDegree(y);
    auto ks = vertices(y);
    vector<T> s0(y.span());
    int X = ksOld.size();
    int Y = ks.size();

    // INSERTIONS:
    // Adjust ranks for insertions.
    adjustRanks(s0, r0, ksOld, ks, 0.0f, float(X)/(Y+1), 1.0f/(Y+1));

    // Find nvGraph-based pagerank.
    float tb0 = measureDuration([&]() { b0 = pagerankNvgraph(y, yt, init, o); });
    printRow(tb0, y, b0, b0, "I:pagerankNvgraph (static)");
    float tc0 = measureDuration([&]() { c0 = pagerankNvgraph(y, yt, &s0, o); });
    printRow(tc0, y, b0, c0, "I:pagerankNvgraph (incremental)");

    // Find sequential Monolithic pagerank.
    float tb1 = measureDuration([&]() { b1 = pagerankMonolithicSeq(y, yt, init, o); });
    printRow(tb1, y, b0, b1, "I:pagerankMonolithicSeq (static)");
    float tc1 = measureDuration([&]() { c1 = pagerankMonolithicSeq(y, yt, &s0, o); });
    printRow(tc1, y, b0, c1, "I:pagerankMonolithicSeq (incremental)");
    float td1 = measureDuration([&]() { d1 = pagerankMonolithicSeqDynamic(x, xt, y, yt, &s0, o); });
    printRow(td1, y, b0, d1, "I:pagerankMonolithicSeq (dynamic)");

    // Find OpenMP-based Monolithic pagerank.
    float tb2 = measureDuration([&]() { b2 = pagerankMonolithicOmp(y, yt, init, o); });
    printRow(tb2, y, b0, b2, "I:pagerankMonolithicOmp (static)");
    float tc2 = measureDuration([&]() { c2 = pagerankMonolithicOmp(y, yt, &s0, o); });
    printRow(tc2, y, b0, c2, "I:pagerankMonolithicOmp (incremental)");
    float td2 = measureDuration([&]() { d2 = pagerankMonolithicOmpDynamic(x, xt, y, yt, &s0, o); });
    printRow(td2, y, b0, d2, "I:pagerankMonolithicOmp (dynamic)");

    // Find CUDA-based Monolithic pagerank.
    float tb3 = measureDuration([&]() { b3 = pagerankMonolithicCuda(y, yt, init, o); });
    printRow(tb3, y, b0, b3, "I:pagerankMonolithicCuda (static)");
    float tc3 = measureDuration([&]() { c3 = pagerankMonolithicCuda(y, yt, &s0, o); });
    printRow(tc3, y, b0, c3, "I:pagerankMonolithicCuda (incremental)");
    float td3 = measureDuration([&]() { d3 = pagerankMonolithicCudaDynamic(x, xt, y, yt, &s0, o); });
    printRow(td3, y, b0, d3, "I:pagerankMonolithicCuda (dynamic)");

    // Find sequential Levelwise pagerank.
    float tb4 = measureDuration([&]() { b4 = pagerankLevelwiseSeq(y, yt, init, o); });
    printRow(tb4, y, b0, b4, "I:pagerankLevelwiseSeq (static)");
    float tc4 = measureDuration([&]() { c4 = pagerankLevelwiseSeq(y, yt, &s0, o); });
    printRow(tc4, y, b0, c4, "I:pagerankLevelwiseSeq (incremental)");
    float td4 = measureDuration([&]() { d4 = pagerankLevelwiseSeqDynamic(x, xt, y, yt, &s0, o); });
    printRow(td4, y, b0, d4, "I:pagerankLevelwiseSeq (dynamic)");

    // Find OpenMP-based Levelwise pagerank.
    float tb5 = measureDuration([&]() { b5 = pagerankLevelwiseOmp(y, yt, init, o); });
    printRow(tb5, y, b0, b5, "I:pagerankLevelwiseOmp (static)");
    float tc5 = measureDuration([&]() { c5 = pagerankLevelwiseOmp(y, yt, &s0, o); });
    printRow(tc5, y, b0, c5, "I:pagerankLevelwiseOmp (incremental)");
    float td5 = measureDuration([&]() { d5 = pagerankLevelwiseOmpDynamic(x, xt, y, yt, &s0, o); });
    printRow(td5, y, b0, d5, "I:pagerankLevelwiseOmp (dynamic)");

    // Find CUDA-based Levelwise pagerank.
    float tb6 = measureDuration([&]() { b6 = pagerankLevelwiseCuda(y, yt, init, o); });
    printRow(tb6, y, b0, b6, "I:pagerankLevelwiseCuda (static)");
    float tc6 = measureDuration([&]() { c6 = pagerankLevelwiseCuda(y, yt, &s0, o); });
    printRow(tc6, y, b0, c6, "I:pagerankLevelwiseCuda (incremental)");
    float td6 = measureDuration([&]() { d6 = pagerankLevelwiseCudaDynamic(x, xt, y, yt, &s0, o); });
    printRow(td6, y, b0, d6, "I:pagerankLevelwiseCuda (dynamic)");

    // DELETIONS:
    // Adjust ranks for deletions.
    auto s1 = b0.ranks;
    vector<T> r1(x.span());
    adjustRanks(r1, s1, ks, ksOld, 0.0f, float(Y)/(X+1), 1.0f/(X+1));

    // Find nvGraph-based pagerank.
    float te0 = measureDuration([&]() { e0 = pagerankNvgraph(x, xt, init, o); });
    printRow(te0, y, e0, e0, "D:pagerankNvgraph (static)");
    float tf0 = measureDuration([&]() { f0 = pagerankNvgraph(x, xt, &r1, o); });
    printRow(tf0, y, e0, f0, "D:pagerankNvgraph (incremental)");

    // Find sequential Monolithic pagerank.
    float te1 = measureDuration([&]() { e1 = pagerankMonolithicSeq(x, xt, init, o); });
    printRow(te1, y, e0, e1, "D:pagerankMonolithicSeq (static)");
    float tf1 = measureDuration([&]() { f1 = pagerankMonolithicSeq(x, xt, &r1, o); });
    printRow(tf1, y, e0, f1, "D:pagerankMonolithicSeq (incremental)");
    float tg1 = measureDuration([&]() { g1 = pagerankMonolithicSeqDynamic(y, yt, x, xt, &r1, o); });
    printRow(tg1, y, e0, g1, "D:pagerankMonolithicSeq (dynamic)");

    // Find OpenMP-based Monolithic pagerank.
    float te2 = measureDuration([&]() { e2 = pagerankMonolithicOmp(x, xt, init, o); });
    printRow(te2, y, e0, e2, "D:pagerankMonolithicOmp (static)");
    float tf2 = measureDuration([&]() { f2 = pagerankMonolithicOmp(x, xt, &r1, o); });
    printRow(tf2, y, e0, f2, "D:pagerankMonolithicOmp (incremental)");
    float tg2 = measureDuration([&]() { g2 = pagerankMonolithicOmpDynamic(y, yt, x, xt, &r1, o); });
    printRow(tg2, y, e0, g2, "D:pagerankMonolithicOmp (dynamic)");

    // Find CUDA-based Monolithic pagerank.
    float te3 = measureDuration([&]() { e3 = pagerankMonolithicCuda(x, xt, init, o); });
    printRow(te3, y, e0, e3, "D:pagerankMonolithicCuda (static)");
    float tf3 = measureDuration([&]() { f3 = pagerankMonolithicCuda(x, xt, &r1, o); });
    printRow(tf3, y, e0, f3, "D:pagerankMonolithicCuda (incremental)");
    float tg3 = measureDuration([&]() { g3 = pagerankMonolithicCudaDynamic(y, yt, x, xt, &r1, o); });
    printRow(tg3, y, e0, g3, "D:pagerankMonolithicCuda (dynamic)");

    // Find sequential Levelwise pagerank.
    float te4 = measureDuration([&]() { e4 = pagerankLevelwiseSeq(x, xt, init, o); });
    printRow(te4, y, e0, e4, "D:pagerankLevelwiseSeq (static)");
    float tf4 = measureDuration([&]() { f4 = pagerankLevelwiseSeq(x, xt, &r1, o); });
    printRow(tf4, y, e0, f4, "D:pagerankLevelwiseSeq (incremental)");
    float tg4 = measureDuration([&]() { g4 = pagerankLevelwiseSeqDynamic(y, yt, x, xt, &r1, o); });
    printRow(tg4, y, e0, g4, "D:pagerankLevelwiseSeq (dynamic)");

    // Find OpenMP-based Levelwise pagerank.
    float te5 = measureDuration([&]() { e5 = pagerankLevelwiseOmp(x, xt, init, o); });
    printRow(te5, y, e0, e5, "D:pagerankLevelwiseOmp (static)");
    float tf5 = measureDuration([&]() { f5 = pagerankLevelwiseOmp(x, xt, &r1, o); });
    printRow(tf5, y, e0, f5, "D:pagerankLevelwiseOmp (incremental)");
    float tg5 = measureDuration([&]() { g5 = pagerankLevelwiseOmpDynamic(y, yt, x, xt, &r1, o); });
    printRow(tg5, y, e0, g5, "D:pagerankLevelwiseOmp (dynamic)");

    // Find CUDA-based Levelwise pagerank.
    float te6 = measureDuration([&]() { e6 = pagerankLevelwiseCuda(x, xt, init, o); });
    printRow(te6, y, e0, e6, "D:pagerankLevelwiseCuda (static)");
    float tf6 = measureDuration([&]() { f6 = pagerankLevelwiseCuda(x, xt, &r1, o); });
    printRow(tf6, y, e0, f6, "D:pagerankLevelwiseCuda (incremental)");
    float tg6 = measureDuration([&]() { g6 = pagerankLevelwiseCudaDynamic(y, yt, x, xt, &r1, o); });
    printRow(tg6, y, e0, g6, "D:pagerankLevelwiseCuda (dynamic)");

    // New graph is now old.
    xo = move(yo);
  }
}


void runPagerank(const string& data, int repeat) {
  int M = countLines(data), steps = 10;
  printf("Temporal edges: %d\n", M);
  for (int batch=10, i=0; batch<M; batch*=i&1? 2:5, i++) {
    int skip = max(M/steps - batch, 0);
    printf("\n# Batch size %.0e\n", (double) batch);
    runPagerankBatch(data, repeat, skip, batch);
  }
}


int main(int argc, char **argv) {
  char *file = argv[1];
  int repeat = argc>2? stoi(argv[2]) : 5;
  printf("Using graph %s ...\n", file);
  string d = readFile(file);
  runPagerank(d, repeat);
    printf("\n");
  return 0;
}
