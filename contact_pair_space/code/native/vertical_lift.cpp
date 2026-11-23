// vertical_lift.cpp  (vertical_lift_v1.cpp of 15 Nov 2026, unchanged arithmetic)
// Exact smallest upward vertical translation t >= 0 that removes every overlap of a moving ball union A with a second
// ball union B and with the table half-space z >= 0 (all in the world frame; z is the world vertical).
// Moving body A: balls a_i relative to its centre of mass, pose (Ra, ca): p_i = Ra a_i + ca.
// Other body B:  balls b_j relative to its centre of mass, pose (Rb, cb): q_j = Rb b_j + cb.
// A pair overlaps at offset t when |p_i + t e_z - q_j| < r_i + r_j - tol; table: p_iz + t - r_i < -tol.
// An overlapping pair stops overlapping (moving up) at u_ij = (q_jz - p_iz) + sqrt((r_i + r_j)^2 - |p_i - q_j|_xy^2);
// a table pair at u_i = r_i - p_iz. Iteration: t <- max{u over pairs overlapping at t} until no pair overlaps at t.
// Every overlap interval along the vertical line is an open interval (l, u); the iteration passes the union of the
// intervals that contain the start exactly, so the result is the smallest t >= 0 at which nothing overlaps.
// Branch and bound over sphere trees: a node sphere contains its balls, so for a node pair with centre distance D and
// radii sum S every contained ball pair has distance - (r_i + r_j) >= D - S, and every contained upper end satisfies
// u <= (Q_z - P_z) + S (P, Q node centres). The table bound per node: u_i <= rad_I - P_z.
// C ABI for ctypes. Build: ../build_native.sh (c++ -O3 -std=c++17, shared library in contact_pair/_native/)
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

namespace {
const double NEG = -std::numeric_limits<double>::infinity();

struct Tree {
  const double* cen; const double* rad; const int32_t* left; const int32_t* right; const int32_t* lo; const int32_t* cnt;
  const double* bc; const double* br;
};

inline void xf(const double* R, const double* c, const double* a, double* p) {
  p[0] = R[0] * a[0] + R[1] * a[1] + R[2] * a[2] + c[0];
  p[1] = R[3] * a[0] + R[4] * a[1] + R[5] * a[2] + c[1];
  p[2] = R[6] * a[0] + R[7] * a[1] + R[8] * a[2] + c[2];
}

struct Ctx { Tree A, B; int use_other, use_table; double tol; };

struct Work { std::vector<std::pair<int32_t, int32_t>> st2; std::vector<int32_t> st1; int64_t visited = 0; };

// largest upper end over all pairs that overlap at offset t (NEG if none)
double max_upper(const Ctx& C, const double* Ra, const double* ca, const double* Rb, const double* cb, double t, Work& W) {
  double best = NEG;
  if (C.use_table) {
    W.st1.clear(); W.st1.push_back(0);
    while (!W.st1.empty()) {
      const int32_t I = W.st1.back(); W.st1.pop_back(); ++W.visited;
      double P[3]; xf(Ra, ca, C.A.cen + 3 * I, P);
      if (P[2] + t - C.A.rad[I] >= -C.tol) continue;        // no ball of the node reaches below the table
      if (C.A.rad[I] - P[2] <= best) continue;              // cannot beat the current best upper end
      if (C.A.left[I] < 0) {
        const int32_t k0 = C.A.lo[I], k1 = k0 + C.A.cnt[I];
        for (int32_t k = k0; k < k1; ++k) {
          double p[3]; xf(Ra, ca, C.A.bc + 3 * k, p);
          const double r = C.A.br[k];
          if (p[2] + t - r < -C.tol) best = std::max(best, r - p[2]);
        }
      } else { W.st1.push_back(C.A.left[I]); W.st1.push_back(C.A.right[I]); }
    }
  }
  if (C.use_other) {
    W.st2.clear(); W.st2.push_back({0, 0});
    while (!W.st2.empty()) {
      const auto pr = W.st2.back(); W.st2.pop_back(); ++W.visited;
      const int32_t I = pr.first, J = pr.second;
      double P[3], Q[3]; xf(Ra, ca, C.A.cen + 3 * I, P); xf(Rb, cb, C.B.cen + 3 * J, Q);
      const double dx = P[0] - Q[0], dy = P[1] - Q[1], dz = P[2] + t - Q[2];
      const double S = C.A.rad[I] + C.B.rad[J];
      const double D = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (D - S >= -C.tol) continue;                         // no contained pair overlaps
      if ((Q[2] - P[2]) + S <= best) continue;               // no contained upper end beats best
      const bool la = C.A.left[I] < 0, lb = C.B.left[J] < 0;
      if (la && lb) {
        const int32_t i0 = C.A.lo[I], i1 = i0 + C.A.cnt[I], j0 = C.B.lo[J], j1 = j0 + C.B.cnt[J];
        for (int32_t i = i0; i < i1; ++i) {
          double p[3]; xf(Ra, ca, C.A.bc + 3 * i, p); const double ri = C.A.br[i];
          for (int32_t j = j0; j < j1; ++j) {
            double q[3]; xf(Rb, cb, C.B.bc + 3 * j, q); const double s = ri + C.B.br[j];
            const double ex = p[0] - q[0], ey = p[1] - q[1], ez = p[2] + t - q[2];
            const double d = std::sqrt(ex * ex + ey * ey + ez * ez);
            if (d - s < -C.tol) {
              const double h2 = ex * ex + ey * ey;
              const double w = std::sqrt(std::max(0.0, s * s - h2));
              best = std::max(best, (q[2] - p[2]) + w);
            }
          }
        }
      } else if (lb || (!la && C.A.rad[I] >= C.B.rad[J])) {
        W.st2.push_back({C.A.left[I], J}); W.st2.push_back({C.A.right[I], J});
      } else {
        W.st2.push_back({I, C.B.left[J]}); W.st2.push_back({I, C.B.right[J]});
      }
    }
  }
  return best;
}
}  // namespace

extern "C" {

// For each of n poses: smallest t >= 0 such that the moving body lifted by t overlaps neither the other body nor the
// table. status: 0 no overlap at the start (t = 0); 1 lifted; 2 iteration limit reached (t is a lower bound).
int vl_lift(const double* A_cen, const double* A_rad, const int32_t* A_left, const int32_t* A_right, const int32_t* A_lo,
            const int32_t* A_cnt, const double* A_bc, const double* A_br,
            const double* B_cen, const double* B_rad, const int32_t* B_left, const int32_t* B_right, const int32_t* B_lo,
            const int32_t* B_cnt, const double* B_bc, const double* B_br, int use_other, int use_table,
            int n, const double* Ra, const double* ca, const double* Rb, const double* cb, double tol, int max_iter,
            int n_threads, double* t_out, int32_t* status_out, int32_t* iters_out, int64_t* visited_out) {
  Ctx C;
  C.A = Tree{A_cen, A_rad, A_left, A_right, A_lo, A_cnt, A_bc, A_br};
  C.B = Tree{B_cen, B_rad, B_left, B_right, B_lo, B_cnt, B_bc, B_br};
  C.use_other = use_other; C.use_table = use_table; C.tol = tol;
  std::atomic<int> next(0);
  auto worker = [&]() {
    Work W;
    for (;;) {
      const int k = next.fetch_add(1);
      if (k >= n) break;
      W.visited = 0;
      const double* RA = Ra + 9 * k; const double* cA = ca + 3 * k; const double* RB = Rb + 9 * k; const double* cB = cb + 3 * k;
      double t = 0.0; int it = 0, status = 0;
      for (;;) {
        const double u = max_upper(C, RA, cA, RB, cB, t, W);
        if (!(u > t + 1e-12)) break;                        // nothing overlaps at t (or only rounding at its own end)
        t = u; status = 1; ++it;
        if (it >= max_iter) { status = 2; break; }
      }
      t_out[k] = t; status_out[k] = status; iters_out[k] = it; visited_out[k] = W.visited;
    }
  };
  const int nt = std::max(1, std::min(n_threads, n));
  std::vector<std::thread> pool;
  for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
  worker();
  for (auto& th : pool) th.join();
  return 0;
}

}  // extern "C"
