// C++ simulation core for the Monte Carlo pricer, exposed to Python with pybind11.
//
// Same models as the Python package, but path by path: each simulated payoff is
// added to a running sum and sum of squares, so memory stays O(1) whatever n is,
// and the paths are split across CPU threads.
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

struct Stats {
    double sum = 0.0, sum_sq = 0.0;
    long long n = 0;
    void add(double y) { sum += y; sum_sq += y * y; ++n; }
};

// Runs simulate_one n times, split across threads, and returns (price, standard error).
// Each thread has its own random generator (seeded seed + thread index), so results
// are reproducible for a given (seed, threads) pair but change if threads changes.
template <class F>
py::tuple run(long long n, unsigned long long seed, int threads, double discount, F simulate_one) {
    if (n < 2) throw std::invalid_argument("n must be at least 2");
    if (threads <= 0) threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<Stats> parts(threads);
    {
        py::gil_scoped_release release;  // let Python do other work while we compute
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t) {
            pool.emplace_back([&, t] {
                std::mt19937_64 rng(seed + 1000003ULL * t);
                std::normal_distribution<double> z;
                long long count = n / threads + (t < n % threads ? 1 : 0);
                for (long long i = 0; i < count; ++i) parts[t].add(simulate_one(rng, z));
            });
        }
        for (auto& th : pool) th.join();
    }
    Stats total;
    for (const auto& p : parts) { total.sum += p.sum; total.sum_sq += p.sum_sq; total.n += p.n; }
    double mean = total.sum / total.n;
    // ponytail: textbook variance formula; switch to Welford if payoffs ever get huge vs their spread
    double var = (total.sum_sq - total.n * mean * mean) / (total.n - 1);
    return py::make_tuple(discount * mean, discount * std::sqrt(var / total.n));
}

bool is_call(const std::string& kind) {
    if (kind == "call") return true;
    if (kind == "put") return false;
    throw std::invalid_argument("kind must be 'call' or 'put', not '" + kind + "'");
}

py::tuple european(double S0, double K, double T, double r, double sigma, long long n,
                   const std::string& kind, unsigned long long seed, int threads) {
    bool call = is_call(kind);
    double drift = (r - 0.5 * sigma * sigma) * T, vol = sigma * std::sqrt(T);
    return run(n, seed, threads, std::exp(-r * T), [=](auto& rng, auto& z) {
        double ST = S0 * std::exp(drift + vol * z(rng));
        return call ? std::max(ST - K, 0.0) : std::max(K - ST, 0.0);
    });
}

py::tuple asian_call(double S0, double K, double T, double r, double sigma, long long n, int m,
                     unsigned long long seed, int threads) {
    double dt = T / m, drift = (r - 0.5 * sigma * sigma) * dt, vol = sigma * std::sqrt(dt);
    return run(n, seed, threads, std::exp(-r * T), [=](auto& rng, auto& z) {
        double S = S0, total = 0.0;
        for (int k = 0; k < m; ++k) {
            S *= std::exp(drift + vol * z(rng));
            total += S;
        }
        return std::max(total / m - K, 0.0);
    });
}

py::tuple down_and_out_call(double S0, double K, double H, double T, double r, double sigma,
                            long long n, int m, unsigned long long seed, int threads) {
    double dt = T / m, drift = (r - 0.5 * sigma * sigma) * dt, vol = sigma * std::sqrt(dt);
    return run(n, seed, threads, std::exp(-r * T), [=](auto& rng, auto& z) {
        double S = S0;
        for (int k = 0; k < m; ++k) {
            S *= std::exp(drift + vol * z(rng));
            if (S <= H) return 0.0;  // knocked out: no need to simulate the rest of the path
        }
        return std::max(S - K, 0.0);
    });
}

}  // namespace

PYBIND11_MODULE(mcpricer_cpp, mod) {
    mod.doc() = "C++ Monte Carlo core. Every function returns (price, standard_error). "
                "threads=0 uses every CPU core.";
    mod.def("european", &european, "European call or put.",
            py::arg("S0"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"), py::arg("n"),
            py::arg("kind") = "call", py::arg("seed") = 0, py::arg("threads") = 0);
    mod.def("asian_call", &asian_call, "Call on the arithmetic average of m fixings (plain MC).",
            py::arg("S0"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"), py::arg("n"),
            py::arg("m"), py::arg("seed") = 0, py::arg("threads") = 0);
    mod.def("down_and_out_call", &down_and_out_call, "Down-and-out call, barrier checked at m dates.",
            py::arg("S0"), py::arg("K"), py::arg("H"), py::arg("T"), py::arg("r"), py::arg("sigma"),
            py::arg("n"), py::arg("m"), py::arg("seed") = 0, py::arg("threads") = 0);
}
