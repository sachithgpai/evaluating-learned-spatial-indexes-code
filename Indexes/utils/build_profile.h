#ifndef BUILD_PROFILE_H
#define BUILD_PROFILE_H

/**
 * @file build_profile.h
 * @brief Splitting `build_time` into the parts a reviewer can act on.
 *
 * Today every index reports one number. That number means something different
 * per index: for STR it is a sort and a pack, for RSMI it is 256 epochs of SGD
 * in a separate Python process, for FLOOD it is a random search whose length
 * nobody recorded. Reporting them in one column invites exactly the comparison
 * that cannot be made.
 *
 * The split here is deliberately coarse -- four timed phases, not a call tree.
 * Anything finer would have to be defended line by line, and the interesting
 * aggregates are sums of these four anyway:
 *
 *     learning                = learn_s + learn_oracle_s + eval_s
 *     workload-awareness      = workload_model_s + workload_oracle_s
 *     plain structure         = construct_s + serialize_s
 *
 * Those three are disjoint and sum to the whole build, so they stack in a figure
 * and a reader can check the arithmetic.
 *
 * A model is consulted in two places and both are timed: once when it is BUILT
 * (workload_model_s / learn_s) and repeatedly while the index is being ASSEMBLED
 * (the *_oracle_s pair). The second is where the cost actually lives -- RW builds
 * its query estimator in about a millisecond and then consults it 1.8 million
 * times. Timing those calls was measured at 75 ns per pair, 0.54% of RW's build,
 * so the clock does not meaningfully perturb what it reports.
 */

#include <chrono>
#include <cstdint>
#include <string>


/**
 * Where an index's build cost was actually incurred.
 *
 * `kOfflinePython` exists because RSMI's number arrives from a `.time` file
 * written by a separate process and covers a different span than every
 * in-process measurement. Anything comparing build_time across indexes has to
 * be able to see that, so it travels with the row rather than living in a
 * footnote.
 */
enum class BuildAccounting { kInProcess, kOfflinePython };

inline const char* BuildAccountingName(BuildAccounting accounting){
    return accounting == BuildAccounting::kOfflinePython ? "offline_python" : "in_process";
}


struct BuildProfile{
    // ---- timed phases, seconds ----
    double workload_model_s = 0.0;  // building a structure over the QUERIES
    double learn_s          = 0.0;  // fitting parameters, or building search candidates
    double eval_s           = 0.0;  // scoring a candidate/model against the workload
    double workload_oracle_s= 0.0;  // consulting the query model DURING construction
    double learn_oracle_s   = 0.0;  // consulting the data model DURING construction
    double construct_s      = 0.0;  // DERIVED at log time: wall clock minus the rest
    double serialize_s      = 0.0;  // writing or reading a trained artifact

    // ---- counters that make the phases reproducible ----
    uint64_t search_trials       = 0;   // random-search iterations actually run
    uint64_t search_improvements = 0;   // iterations that beat the incumbent
    uint64_t oracle_calls        = 0;   // cost-model lookups during construction
    uint64_t training_queries    = 0;   // queries the build consumed
    uint64_t learned_nodes       = 0;   // nodes that fitted a model
    uint64_t fallback_nodes      = 0;   // nodes that took the non-learned path

    BuildAccounting accounting = BuildAccounting::kInProcess;

    /** The phases an index claimed explicitly. Construction is the remainder. */
    double ClaimedSeconds() const {
        return workload_model_s + workload_oracle_s + learn_s + learn_oracle_s
             + eval_s + serialize_s;
    }

    /**
     * The two reported components. Disjoint, so they sum with construct_s and
     * serialize_s to the whole build and can be stacked in a figure.
     *
     * `eval_s` sits in learning rather than being shared: for FLOOD and QD,
     * scoring a candidate against the workload IS the learning signal -- the
     * search has no other objective. That those indexes learn *from* the
     * workload is a property worth stating in prose, not a reason to double-count
     * seconds into two columns that then cannot be added up.
     */
    double LearningSeconds() const { return learn_s + learn_oracle_s + eval_s; }
    double WorkloadAwarenessSeconds() const { return workload_model_s + workload_oracle_s; }
};


/**
 * Adds its lifetime to one phase accumulator.
 *
 * Accumulates rather than assigns, so a phase entered once per search trial
 * sums across trials without the caller keeping a running total.
 */
class ScopedPhase{
    public:
        explicit ScopedPhase(double& sink)
            : sink_(&sink), start_(std::chrono::high_resolution_clock::now()) {}

        ~ScopedPhase(){ Stop(); }

        /** End the phase early. Idempotent, so an explicit Stop() then scope exit is safe. */
        void Stop(){
            if(sink_ == nullptr)
                return;
            const auto end = std::chrono::high_resolution_clock::now();
            *sink_ += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count()/1e9;
            sink_ = nullptr;
        }

        ScopedPhase(const ScopedPhase&) = delete;
        ScopedPhase& operator=(const ScopedPhase&) = delete;

    private:
        double* sink_;
        std::chrono::high_resolution_clock::time_point start_;
};


/**
 * The build profile of the index currently under construction.
 *
 * A single global rather than a parameter threaded through twelve constructors
 * and their recursive helpers. The evaluator is single-threaded and builds one
 * index at a time, so there is nothing to race; `ResetBuildProfile()` at the
 * top of each index section is what keeps them separate.
 */
inline BuildProfile& CurrentBuildProfile(){
    static BuildProfile profile;
    return profile;
}

inline void ResetBuildProfile(){
    CurrentBuildProfile() = BuildProfile{};
}


#endif
