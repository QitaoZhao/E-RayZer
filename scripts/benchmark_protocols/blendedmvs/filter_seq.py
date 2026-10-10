# Historical protocol reference; see ../README.md before use.
all_paths_file = "blendedmvs_all.txt"
org_test_file = "blendedmvs_benchmark.txt"
output_train_file = "blendedmvs_depth.txt"
output_test_file = "blendedmvs_depth_benchmark.txt"

# Read all paths from all_paths_file
with open(all_paths_file) as f:
    all_paths = [line.strip() for line in f if line.strip()]

# read benchmark paths to exclude them from training set
with open(org_test_file) as f:
    benchmark_paths = {line.strip() for line in f if line.strip()}

"""
Randomly sample 90% for training and 10% for testing from the UNION of
all_paths and benchmark_paths, with the following constraints:
- Original benchmark paths are ALWAYS in the test set
- Original benchmark paths are EXCLUDED from the training set

Note: If the benchmark set is large (> 10% of the union), it's impossible to
hit an exact 90/10 split under the constraint. In that case, we fill the
training set from eligible (non-benchmark) items up to the 90% target or the
available count, whichever is smaller. The remainder plus all benchmark paths
become the test set.
"""

import random

# Build union of all paths and benchmark paths
union_set = set(all_paths) | benchmark_paths

# Eligible for training = union minus benchmark
eligible_train = list(union_set - benchmark_paths)

random.seed(42)
random.shuffle(eligible_train)

target_train_count = int(0.9 * len(union_set))
train_count = min(target_train_count, len(eligible_train))

train_paths = eligible_train[:train_count]
remaining_candidates = set(eligible_train[train_count:])

# Test contains all remaining non-benchmark candidates plus ALL benchmark paths
test_paths = list(remaining_candidates | benchmark_paths)

# Write results
with open(output_train_file, "w") as f:
    f.write("\n".join(train_paths))
with open(output_test_file, "w") as f:
    f.write("\n".join(test_paths))

total_union = len(union_set)
print(f"✅ Train set: {len(train_paths)} paths")
print(f"✅ Test set (incl. original benchmark): {len(test_paths)} paths")
print(f"📊 Total (union): {total_union} paths")

# Diagnostics
bench_count = len(benchmark_paths)
eligible_count = len(eligible_train)
if target_train_count > eligible_count:
    print(
        "ℹ️  Target train count exceeds non-benchmark pool; "
        f"target={target_train_count}, eligible={eligible_count}. "
        "Filled train with all available eligible items; test contains the rest + all benchmarks."
    )
print(f"🔎 Benchmark (forced to test): {bench_count} paths")
