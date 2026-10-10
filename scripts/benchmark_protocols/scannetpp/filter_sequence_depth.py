# Historical protocol reference; see ../README.md before use.

# The official NVS eval split does not include depth. So randomly choose 10% sequences for benchmark

from flash_s3_dataloader.s3_io import _get_s3_client, load_s3_json
from tqdm import tqdm
import random

s3_client = _get_s3_client()
sequence_path = "./scannetpp.txt"
sequence_path_benchmark = "./scannetpp_benchmark.txt"

sequences_depth_all = []

# Open the .txt file
with open(sequence_path, "r") as f:
    sequences = f.read().splitlines()

with open(sequence_path_benchmark, "r") as f:
    sequences_benchmark = f.read().splitlines()

for seq_anno in tqdm(sequences):
    anno_data = load_s3_json(seq_anno, s3_client)
    if "depth_path" in anno_data["frames"][0]:
        sequences_depth_all.append(seq_anno)

for seq_anno in tqdm(sequences_benchmark):
    anno_data = load_s3_json(seq_anno, s3_client)
    if "depth_path" in anno_data["frames"][0]:
        sequences_depth_all.append(seq_anno)

# randomly choose 10% sequences for benchmark

num_benchmark = max(1, int(0.1 * len(sequences_depth_all)))
sequences_benchmark_depth = random.sample(sequences_depth_all, num_benchmark)

# the rest of them are training sequences:
sequences_depth = [seq for seq in sequences_depth_all if seq not in sequences_benchmark_depth]

# write the sequences
with open(sequence_path.replace("scannetpp.txt", "scannetpp_depth.txt"), "w") as f:
    f.write("\n".join(sequences_depth))
with open(sequence_path_benchmark.replace("scannetpp_benchmark.txt", "scannetpp_depth_benchmark.txt"), "w") as f:
    f.write("\n".join(sequences_benchmark_depth))
