# Benchmark preparation scripts

Use the [fixed sequence lists](../../docs/eval.md#scannet-and-blendedmvs) for
paper comparisons. These scripts document how the subsets were prepared;
re-running them may select different sequences.

## ScanNet++

- `filter_sequence.py`: splits processed sequences using `nvs_test.txt`.
- `filter_sequence_depth.py`: selects sequences with depth and samples 10%
  for evaluation.

## BlendedMVS

- `convert_blendedmvs.py`: converts processed JPG/EXR/NPZ data to camera metadata.
- `filter_seq.py`: retains the original benchmark in the test set and targets
  90% training from the remaining sequences.
