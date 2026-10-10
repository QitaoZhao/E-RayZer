# Historical protocol reference; see ../README.md before use.
from tqdm import tqdm
from flash_s3_dataloader.s3_io import list_s3_dir, load_s3_json


with open("nvs_test.txt", "r") as f:
    benchmark_sequences = f.read().splitlines()

dataset_dir = "s3://YOUR_BUCKET/scannetpp_processed/"
sequence_dirs = list_s3_dir(dataset_dir)

camera_path_list = []
camera_path_list_benchmark = []
for sequence_dir in tqdm(sequence_dirs):
    camera_path = sequence_dir + "opencv_cameras.json"
    camera_data = load_s3_json(camera_path)
    scene_name = camera_data["scene_name"].split("_")[0]
    if scene_name in benchmark_sequences:
        camera_path_list_benchmark.append(camera_path)
    else:
        camera_path_list.append(camera_path)


txt_path = "scannetpp.txt"
with open(txt_path, "w") as f:
    for d in camera_path_list:
        f.write(f"{d}\n")

print(f"Saved {len(camera_path_list)} entries to {txt_path}")

txt_path = "scannetpp_benchmark.txt"
with open(txt_path, "w") as f:
    for d in camera_path_list_benchmark:
        f.write(f"{d}\n")

print(f"Saved {len(camera_path_list_benchmark)} entries to {txt_path}")
