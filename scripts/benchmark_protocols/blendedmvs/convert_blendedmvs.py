# Historical protocol reference; see ../README.md before use.
import os
import torch
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from flash_s3_dataloader.s3_io import (
    _get_s3_client,
    load_s3_image,
    load_s3_numpy,
    list_s3_dir,
    load_s3_exr,
)

# Global client for each worker
_s3_client = None
def init_worker():
    global _s3_client
    _s3_client = _get_s3_client()


def process_scene(scene_name):
    """Process one scene and save opencv_cameras.json"""
    scene_dir = os.path.join(dataset_dir, scene_name)

    save_dir = os.path.join(output_dir, scene_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "opencv_cameras.json")

    seq_data = {"scene_name": scene_name, "frames": []}

    # List files and extract unique frame names
    file_list = list_s3_dir(scene_dir + "/", s3_client=_s3_client)
    frame_list = {os.path.basename(f).split(".")[0] for f in file_list if not os.path.basename(f).startswith("opencv_cameras")}
    frame_list = sorted(frame_list, key=lambda x: int(x))

    for frame_name in frame_list:
        cam_path = os.path.join(scene_dir, f"{frame_name}.npz")

        # Load image & depth
        image = load_s3_image(os.path.join(scene_dir, f"{frame_name}.jpg"), s3_client=_s3_client)
        _ = load_s3_exr(os.path.join(scene_dir, f"{frame_name}.exr"), s3_client=_s3_client)

        # Load camera intrinsics/extrinsics
        cam = load_s3_numpy(cam_path, s3_client=_s3_client)

        W, H = image.size
        intrinsics = torch.from_numpy(cam["intrinsics"])   # (3, 3)
        R = torch.from_numpy(cam["R_cam2world"])           # (3, 3)
        T = torch.from_numpy(cam["t_cam2world"])           # (3,)

        # Build 4x4 c2w
        c2w = torch.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = T

        # Invert for w2c
        w2c = torch.linalg.inv(c2w)

        # Intrinsics params
        fx, fy = intrinsics[0, 0].item(), intrinsics[1, 1].item()
        cx, cy = intrinsics[0, 2].item(), intrinsics[1, 2].item()

        frame_data = {
            "file_path": f"{frame_name}.jpg",
            "depth_path": f"{frame_name}.exr",
            "w": W,
            "h": H,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "w2c": w2c.tolist(),
        }
        seq_data["frames"].append(frame_data)

    # Save JSON
    with open(save_path, "w") as f:
        json.dump(seq_data, f, indent=4, ensure_ascii=False)

    # Return S3 path for collection
    return f"{dataset_dir}/{scene_name}/opencv_cameras.json"


if __name__ == "__main__":
    # Config
    dataset_dir = "s3://YOUR_BUCKET/blendedmvs_processed"
    output_dir = "./blendedmvs_processed"
    os.makedirs(output_dir, exist_ok=True)

    with open("BlendedMVG.txt", "r") as f:
        scene_list = [line.strip() for line in f if line.strip()]

    # Process scenes in parallel
    with Pool(processes=cpu_count() - 2, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap(process_scene, scene_list), total=len(scene_list), desc="Scenes"))

    # Save final list of JSON paths
    all_json_paths = [r for r in results if r is not None]
    list_file = "blendedmvs_all.txt"
    with open(list_file, "w") as f:
        f.writelines(p + "\n" for p in all_json_paths)

    print(f"📄 Collected {len(all_json_paths)} JSON paths")
    print(f"💾 Written to {list_file}")
