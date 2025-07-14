import os
import json

from tqdm.auto import tqdm

from ego4d.dataset.oscc import OSCCDataset
from ego4d.dataset.pnr import PNRDataset
from ego4d.dataset.ar import ARDataset
from ego4d.dataset.mq import MQDataset
from ego4d.dataset.lta import LTADataset

FPS = 30.0


def extract_video(sample, task: str, input_videos_path: str, output_videos_path: str, fps: float = FPS):

    ipath = os.path.join(input_videos_path, f"{sample.video_uid}.mp4")
    opath = os.path.join(output_videos_path, task, f"{sample.clip_uid}.mp4")

    if os.path.exists(opath):
        return

    print(f"{opath} is missing...")

    st = sample.video_start_frame / fps
    et = sample.video_end_frame / fps

    assert et > st

    os.system(f"ffmpeg -i {ipath} -ss {st:.2f} -t {(et - st):.2f} -c:v copy -c:a copy -y {opath}")


if __name__ == "__main__":

    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--ann-path", type=str, default="../../data/ego4d/raw/annotations/v1/")
    # args.add_argument("--input-videos-path", type=str, default="/home/speirone/ego-graph/timechat/TimeChat/ego4d_data/")
    args.add_argument("--input-videos-path", type=str, default="/data2/speirone/ego4d_videos_tmp/tmp/")
    args.add_argument("--output-videos-path", type=str, default="/data2/speirone/ego4d_hoi_videos_trimmed")
    args.add_argument("--split", type=str, default="train", choices=["train", "val"])
    args = args.parse_args()

    all_datasets = {
        "ar": ARDataset(args.split, root=args.ann_path),  # done val (2 samples are missing)
        "oscc": OSCCDataset(args.split, root=args.ann_path),  # done val
        "pnr": PNRDataset(args.split, root=args.ann_path),  # done train, val
        "lta": LTADataset(args.split, root=args.ann_path),  # done val
        "mq": MQDataset(args.split, root=args.ann_path),  # done train, val
    }

    # for vid in tqdm(set(x.video_uid for dset in all_datasets.values() for x in dset)):
    #     if not os.path.exists(f"/data2/speirone/ego4d_videos_tmp/tmp/{vid}.mp4"):
    #         os.system(f"rsync speirone@thanos:/home/speirone/ego4d_resized_videos/{vid}.mp4 /data2/speirone/ego4d_videos_tmp/tmp/")

    for task, dataset in all_datasets.items():
        print(f"Processing task {task}...")

        os.makedirs(os.path.join(args.output_videos_path, task), exist_ok=True)

        for sample in tqdm(dataset, total=len(dataset)):
            extract_video(sample, task, args.input_videos_path, args.output_videos_path, FPS)

    # Save samples
    samples = [sample.generate_it_sample() for dataset in all_datasets.values() for sample in dataset]
    for sample in samples:
        sample['video'] = f"{sample['source'].replace('ego4d_', '')}/{sample['video']}"
    json.dump(samples, open("ego4d_it.json", "w"))
