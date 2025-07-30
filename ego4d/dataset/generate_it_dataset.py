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

    # if os.path.exists(opath):
    #     return

    if not os.path.exists(ipath):
        print(f"Copying {sample.video_uid}.mp4...")
        os.system(f"rsync speirone@thanos:/home/speirone/ego4d_resized_videos/{sample.video_uid}.mp4 /data2/speirone/ego4d_videos_tmp/tmp/")

    # print(sample)
    # print(f"{opath} from {sample.video_uid} is missing...")

    st = sample.video_start_frame / fps
    et = sample.video_end_frame / fps

    assert et > st

    ret = os.system(f"ffmpeg -i {ipath} -ss {st:.2f} -t {(et - st):.2f} -c:v copy -c:a copy -y {opath} > /dev/null 2>&1")
    if ret != 0:
        print(f"Return code != 0. Removing the generated video {opath}")


if __name__ == "__main__":

    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--ann-path", type=str, default="ego4d/annotations/v1/")
    # args.add_argument("--input-videos-path", type=str, default="/home/speirone/ego-graph/timechat/TimeChat/ego4d_data/")
    args.add_argument("--input-videos-path", type=str, default="/data2/speirone/ego4d_videos_tmp/tmp/")
    args.add_argument("--output-videos-path", type=str, default="/data2/speirone/ego4d_hoi_videos_trimmed")
    args = args.parse_args()

    all_datasets = {
        "ar": ARDataset("train", root=args.ann_path),
        "oscc": OSCCDataset("train", root=args.ann_path),
        "pnr": PNRDataset("train", root=args.ann_path),
        "lta": LTADataset("train", root=args.ann_path),
        "mq": MQDataset("train", root=args.ann_path),
    }

    # for task, dataset in all_datasets.items():
    #     print(f"Processing task {task}...")

    #     os.makedirs(os.path.join(args.output_videos_path, task), exist_ok=True)

    #     for sample in tqdm(dataset, total=len(dataset)):
    #         extract_video(sample, task, args.input_videos_path, args.output_videos_path, FPS)

    # all_datasets = {
    #     "ar": ARDataset("val", root=args.ann_path),
    #     "oscc": OSCCDataset("val", root=args.ann_path),
    #     "pnr": PNRDataset("val", root=args.ann_path),
    #     "lta": LTADataset("val", root=args.ann_path),
    #     "mq": MQDataset("val", root=args.ann_path),
    # }

    # for task, dataset in all_datasets.items():
    #     print(f"Processing task {task}...")

    #     os.makedirs(os.path.join(args.output_videos_path, task), exist_ok=True)

    #     for sample in tqdm(dataset, total=len(dataset)):
    #         extract_video(sample, task, args.input_videos_path, args.output_videos_path, FPS)

    # Save samples
    samples = [sample.generate_it_sample() for dataset in all_datasets.values() for sample in dataset]
    for sample in samples:
        sample['video'] = f"{sample['source'].replace('ego4d_', '')}/{sample['video']}"
    print(f"There are {len(samples)} samples...")
    json.dump(samples, open("ego4d_it.json", "w"))
