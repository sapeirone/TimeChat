import argparse
from timechat.common.config import Config
from timechat.common.registry import registry
import decord

decord.bridge.set_bridge("torch")

# imports modules for registration
from timechat.datasets.builders import *
from timechat.models import *
from timechat.processors import *
from timechat.runners import *
from timechat.tasks import *

import os
import os.path as osp

from collections import defaultdict

from tqdm.auto import tqdm

from typing import List

import cv2

import torch

torch.set_grad_enabled(False)  # Disable gradients for inference


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="eval_configs/timechat.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--text-query", default="What is he doing?", help="question the video")
    parser.add_argument("--video-path", default="examples/hotdog.mp4", help="path to video file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair " "in xxx=yyy format will be merged into config file (deprecate), " "change to --cfg-options instead.",
    )
    args = parser.parse_args(args=[])
    return args


def build_model(ckpt="ckpt/timechat/timechat_7b.pth", long_context=False):
    print("Initializing Chat")
    args = parse_args()

    if long_context:
        print("Using long context configuration.")
        args.cfg_path = "eval_configs/timechat_long_context.yaml"

    cfg = Config(args)

    assert osp.exists(ckpt), f"Model checkpoint {ckpt} does not exist."

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_config.ckpt = ckpt
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f"cuda:{args.gpu_id}")
    model.eval()

    vis_proc_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_proc = registry.get_processor_class(vis_proc_cfg.name).from_config(vis_proc_cfg)

    return model, vis_proc


def save_frames_from_list(
    video_uids: List[str], clip_uids: List[str], video_start_frames: List[float], video_end_frames: List[float], ego4d_data_path: str, output_dir: str = "/data2/speirone/egopack_rebuttal_llm_videos/"
):

    # Creating the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Grouping samples by video_uid to speed up the process
    grouped_samples = defaultdict(list)
    for video_uid, clip_uid, video_start_frame, video_end_frame in zip(video_uids, clip_uids, video_start_frames, video_end_frames):
        grouped_samples[video_uid].append((clip_uid, video_start_frame, video_end_frame))

    for video_uid, video_samples in tqdm(grouped_samples.items()):

        input_path = osp.join(ego4d_data_path, f"{video_uid}.mp4")

        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # If all clips for this video already exist, skip processing this video
        if all(os.path.exists(osp.join(output_dir, f"{clip_uid}.mp4")) for (clip_uid, _, _) in video_samples):
            continue

        # Prepare output video writers for each clip
        outs = [(csf, vef, cv2.VideoWriter(osp.join(output_dir, f"{clip_uid}.mp4"), fourcc, fps, (width, height))) for clip_uid, csf, vef in video_samples]

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            for vsf, vef, out in outs:
                if vsf <= frame_idx <= vef:
                    out.write(frame)

            frame_idx += 1

        cap.release()
        for _, _, out in outs:
            out.release()
