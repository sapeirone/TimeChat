# TimeChat eval for Moment Queries on Ego4D dataset

import re
import argparse
import os.path as osp
import warnings

from tqdm.auto import tqdm

import numpy as np
import spacy

import editdistance

from eval_utils import build_model
from ego4d.dataset.mq import MQDataset
from timechat.conversation.conversation_video import Chat, conv_llava_llama_2

nlp = spacy.load("en_core_web_sm")

warnings.filterwarnings("ignore", category=UserWarning)


SYSTEM_PROMPT = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."

TASK_DESCRIPTION = "You are given a video containing several human activities. "

PROMPT = "Find all the video segments that corresponds to the given textual query '{}' and determine their start and end seconds. If the action appears more than once, return all the occurrences."


def ask(video_uid, query, timechat_model, timechat_vis_processor, activities_list, num_frames=8, data_path: str = "ego4d_hoi_trimmed_videos/ar"):
    """Ask the TimeChat model about MQ samples and return the raw unparsed response of the llm."""
    chat = Chat(timechat_model, timechat_vis_processor, device="cuda")

    frames = []
    state = conv_llava_llama_2.copy()
    state.system = SYSTEM_PROMPT

    # Description of the task
    chat.ask(TASK_DESCRIPTION.format(activities_list), state)

    # Feed the sample and ask the question
    path = osp.join(data_path, video_uid + ".mp4")
    chat.upload_video_without_audio(video_path=path, conv=state, img_list=frames, n_frms=num_frames)
    chat.ask(PROMPT.format(query), state, role="USER")

    return chat.answer(conv=state, img_list=frames, num_beams=1, temperature=1.0, max_length=3000)[0]


def eval_ed(preds, labels):
    """
    Damerau–Levenshtein edit distance from: https://github.com/gfairchild/pyxDamerauLevenshtein.
    For each sample, we take the smallest edit distance among all the K predicted sequences.
    """
    N, Z, K = preds.shape
    dists = []
    for n in range(N):
        dist = min([editdistance.eval(preds[n, :, k], labels[n]) / Z for k in range(K)])
        dists.append(dist)
    return np.array(dists)


if __name__ == "__main__":
    # Example usage

    args = argparse.ArgumentParser(description="Ego4D MQ ICL Demo")
    args.add_argument("--timechat-ckpt", type=str, default="ckpt/timechat/timechat_7b.pth")
    args.add_argument("--num-frames", type=int, default=8, help="Number of frames to sample from the video.")
    args.add_argument("--video-path", type=str, default="ego4d_hoi_trimmed_videos/mq", help="Processed video path to use for the Ego4D dataset.")

    args = args.parse_args()

    print("\n")
    print("###########################")
    print(f"Using {args.num_frames} frames.")
    print(f"Video path: {args.video_path}")
    print(f"TimeChat ckpt: {args.timechat_ckpt}")
    print("###########################")
    print("\n")

    # Build the TimeChat model
    model, vis_processor = build_model(ckpt=args.timechat_ckpt)

    # Action Recognition dataset
    print("Loading MQ dataset...")
    dset_train = MQDataset(split="train")
    dset_val = MQDataset(split="val")

    print(f"Loaded {len(dset_val)} validation samples.")

    # Collect here the number of correct verb and noun predictions
    avg_tiou, r1_01, r1_05 = [], 0, 0

    # Keep track of the number of samples for which it was not possible to compute the metric
    broken_samples = 0

    pbar = tqdm(dset_val, total=len(dset_val), desc="Processing MQ videos...")
    for video in pbar:

        vsf, vef = video.video_start_frame, video.video_end_frame

        try:
            for segment in video.segments:
                ssf, sef = segment.video_start_frame, segment.video_end_frame
                label = segment.label.replace("_/_", "_or_").replace("_", " ")

                nst, net = (ssf - vsf) / 30.0, (sef - vsf) / 30.0

                response = ask(
                    video_uid=video.clip_uid,
                    query=label,
                    timechat_model=model,
                    timechat_vis_processor=vis_processor,
                    num_frames=args.num_frames,
                    activities_list=dset_val.class_labels,
                    data_path=args.video_path,
                )

                best_iou = 0.0
                for r in response.split(". "):
                    # print(r)

                    match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*seconds", r)
                    if match:
                        pred_start = float(match.group(1))
                        pred_end = float(match.group(2))

                        inter_start = max(nst, pred_start)
                        inter_end = min(net, pred_end)
                        inter = max(0, inter_end - inter_start)
                        union = max(net, pred_end) - min(nst, pred_start)
                        tiou = inter / union if union > 0 else 0.0

                        best_iou = max(best_iou, tiou)

                # print(best_iou)
                avg_tiou.append(best_iou)
                r1_01 += best_iou >= 0.1
                r1_05 += best_iou >= 0.5

            # tqdm.write(f"Current Average Temporal IoU: {np.mean(avg_temporal_iou):.4f}")
            pbar.set_description(f"Processing videos... (Avg tIoU: {np.mean(avg_tiou):.8f}, R@1 (IoU=0.1): {100 * r1_01 / len(avg_tiou):.2f}), R@1 (IoU=0.5): {100 * r1_05 / len(avg_tiou):.2f}).")

        except Exception as e:  # pylint: disable=broad-except
            broken_samples += 1
            print("Error processing video %s: %s", video.clip_uid, e)
            continue

    print(f"R@1 (IoU=0.1): {100 * r1_01 / len(avg_tiou):.2f}, R@1 (IoU=0.5): {100 * r1_05 / len(avg_tiou):.2f}")
    print(f"Number of broken samples during evalution: {broken_samples}.")
