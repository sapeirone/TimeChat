"""
Evaluation code for PNR
"""

import os.path as osp
import random
import warnings

import re
import argparse

from statistics import mean

from tqdm.auto import tqdm

from eval_utils import build_model
from ego4d.dataset.pnr import PNRDataset
from timechat.conversation.conversation_video import Chat, conv_llava_llama_2

warnings.filterwarnings("ignore", category=UserWarning)


SYSTEM_PROMPT = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."

TASK_DESCRIPTION = (
    "You are given a short video of an object state change. "
    + "An object state change could be, for example, an apple being sliced or a glass being filled. "
    + "You should output the salient timestamp in seconds (just a single value) "
    + "within the boundaries of the video that corresponds to the object state change."
)

EXAMPLES_PROMPT = "Look at the following examples: "

PROMPT = "The object state change happens at "


def ask(video_uid: str, positive_clips, timechat_model, vis_processor, n_frames: int = 8, n_icl: int = 0, data_path: str = "ego4d_hoi_trimmed_videos/oscc"):
    """Ask the TimeChat model about PNR samples and return the raw unparsed response of the llm."""

    # Initialize the chat
    chat = Chat(timechat_model, vis_processor, device="cuda")

    frames = []
    state = conv_llava_llama_2.copy()
    state.system = SYSTEM_PROMPT

    # Description of the task
    chat.ask(TASK_DESCRIPTION, state, role="USER")

    if n_icl > 0:
        # For PNR we only have positive prompts
        chat.ask(EXAMPLES_PROMPT, state)
        positive_clips = positive_clips[:n_icl]

        for icl_sample in random.sample(positive_clips, min(n_icl, len(positive_clips))):
            path = osp.join(data_path, icl_sample.clip_uid + ".mp4")
            chat.upload_video_without_audio(video_path=path, conv=state, img_list=frames, n_frms=n_frames)

            chat.ask(PROMPT, state, role="USER")

            # Conver the absolute frame number to a relative timestamp within the boundaries of the video
            rel_timestamp = (icl_sample.video_pnr_frame - icl_sample.video_start_frame) / 30.0
            chat.ask(f"{rel_timestamp:.1f} seconds", state, role="USER")

    # Feed the sample and ask the question
    path = osp.join(data_path, video_uid + ".mp4")
    chat.upload_video_without_audio(video_path=path, conv=state, img_list=frames, n_frms=n_frames)
    chat.ask(PROMPT, state, role="USER")

    # Return the response of the LLM
    return chat.answer(conv=state, img_list=frames, num_beams=1, temperature=1.0, max_length=3000)[0]


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="Ego4D PNR ICL Eval")
    args.add_argument("--timechat-ckpt", type=str, default="ckpt/timechat/timechat_7b.pth")
    args.add_argument("--num-frames", type=int, default=8, help="Number of frames to sample from the video.")
    args.add_argument("--icl-examples", type=int, default=0, help="Number of in-context learning examples to use (0 means no ICL samples).")
    args.add_argument("--video-path", type=str, default="ego4d_hoi_trimmed_videos/oscc", help="Path to the Ego4d videos.")

    args = args.parse_args()

    print("\n")
    print("###########################")
    print(f"Using {args.num_frames} frames and {args.icl_examples} ICL examples.")
    print(f"Video path: {args.video_path}")
    print(f"TimeChat ckpt: {args.timechat_ckpt}")
    print("###########################")
    print("\n")

    # Build the TimeChat model
    model, vis_processor = build_model(ckpt=args.timechat_ckpt)

    # Build the PNR dataset
    print("Loading PNR dataset...")
    dset_train = PNRDataset(split="train")
    dset_val = PNRDataset(split="val")

    print(f"Loaded {len(dset_train)} training samples and {len(dset_val)} validation samples.")

    # Collect here the absolute temporal localization errors
    errors = []

    # Keep track of the number of samples for which it was not possible to
    # compute the metric
    broken_samples = 0

    pbar = tqdm(list(dset_val), total=len(dset_val), desc="Processing PNR videos...")
    for sample in pbar:

        try:
            response = ask(
                video_uid=sample.clip_uid,
                positive_clips=list(dset_train),
                timechat_model=model,
                vis_processor=vis_processor,
                n_frames=args.num_frames,
                n_icl=args.icl_examples,
                data_path=args.video_path,
            )

            # Extract the float number corresponding to the PNR timestamp from the LLM answer
            match = re.search(r"[-+]?\d*\.\d+|\d+", response)
            response = float(match.group()) if match else None

            # Compute absolute localization errors
            gt_rel_timestamp = (sample.video_pnr_frame - sample.video_start_frame) / 30.0
            errors.append(abs(gt_rel_timestamp - response))

            pbar.set_description(f"Processing PNR samples... (Err.: {mean(errors):.2f}).")

        except Exception as e:  # pylint: disable=broad-except
            broken_samples += 1
            print("Error processing video %s: %s", sample.clip_uid, e)
            continue

    print(f"Avg. localization error: {mean(errors):.2f}")
    print(f"Number of broken samples during evalution: {broken_samples}.")
