# ICL using TimeChat for Object State Change Classification (OSCC) on Ego4D dataset

import os.path as osp
import random
import warnings

from tqdm.auto import tqdm

from eval_utils import build_model
from ego4d.dataset.oscc import OSCCDataset
from timechat.conversation.conversation_video import Chat, conv_llava_llama_2

warnings.filterwarnings("ignore", category=UserWarning)


SYSTEM_PROMPT = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."

PROMPT = "Tell if the given video contains an object state change and output either 'yes' or 'no'. "


def ask(video_uid: str, timechat_model, vis_processor, n_frames: int = 8, data_path: str = "ego4d_hoi_trimmed_videos/oscc"):
    """Ask the TimeChat model about OSCC samples and return the raw unparsed response of the llm."""
    chat = Chat(timechat_model, vis_processor, device="cuda")

    frames = []
    state = conv_llava_llama_2.copy()
    state.system = SYSTEM_PROMPT

    # Feed the sample and ask the question
    path = osp.join(data_path, video_uid + ".mp4")
    chat.upload_video_without_audio(video_path=path, conv=state, img_list=frames, n_frms=n_frames)
    chat.ask(PROMPT, state, role="USER")

    # Return the response of the LLM
    return chat.answer(conv=state, img_list=frames, num_beams=1, temperature=1.0, max_length=3000)[0]


if __name__ == "__main__":

    PROMPT = "Is there an object state change?"  # "Tell if the video contains an object state change and answer 'Object state changed' or 'Object state not changed'."
    # PROMPT_ICL = "Is there an object state change?"

    import argparse

    args = argparse.ArgumentParser(description="Ego4D OSCC FT Eval")
    args.add_argument("--timechat-ckpt", type=str, default="ckpt/timechat/timechat_7b.pth")
    args.add_argument("--num-frames", type=int, default=8, help="Number of frames to sample from the video.")
    args.add_argument("--video-path", type=str, default="ego4d_hoi_trimmed_videos/oscc", help="Processed video path to use for the Ego4D dataset.")

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

    # Object State Change Classification (OSCC) dataset
    print("Loading OSCC dataset...")
    dset_train = OSCCDataset(split="train")
    dset_val = OSCCDataset(split="val")

    print(f"Loaded {len(dset_train)} training samples and {len(dset_val)} validation samples.")

    # Collect here the number of correct predictions
    correct, n = 0, 0
    # And the number of yes/no predictions
    n_yes, n_no = 0, 0
    
    # Keep track of the number of samples for which it was not possible to compute the metric
    broken_samples = 0
    
    samples = list(dset_val)
    import random; random.seed(42); random.shuffle(samples)

    pbar = tqdm(samples, total=len(samples), desc="Processing videos...")
    for sample in pbar:

        try:
            response = ask(
                video_uid=sample.clip_uid,
                timechat_model=model,
                vis_processor=vis_processor,
                n_frames=args.num_frames,
                data_path=args.video_path,
            )

            # Parse the response in a quite permissive way
            response = response.strip().lower()
            response_is_positive = any(term in response.lower() for term in ["yes", "object state changed"])

            n += 1
            correct += (response_is_positive and sample.label == 1) or (not response_is_positive and sample.label == 0)
            n_yes += response_is_positive
            n_no += not response_is_positive

            pbar.set_description(f"Processing videos... (Acc.: {correct / n:.2f} (n_yes={n_yes}, n_no={n_no}).")

        except Exception as e:  # pylint: disable=broad-except
            broken_samples += 1
            print("Error processing video %s: %s", sample.clip_uid, e)
            continue

    print(f"Accuracy: {100 * correct / n:.4f} ({n} samples).")
    print(f"Number of broken samples during evalution: {broken_samples}.")
