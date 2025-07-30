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

TASK_DESCRIPTION = (
    "You are asked to determine whether a video contains an object state change. "
    + "This could be, for example, an apple being sliced or a glass being filled. "
    + "If there is a visible transformation in the object's physical state, answer 'Object state changed'. "
    + "Otherwise, answer 'Object state not changed'."
)

EXAMPLES_PROMPT = "Look at the following examples: "

PROMPT = "Is there an object state change?"


def ask(video_uid: str, positive_clips, negative_clips, timechat_model, vis_processor, n_frames: int = 8, n_icl: int = 0, data_path: str = "ego4d_hoi_trimmed_videos/oscc"):
    """Ask the TimeChat model about OSCC samples and return the raw unparsed response of the llm."""
    chat = Chat(timechat_model, vis_processor, device="cuda")

    frames = []
    state = conv_llava_llama_2.copy()
    state.system = SYSTEM_PROMPT

    # Description of the task
    chat.ask(TASK_DESCRIPTION, state)

    if n_icl > 0:
        # Prompt the model with positive and negative examples
        chat.ask(EXAMPLES_PROMPT, state)

        positive_prompts = random.sample(positive_clips, min(n_icl, len(positive_clips)))
        negative_prompts = random.sample(negative_clips, min(n_icl, len(negative_clips)))
        for pos, neg in zip(positive_prompts, negative_prompts):

            try:
                # Negative example
                path = osp.join(data_path, neg.clip_uid + ".mp4")
                chat.upload_video_without_audio(video_path=path, conv=state, img_list=frames, n_frms=n_frames)
                chat.ask(PROMPT, state, role="USER")
                chat.ask("Object state not changed.", state, role="USER")
            except Exception:  # pylint: disable=broad-except
                print(f"Failed to load negative prompt with clip uid {neg.clip_uid}...")

            try:
                # Negative example
                path = osp.join(data_path, pos.clip_uid + ".mp4")
                chat.upload_video_without_audio(video_path=path, conv=state, img_list=frames, n_frms=n_frames)
                chat.ask(PROMPT, state, role="USER")
                chat.ask("Object state changed.", state, role="USER")
            except Exception:  # pylint: disable=broad-except
                print(f"Failed to load negative prompt with clip uid {pos.clip_uid}...")

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

    args = argparse.ArgumentParser(description="Ego4D OSCC ICL Eval")
    args.add_argument("--ann-path", type=str, default="ego4d/annotations/v1/")
    args.add_argument("--timechat-ckpt", type=str, default="ckpt/timechat/timechat_7b.pth")
    args.add_argument("--num-frames", type=int, default=8, help="Number of frames to sample from the video.")
    args.add_argument("--icl-examples", type=int, default=0, help="Number of in-context learning examples to use (0 means no ICL samples).")
    args.add_argument("--video-path", type=str, default="ego4d_hoi_trimmed_videos/oscc", help="Processed video path to use for the Ego4D dataset.")

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

    # Object State Change Classification (OSCC) dataset
    print("Loading OSCC dataset...")
    dset_train = OSCCDataset(split="train", root=args.ann_path)
    dset_val = OSCCDataset(split="val", root=args.ann_path)

    print(f"Loaded {len(dset_train)} training samples and {len(dset_val)} validation samples.")

    # Collect here the number of correct predictions
    correct, n = 0, 0
    # And the number of yes/no predictions
    n_yes, n_no = 0, 0
    
    # Keep track of the number of samples for which it was not possible to compute the metric
    broken_samples = 0

    pbar = tqdm(dset_val, total=len(dset_val), desc="Processing videos...")
    for sample in pbar:

        try:
            response = ask(
                video_uid=sample.clip_uid,
                negative_clips=[x for x in dset_train if x.label == 0],
                positive_clips=[x for x in dset_train if x.label == 1],
                timechat_model=model,
                vis_processor=vis_processor,
                n_frames=args.num_frames,
                n_icl=args.icl_examples,
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
