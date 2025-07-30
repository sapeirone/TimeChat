# ICL using TimeChat for Action Recognition on Ego4D dataset

import argparse
import os.path as osp
import random
import warnings

from tqdm.auto import tqdm

from eval_utils import build_model
from ego4d.dataset.ar import ARDataset
from timechat.conversation.conversation_video import Chat, conv_llava_llama_2

import spacy

nlp = spacy.load("en_core_web_sm")

warnings.filterwarnings("ignore", category=UserWarning)


SYSTEM_PROMPT = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."

TASK_DESCRIPTION = (
    "You are asked to determine the action being performed in a video, "
    + "in terms of a verb and noun pair. The format should only be '(verb, noun)'. "
    + "Use only verbs from {} and nouns from {}. For example, '(wash, dish)'."
)

EXAMPLES_PROMPT = "Look at the following examples: "

PROMPT = "Describe the action shown in the video with a (verb, noun) pair: "


def ask(video_uid, timechat_model, timechat_vis_processor, verbs_list, nouns_list, num_frames=8, n_icl: int = 0, icl_clips = [], data_path: str = "ego4d_hoi_trimmed_videos/ar"):
    """Ask the TimeChat model about AR samples and return the raw unparsed response of the llm."""
    chat = Chat(timechat_model, timechat_vis_processor, device="cuda")

    frames = []
    state = conv_llava_llama_2.copy()
    state.system = SYSTEM_PROMPT

    # Description of the task
    chat.ask(TASK_DESCRIPTION.format(verbs_list, nouns_list), state)

    if n_icl > 0:
        # For PNR we only have positive prompts
        chat.ask(EXAMPLES_PROMPT, state)

        for icl_sample in random.sample(icl_clips, min(n_icl, len(icl_clips))):
            path = osp.join(data_path, icl_sample.clip_uid + ".mp4")
            chat.upload_video_without_audio(video_path=path, conv=state, img_list=frames, n_frms=num_frames)

            chat.ask(PROMPT, state, role="USER")
            chat.ask(f"({icl_sample.verb}, {icl_sample.noun})", state, role="USER")

    # Feed the sample and ask the question
    path = osp.join(data_path, video_uid + ".mp4")
    chat.upload_video_without_audio(video_path=path, conv=state, img_list=frames, n_frms=num_frames)
    chat.ask(PROMPT, state, role="USER")

    return chat.answer(conv=state, img_list=frames, num_beams=1, temperature=1.0, max_length=4096, max_new_tokens=256)[0]


if __name__ == "__main__":
    # Example usage

    args = argparse.ArgumentParser(description="Ego4D AR ICL Demo")
    args.add_argument("--ann-path", type=str, default="ego4d/annotations/v1/")
    args.add_argument("--timechat-ckpt", type=str, default="ckpt/timechat/timechat_7b.pth")
    args.add_argument("--num-frames", type=int, default=8, help="Number of frames to sample from the video.")
    args.add_argument("--icl-examples", type=int, default=0, help="Number of in-context learning examples to use (0 means no ICL samples).")
    args.add_argument("--video-path", type=str, default="ego4d_hoi_trimmed_videos/ar", help="Processed video path to use for the Ego4D dataset.")

    args = args.parse_args()

    print("\n")
    print("###########################")
    print(f"Using {args.num_frames} frames and {args.icl_examples} ICL examples.")
    print(f"Video path: {args.video_path}")
    print(f"TimeChat ckpt: {args.timechat_ckpt}")
    print("###########################")
    print("\n")

    # Build the TimeChat model
    model, vis_processor = build_model(ckpt=args.timechat_ckpt, long_context=True)

    # Action Recognition dataset
    print("Loading AR dataset...")
    dset_train = ARDataset(split="train", root=args.ann_path)
    dset_val = ARDataset(split="val", root=args.ann_path)

    print(f"Loaded {len(dset_val)} validation samples.")

    # Collect here the number of correct verb and noun predictions
    verbs_correct, nouns_correct = [], []

    # Keep track of the number of samples for which it was not possible to compute the metric
    broken_samples = 0
    
    samples = list(dset_val)

    pbar = tqdm(samples, total=len(samples), desc="Processing AR videos...")
    for sample in pbar:

        try:
            response = ask(
                video_uid=sample.clip_uid,
                timechat_model=model,
                timechat_vis_processor=vis_processor,
                num_frames=args.num_frames,
                verbs_list=dset_val.verb_labels,
                nouns_list=dset_val.noun_labels,
                data_path=args.video_path,
                n_icl=args.icl_examples,
                icl_clips=list(dset_train)
            )

            verb, noun = None, None
            for token in nlp(response):
                if token.pos_ == "VERB" and verb is None and token.lemma_ in dset_val.verb_labels:
                    verb = token.lemma_
                if token.pos_ == "NOUN" and noun is None and token.text in dset_val.noun_labels:
                    noun = token.text
                    
            verbs_correct.append(verb is not None and verb.lower() == sample.verb)
            nouns_correct.append(noun is not None and noun.lower() == sample.noun)

            verbs_acc = 100 * sum(verbs_correct) / len(verbs_correct)
            nouns_acc = 100 * sum(nouns_correct) / len(nouns_correct)
            pbar.set_description(f"Verbs acc: {verbs_acc:.2f}, Nouns acc: {nouns_acc:.2f}.")

        except Exception as e:  # pylint: disable=broad-except
            broken_samples += 1
            print("Error processing video %s: %s", sample.clip_uid, e)
            continue

    verbs_acc = 100 * sum(verbs_correct) / len(verbs_correct)
    nouns_acc = 100 * sum(nouns_correct) / len(nouns_correct)
    print(f"Verbs acc: {verbs_acc:.2f}, Nouns acc: {nouns_acc:.2f}.")
    print(f"Number of broken samples during evalution: {broken_samples}.")
