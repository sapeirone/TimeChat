# ICL using TimeChat for Action Recognition on Ego4D dataset

import argparse
import os.path as osp
import random
import warnings

from statistics import mean

from tqdm.auto import tqdm

import numpy as np
import spacy

import editdistance

from eval_utils import build_model
from ego4d.dataset.lta import LTADataset
from timechat.conversation.conversation_video import Chat, conv_llava_llama_2

nlp = spacy.load("en_core_web_sm")

warnings.filterwarnings("ignore", category=UserWarning)


SYSTEM_PROMPT = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."

PROMPT = "Given a short video segment predict the future action as (verb, noun) pairs based on the provided context."


def ask(sample, timechat_model, timechat_vis_processor, num_frames=8, data_path: str = "ego4d_hoi_trimmed_videos/lta"):
    """Ask the TimeChat model about LTA samples and return the raw unparsed response of the llm."""
    chat = Chat(timechat_model, timechat_vis_processor, device="cuda")

    frames = []
    state = conv_llava_llama_2.copy()
    state.system = SYSTEM_PROMPT

    # Feed the sample and ask the question
    path = osp.join(data_path, sample.clip_uid + ".mp4")
    chat.upload_video_without_audio(video_path=path, conv=state, img_list=frames, n_frms=num_frames)
    #chat.ask(PROMPT + "[" + ",".join(f"({verb}, {noun})" for verb, noun in zip(sample.input_verb_labels, sample.input_noun_labels)) + "] => ", state, role="USER")
    chat.ask(PROMPT, state, role="USER")

    return chat.answer(conv=state, img_list=frames, num_beams=1, temperature=0.5, max_length=3000)[0]


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

    args = argparse.ArgumentParser(description="Ego4D LTA ICL Demo")
    args.add_argument("--ann-path", type=str, default="ego4d/annotations/v1/")
    args.add_argument("--timechat-ckpt", type=str, default="ckpt/timechat/timechat_7b.pth")
    args.add_argument("--num-frames", type=int, default=8, help="Number of frames to sample from the video.")
    args.add_argument("--video-path", type=str, default="ego4d_hoi_trimmed_videos/lta", help="Processed video path to use for the Ego4D dataset.")

    args = args.parse_args()

    print("\n")
    print("###########################")
    print(f"Using {args.num_frames} frames.")
    print(f"Video path: {args.video_path}")
    print(f"TimeChat ckpt: {args.timechat_ckpt}")
    print("###########################")
    print("\n")

    # Build the TimeChat model
    model, vis_processor = build_model(ckpt=args.timechat_ckpt, long_context=True)

    # Action Recognition dataset
    print("Loading AR dataset...")
    dset_train = LTADataset(split="train", root=args.ann_path)
    dset_val = LTADataset(split="val", root=args.ann_path)

    print(f"Loaded {len(dset_val)} validation samples.")

    # Collect here the number of correct verb and noun predictions
    verbs_ed, nouns_ed = [], []

    # Keep track of the number of samples for which it was not possible to compute the metric
    broken_samples = 0

    pbar = tqdm(dset_val, total=len(dset_val), desc="Processing LTA videos...")
    for sample in pbar:

        try:
            response = ask(
                sample=sample,
                timechat_model=model,
                timechat_vis_processor=vis_processor,
                num_frames=args.num_frames,
                data_path=args.video_path,
            )

            # try to separate the response into a list of pairs
            if "(" not in response and '\n' not in response:
                verbs_ed.append(1)
                nouns_ed.append(1)
                continue

            verbs, nouns = [], []

            response = response.split(":")[1] if ":" in response else response
            for part in (response.split("(") if '(' in response else response.split('\n')):
                part = part.strip()
                if len(part) == 0:
                    continue
                
                verb, noun = None, None

                for token in nlp(part):
                    if token.pos_ == "VERB" and verb is None and token.lemma_ in dset_val.verb_labels:
                        verb = token.lemma_
                    if token.pos_ == "NOUN" and noun is None and token.text in dset_val.noun_labels:
                        noun = token.text

                verbs.append(verb)
                nouns.append(noun)

            verbs = [-1 if label is None else dset_val.verb_labels.index(label) for label in verbs]
            nouns = [-1 if label is None else dset_val.noun_labels.index(label) for label in nouns]

            verbs = np.array(verbs[: min(20, len(verbs))] + [-1] * max(0, 20 - len(verbs)))
            nouns = np.array(nouns[: min(20, len(nouns))] + [-1] * max(0, 20 - len(nouns)))

            verbs = verbs[None, :, None]
            nouns = nouns[None, :, None]

            verbs_gt = np.array([dset_val.verb_labels.index(label) for label in sample.verb_labels])
            nouns_gt = np.array([dset_val.noun_labels.index(label) for label in sample.noun_labels])

            verbs_gt = verbs_gt[None]
            nouns_gt = nouns_gt[None]

            verbs_ed.append(eval_ed(verbs, verbs_gt).item())
            nouns_ed.append(eval_ed(nouns, nouns_gt).item())

            pbar.set_description(f"Verbs ed: {mean(verbs_ed):.3f}, Nouns acc: {mean(nouns_ed):.3f}.")

        except Exception as e:  # pylint: disable=broad-except
            broken_samples += 1
            print("Error processing video %s: %s", sample.clip_uid, e)
            continue

    print(f"Verbs ed: {mean(verbs_ed):.3f}, Nouns ed: {mean(nouns_ed):.3f}.")
    print(f"Number of broken samples during evalution: {broken_samples}.")
