"""
Dataset for Ego4d LTA.
"""

import json
import os.path as osp
import logging
from typing import List, Tuple

from dataclasses import dataclass

from typing import Dict, Optional, Literal


from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class LTASegment:
    """LTA Sample."""

    def __init__(self, video_uid: str, clip_uid: str, video_start_frame: int, video_end_frame: int, verb_labels: List[int], noun_labels: List[int]):

        self.video_uid: str = video_uid
        self.clip_uid: str = clip_uid

        self.video_start_frame: int = video_start_frame
        self.video_end_frame: int = video_end_frame

        self.verb_labels: List[int] = verb_labels
        self.noun_labels: List[int] = noun_labels

    def __str__(self):
        return (
            f"LTASegment(video_uid={self.video_uid}, clip_uid={self.clip_uid}, "
            f"video_start_frame={self.video_start_frame}, video_end_frame={self.video_end_frame}, "
            f"verb_labels={self.verb_labels}, noun_labels={self.noun_labels})"
        )

    def generate_it_sample(self):
        """Generate instruction tuning sample"""
        return {
            "video": f"{self.clip_uid}.mp4",
            "QA": [
                {
                    "q": "Given a short video segment predict the future action as (verb, noun) pairs based on the provided context.",
                    "a": ", ".join(f"({verb}, {noun})" for verb, noun in zip(self.verb_labels, self.noun_labels)),
                }
            ],
            "source": "ego4d_lta",
        }


@dataclass
class Action:
    """An action in the LTA dataset."""

    idx: int  # action_idx

    video_start_frame: int
    video_end_frame: int

    verb_labels: Optional[int]
    noun_labels: Optional[int]


@dataclass
class Clip:
    """A clip in the LTA dataset, with a list of actions and textual narrations."""

    video_uid: str
    clip_uid: str

    samples: List[Action]


class LTADataset(Dataset):
    """LTA dataset for the Ego4d dataset."""

    def __init__(self, split: Literal["train", "val"], num_input_clips: int = 2, Z: int = 20, root: str = "../../data/ego4d/raw/annotations/v1/"):
        # Initialize the dataset
        super().__init__()

        self.split = split

        self.num_input_clips = num_input_clips
        self.Z = Z

        # Load LTA segments
        self.samples: List[LTASegment] = self._load_annotations(root)

        # Load the verbs and nouns taxonomy
        self.verb_labels, self.noun_labels = self._load_fho_taxonomy(root)
        self.verb_labels = [l if "_" not in l else l.split("_")[0] for l in self.verb_labels]
        self.noun_labels = [l if "_" not in l else l.split("_")[0] for l in self.noun_labels]

    def _load_fho_taxonomy(self, ann_root: str) -> Tuple[List[str], List[str]]:
        path = osp.join(ann_root, "fho_lta_taxonomy.json")

        if not osp.exists(path):
            raise FileNotFoundError(f"Could not find the FHO taxonomy at {path}")

        labels = json.load(open(path, "r"))  # {'verbs': [...], 'nouns': [...]}
        return [x.strip() for x in labels["verbs"]], [x.strip() for x in labels["nouns"]]

    def _load_annotations(self, ann_root: str) -> List[LTASegment]:
        """Load annotations."""
        annotations_path = osp.join(ann_root, f"fho_lta_{self.split}.json")
        annotations = json.load(open(annotations_path, "r"))

        self.clips: Dict[str, Clip] = {}
        for sample in annotations["clips"]:
            video_uid = sample["video_uid"]

            clip_uid = sample["clip_uid"]

            clip = self.clips.get(clip_uid, Clip(video_uid, clip_uid, []))
            self.clips[clip_uid] = clip

            clip_start_frame = sample["clip_parent_start_frame"]
            clip.samples.append(
                Action(
                    sample["action_idx"],
                    clip_start_frame + sample["action_clip_start_frame"],
                    clip_start_frame + sample["action_clip_end_frame"],
                    sample["verb"] if "_" not in sample["verb"] else sample["verb"].split("_")[0],
                    sample["noun"] if "_" not in sample["noun"] else sample["noun"].split("_")[0],
                )
            )

        # Collect all LTA segments that have at least (self.num_input_clips actions + self.Z) actions
        segments = []

        for clip_uid, clip in self.clips.items():
            clip.samples = list(sorted(clip.samples, key=lambda x: x.idx))

            for i in range(len(clip.samples) - self.num_input_clips - self.Z):
                idx_seen_start, idx_seen_end, idx_unseen_end = i, i + self.num_input_clips, i + self.num_input_clips + self.Z

                input_clips = clip.samples[idx_seen_start:idx_seen_end]
                forecast_clips = clip.samples[idx_seen_end:idx_unseen_end]

                verb_labels = [action.verb_labels for action in forecast_clips]
                noun_labels = [action.noun_labels for action in forecast_clips]

                clip_uid = f"{clip.clip_uid}_{input_clips[-1].idx}"

                segment = LTASegment(clip.video_uid, clip_uid, input_clips[0].video_start_frame, input_clips[-1].video_end_frame, verb_labels, noun_labels)  # type: ignore

                segments.append(segment)

        return segments

    @property
    def class_labels(self) -> Tuple[List[str], List[str]]:
        return (self.verb_labels, self.noun_labels)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]
