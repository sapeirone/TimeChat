"""
Dataset for Ego4d AR.
"""

import json
import logging
import os.path as osp

import warnings
from typing import List, Tuple, Literal

from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=UserWarning)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Action:
    """AR sample"""

    def __init__(self, video_uid: str, clip_uid: str, video_start_frame: int, video_end_frame: int, verb: str, noun: str):
        self.video_uid = video_uid
        self.clip_uid = clip_uid

        self.video_start_frame = video_start_frame
        self.video_end_frame = video_end_frame

        self.verb = verb
        self.noun = noun

    def __str__(self):
        return f"Action(video_uid={self.video_uid}, clip_uid={self.clip_uid}, video_start_frame={self.video_start_frame}, video_end_frame={self.video_end_frame}, verb={self.verb}, noun={self.noun})"

    def generate_it_sample(self):
        """Generate instruction tuning sample"""
        return {
            "video": f"{self.clip_uid}.mp4",
            "QA": [{"q": "Describe the action shown in the video with a (verb, noun) pair.", "a": f"({self.verb}, {self.noun})"}],
            "source": "ego4d_oscc",
        }


class ARDataset(Dataset):

    def __init__(self, split: Literal["train", "val"], num_verbs: int = 115, num_nouns: int = 478, root: str = "../../data/ego4d/raw/annotations/v1/"):
        # Initialize the dataset
        super().__init__()

        self.split = split

        self.num_verbs = num_verbs
        self.num_nouns = num_nouns

        # Load the verbs and nouns taxonomy
        self.verb_labels, self.noun_labels = self._load_fho_taxonomy()
        self.verb_labels = [l if "_" not in l else l.split("_")[0] for l in self.verb_labels]
        self.noun_labels = [l if "_" not in l else l.split("_")[0] for l in self.noun_labels]

        assert len(self.verb_labels) == self.num_verbs, "mismatch in number of verb labels and expected number of verbs"
        assert len(self.noun_labels) == self.num_nouns, "mismatch in number of noun labels and expected number of nouns"

        self.samples: List[Action] = self._load_annotations(ann_root=root)

    def _load_fho_taxonomy(self) -> Tuple[List[str], List[str]]:
        path = osp.join("../../data/ego4d/raw/annotations/v1/fho_lta_taxonomy.json")

        if not osp.exists(path):
            raise FileNotFoundError(f"Could not find the FHO taxonomy at {path}")

        labels = json.load(open(path, "r"))  # {'verbs': [...], 'nouns': [...]}
        return [x.strip() for x in labels["verbs"]], [x.strip() for x in labels["nouns"]]

    def _load_annotations(self, ann_root) -> List[Action]:
        """Load annotations."""
        annotations_path = osp.join(ann_root, f"fho_lta_{self.split}.json")
        annotations = json.load(open(annotations_path, "r"))

        actions: List[Action] = []
        for action in annotations["clips"]:
            video_uid = action["video_uid"]
            clip_uid = action["clip_uid"]

            sf = action["clip_parent_start_frame"] + action["action_clip_start_frame"]
            ef = action["clip_parent_start_frame"] + action["action_clip_end_frame"]

            verb = action["verb"] if "_" not in action["verb"] else action["verb"].split("_")[0]
            noun = action["noun"] if "_" not in action["noun"] else action["noun"].split("_")[0]

            actions.append(Action(video_uid, f"{clip_uid}_{sf}_{ef}", sf, ef, verb, noun))

        return actions

    @property
    def class_labels(self) -> Tuple[List[str], List[str]]:
        return (self.verb_labels, self.noun_labels)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]
