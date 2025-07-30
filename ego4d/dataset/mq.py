"""
Dataset for Ego4d MQ.
"""

import json
import logging
import os.path as osp
from dataclasses import dataclass
from typing import List, Literal

import pandas as pd
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class Segment:
    video_start_frame: int
    video_end_frame: int

    label: str
    label_id: int


class Sample:
    """Ego4d MQ sample"""

    def __init__(self, video_uid: str, clip_uid: str, video_start_frame: int, video_end_frame: int, segments: List[Segment]):
        """Ego4d PNR sample.

        Parameters
        ----------
        video_uid : str
            uid of the video
        clip_uid : str
            uid of the clip
        video_start_frame : int
            start frame of the sample
        video_end_frame : int
            end frame of the sample
        """
        self.video_uid: str = video_uid
        self.clip_uid: str = clip_uid

        self.video_start_frame: int = video_start_frame
        self.video_end_frame: int = video_end_frame

        self.segments: List[Segment] = segments

    def __str__(self):
        segments_str = ", ".join(f"[{s.video_start_frame}-{s.video_end_frame}: {s.label} (id={s.label_id})]" for s in self.segments)
        return f"Sample(video_uid={self.video_uid}, clip_uid={self.clip_uid}, video_start_frame={self.video_start_frame}, video_end_frame={self.video_end_frame}, segments=[{segments_str}])"

    def generate_it_sample(self):
        """Generate instruction tuning sample"""

        segments = [((s.video_start_frame - self.video_start_frame) / 30.0, (s.video_end_frame - self.video_start_frame) / 30.0, s.label) for s in self.segments]
        unique_labels = list(set(s.label for s in self.segments))

        return [
            {
                "video": f"mq/{self.clip_uid}.mp4",
                "length": (self.video_end_frame - self.video_start_frame) / 30.0,
                "QA": [
                    {
                        "q": f"Find all the segments that corresponds to the textual query '{unique_label}' and determine their start and end seconds.",
                        "a": " ".join(f"{ss:.1f} - {es:.1f} seconds." for ss, es, label in segments if label == unique_label),
                    }
                ],
                "source": "ego4d_mq",
            }
            for unique_label in unique_labels
        ]


class MQDataset(Dataset):

    def __init__(self, split: Literal["train", "val"], root: str = "../../data/ego4d/raw/annotations/v1/"):
        # Initialize the dataset

        self.split = split

        labels_path = osp.join(root, "..", "mq_labels.tsv")
        labels = pd.read_csv(labels_path, sep="\t", header=None, names=["id", "label"])
        labels = labels.sort_values(by="id", ascending=True)
        self.labels = [x.strip() for x in labels.label.values]

        self.segments: List[Sample] = self._load_annotations(root)

        # Load the list of unique video ids
        self.video_uids = list(set((entry.video_uid for entry in self.segments)))
        self.clip_uids = list(set((entry.clip_uid for entry in self.segments)))

    def _load_annotations(self, ann_root: str):
        """Load annotations."""
        annotations_path = osp.join(ann_root, f"moments_{self.split}.json")
        if not osp.exists(annotations_path):
            raise FileNotFoundError(f"Could not find the MQ annotations file {annotations_path}.")

        annotations = json.load(open(annotations_path, "r"))

        mq_segments = []

        # for each video
        for video in annotations["videos"]:

            # for each clip inside the video
            for clip in video["clips"]:

                clip_segments: List[Segment] = []

                # for each annotator
                for annotator in clip["annotations"]:

                    for label in annotator["labels"]:
                        if not label["primary"]:
                            continue

                        if any(s.video_start_frame == label["video_start_frame"] and s.video_end_frame == label["video_end_frame"] and s.label == label["label"] for s in clip_segments):
                            # duplicate segment
                            continue

                        label["label"] = label["label"].lstrip('"').rstrip('"')
                        segment = Segment(label["video_start_frame"], label["video_end_frame"], label["label"], self.labels.index(label["label"]))
                        clip_segments.append(segment)

                if len(clip_segments) == 0:
                    logger.debug("No segments found for clip %s. Skipping.", clip["clip_uid"])
                    continue

                mq_entry = Sample(
                    video["video_uid"],
                    clip["clip_uid"],
                    clip["video_start_frame"],
                    clip["video_end_frame"],
                    clip_segments,
                )
                mq_segments.append(mq_entry)

        return mq_segments

    @property
    def class_labels(self) -> List[str]:
        return self.labels

    def __len__(self) -> int:
        return len(self.clip_uids)

    def __getitem__(self, idx):
        return self.segments[idx]


if __name__ == "__main__":
    mq_dataset = MQDataset(split="train")

    for sample in mq_dataset:
        print(sample.generate_it_sample())
    pass
