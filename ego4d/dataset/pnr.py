"""
Dataset for Ego4d PNR.
"""

import json
import logging
import os.path as osp
import warnings
from typing import List, Literal

from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Sample:
    """Ego4d PNR sample."""

    def __init__(self, video_uid: str, clip_uid: str, video_start_frame: int, video_end_frame: int, pnr_frame: int):
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
        pnr_frame : int
            pnr frame of the sample
        """
        self.video_uid = video_uid
        self.clip_uid = clip_uid

        self.video_start_frame = video_start_frame
        self.video_end_frame = video_end_frame

        self.video_pnr_frame = pnr_frame

    def __str__(self):
        return (
            f"Sample(video_uid={self.video_uid}, clip_uid={self.clip_uid}, "
            f"video_start_frame={self.video_start_frame}, video_end_frame={self.video_end_frame}, "
            f"video_pnr_frame={self.video_pnr_frame})"
        )

    def generate_it_sample(self):
        """Generate instruction tuning sample"""
        rel_timestamp = (self.video_pnr_frame - self.video_start_frame) / 30.0

        return {
            "video": f"{self.clip_uid}.mp4",
            "QA": [{
                "q": "Predict the timestamp in seconds of the object state change in the given video.", 
                "a": f"The state change happens at {rel_timestamp:.1f} seconds."
            }],
            "source": "ego4d_pnr",
        }


class PNRDataset(Dataset):
    """OSCC dataset for the Ego4D dataset."""

    def __init__(self, split: Literal["train", "val"], root: str = "../../data/ego4d/raw/annotations/v1/"):
        # Initialize the dataset
        super().__init__()

        self.samples: List[Sample] = self._load_annotations(split, root)

    def _load_annotations(self, split: str, root: str) -> List[Sample]:
        """Load annotations."""
        annotations_path = osp.join(f"{root}/fho_oscc-pnr_{split}.json")
        if not osp.exists(annotations_path):
            raise FileNotFoundError(f"Could not find the OSCC annotations file {annotations_path}.")

        annotations = json.load(open(annotations_path, "r"))

        samples: List[Sample] = []
        for sample in annotations["clips"]:
            state_change = bool(int(sample["state_change"]) if "state_change" in sample else 0)

            if not state_change:
                continue

            unique_id = sample["unique_id"]
            video_uid = sample["video_uid"]

            sf = sample["parent_start_frame"]
            ef = sample["parent_end_frame"]

            pnr_frame = sample["parent_pnr_frame"]

            samples.append(Sample(video_uid, unique_id, sf, ef, pnr_frame))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]
