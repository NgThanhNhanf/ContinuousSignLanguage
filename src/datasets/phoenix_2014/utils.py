import os
import torch
from PIL import Image
from torchvision import transforms

def read_video(frame_folder):
    frames = []
    transform = transforms.ToTensor()

    for filename in os.listdir(frame_folder):
        img = Image.open(os.path.join(frame_folder, filename)).convert('RGB')
        frames.append(transform(img))

    return torch.stack(frames)


from pathlib import Path


import re
from pathlib import Path

def extract_frame_id(frame_path):
    match = re.search(
        r"_fn(\d+)-\d+\.png$",
        Path(frame_path).name
    )

    if match is None:
        raise ValueError(f"Invalid frame path: {frame_path}")

    return int(match.group(1))


def parse_alignment(file_path, id2gloss=None):
    segments = []
    current = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            frame_path, gloss_id = line.rsplit(maxsplit=1)

            frame_path = Path(frame_path)
            video_path = str(frame_path.parent)

            id_frame = extract_frame_id(frame_path)

            gloss = (
                id2gloss[int(gloss_id)]
                if id2gloss is not None
                else gloss_id
            )

            if current is None:
                current = {
                    "video_path": video_path,
                    "id_start_frame": id_frame,
                    "id_end_frame": id_frame,
                    "gloss": gloss
                }
                continue

            if (
                current["video_path"] == video_path
                and current["gloss"] == gloss
                and id_frame == current["id_end_frame"] + 1
            ):
                current["id_end_frame"] = id_frame
            else:
                segments.append(current)

                current = {
                    "video_path": video_path,
                    "id_start_frame": id_frame,
                    "id_end_frame": id_frame,
                    "gloss": gloss
                }

    if current is not None:
        segments.append(current)

    return segments


import pandas as pd
import re


def build_gloss_vocabulary(path):
    # Read trainingClasses.txt
    df = pd.read_csv(path, sep=r"\s+")

    state_to_gloss = {}
    gloss_list = []

    for _, row in df.iterrows():
        state = row["signstate"]
        class_id = int(row["classlabel"])

        # Garbage class
        if state == "si":
            state_to_gloss[class_id] = None
            continue

        # Remove hidden state suffix: 0, 1, 2
        match = re.match(r"^(.+)([012])$", state)

        if match is None:
            raise ValueError(f"Invalid signstate: {state}")

        gloss = match.group(1)

        state_to_gloss[class_id] = gloss

        if gloss not in gloss_list:
            gloss_list.append(gloss)

    # Build gloss vocabulary
    gloss_to_id = {
        gloss: idx
        for idx, gloss in enumerate(gloss_list)
    }

    id_to_gloss = {
        idx: gloss
        for gloss, idx in gloss_to_id.items()
    }

    # Map original class IDs to new gloss IDs
    state_id_to_gloss_id = {
        state_id: (
            gloss_to_id[gloss]
            if gloss is not None
            else None
        )
        for state_id, gloss in state_to_gloss.items()
    }

    return {
        "state_to_gloss": state_to_gloss,
        "gloss_to_id": gloss_to_id,
        "id_to_gloss": id_to_gloss,
        "state_id_to_gloss_id": state_id_to_gloss_id
    }
