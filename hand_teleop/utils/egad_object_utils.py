import json
from pathlib import Path

import numpy as np
import sapien.core as sapien


def get_egad_root_dir():
    current_dir = Path(__file__).parent
    ycb_dir = current_dir.parent.parent / "assets" / "egad"
    return ycb_dir.resolve()


def load_egad_scale():
    eval_file = get_egad_root_dir() / "info_eval_v0.json"
    with eval_file.open() as f:
        egad_scale = {"eval": json.load(f)}
    return egad_scale


def load_egad_name():
    egad_dir = get_egad_root_dir()
    entities = {"eval": "egad_eval_set"}
    exclude = {"eval": ["E0", "F0", "G0", "G1"]}
    name_dict = {}
    for split, sub_dir in entities.items():
        name_dict[split] = list()
        for file in (egad_dir / sub_dir).glob("*.obj"):
            if file.stem not in exclude[split]:
                name_dict[split].append(file.stem)
    return name_dict


EGAD_SCALE = load_egad_scale()
EGAD_NAME = load_egad_name()


def load_egad_object(
        scene: sapien.Scene,
        model_id: str,
        physical_material: sapien.PhysicalMaterial = None,
        density=1000,
        visual_only=False
):
    # Source: https://github.com/haosulab/ManiSkill2022/tree/main/scripts/jigu/egad
    builder = scene.create_actor_builder()
    # A heuristic way to infer split
    split = "train" if "_" in model_id else "eval"
    if split == "eval":
        scale = EGAD_SCALE[split][model_id]["scales"]
    else:
        raise NotImplementedError

    if physical_material is None:
        physical_material = scene.engine.create_physical_material(1, 1, 0.01)
    egad_dir = get_egad_root_dir()
    if not visual_only:
        collision_file = str(egad_dir / "egad_{split}_set_vhacd" / f"{model_id}.obj").format(split=split)
        builder.add_multiple_collisions_from_file(
            filename=collision_file,
            scale=np.array(scale * 3),
            material=physical_material,
            density=density,
        )

    visual_file = str(egad_dir / "egad_{split}_set" / f"{model_id}.obj").format(
        split=split
    )
    builder.add_visual_from_file(
        filename=visual_file,
        scale=np.array(scale * 3),
    )

    if not visual_only:
        actor = builder.build(name=model_id)
    else:
        actor = builder.build_static(name=model_id)
    return actor
