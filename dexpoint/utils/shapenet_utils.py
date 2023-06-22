import json
from pathlib import Path

import numpy as np
import sapien.core as sapien

SHAPENET_CAT = ["02876657", "02946921", "03797390"]


def get_shapenet_root_dir():
    current_dir = Path(__file__).parent
    shapenet_dir = current_dir.parent.parent / "assets" / "shapenet"
    return shapenet_dir.resolve()


def load_shapenet_object_list():
    cat_dict = {}
    info_path = get_shapenet_root_dir() / "info.json"
    with info_path.open("r") as f:
        cat_scale = json.load(f)
    for cat in SHAPENET_CAT:
        object_list_file = get_shapenet_root_dir() / f"{cat}.txt"
        with object_list_file.open("r") as f:
            cat_object_list = f.read().split("\n")
        cat_dict[cat] = {}
        for model_id in cat_object_list:
            if len(model_id) > 0:
                cat_dict[cat][model_id] = cat_scale[cat][model_id]
        cat_dict[cat]["train"] = cat_object_list[:10]
        cat_dict[cat]["eval"] = cat_object_list[10:]

    return cat_dict


CAT_DICT = load_shapenet_object_list()


def load_shapenet_object(
        scene: sapien.Scene,
        cat_id: str,
        model_id: str,
        physical_material: sapien.PhysicalMaterial = None,
        density=1000,
        visual_only=False
):
    builder = scene.create_actor_builder()

    if physical_material is None:
        physical_material = scene.engine.create_physical_material(1.5, 1, 0.01)
    shapenet_dir = get_shapenet_root_dir()
    collision_file = str(shapenet_dir / cat_id / model_id / "convex.obj")
    visual_file = str(shapenet_dir / cat_id / model_id / "model.obj")
    info = CAT_DICT[cat_id][model_id]
    scale = info["scales"]
    height = info["height"]
    if not visual_only:
        builder.add_multiple_collisions_from_file(
            filename=collision_file,
            scale=np.array(scale * 3),
            material=physical_material,
            density=density,
            pose=sapien.Pose(q=np.array([0.7071, 0.7071, 0, 0]))
        )

    builder.add_visual_from_file(filename=visual_file, scale=np.array(scale * 3),
                                 pose=sapien.Pose(q=np.array([0.7071, 0.7071, 0, 0])))

    if not visual_only:
        actor = builder.build(name=model_id)
    else:
        actor = builder.build_static(name=model_id)
    return actor, height
