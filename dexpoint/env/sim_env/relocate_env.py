import numpy as np
import sapien.core as sapien

from dexpoint.env.sim_env.base import BaseSimulationEnv
from dexpoint.real_world import lab
from dexpoint.utils.egad_object_utils import load_egad_object, EGAD_NAME
from dexpoint.utils.render_scene_utils import set_entity_color
from dexpoint.utils.shapenet_utils import load_shapenet_object, SHAPENET_CAT, CAT_DICT
from dexpoint.utils.ycb_object_utils import load_ycb_object, YCB_SIZE, YCB_ORIENTATION


class LabRelocateEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=10, object_category="YCB", object_name="tomato_soup_can",
                 randomness_scale=1, friction=1, use_visual_obs=False, **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, use_visual_obs=use_visual_obs, **renderer_kwargs)

        # Object info
        self.object_category = object_category
        self.object_name = object_name
        self.object_scale = 1
        self.target_pose = sapien.Pose()

        # Dynamics info
        self.randomness_scale = randomness_scale
        self.friction = friction

        # Construct scene
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(0.005)

        # Dummy camera creation to initial geometry object
        if self.renderer and not self.no_rgb:
            cam = self.scene.add_camera("init_not_used", width=10, height=10, fovy=1, near=0.1, far=1)
            self.scene.remove_camera(cam)

        # Load table
        self.tables = self.create_lab_tables(table_height=0.6)

        # Load object
        self.manipulated_object, self.target_object, self.object_height = self.load_object(object_name)

    def load_object(self, object_name):
        if self.object_category.lower() == "ycb":
            manipulated_object = load_ycb_object(self.scene, object_name)
            target_object = load_ycb_object(self.scene, object_name, visual_only=True)
            target_object.set_name("target_object")
            object_height = YCB_SIZE[self.object_name][2] / 2
        elif self.object_category.lower() == "egad":
            if self.object_name == "any_eval":
                names = EGAD_NAME["eval"]
                object_name = self.np_random.choice(names)
            elif self.object_name == "any_train":
                names = EGAD_NAME["train"]
                object_name = self.np_random.choice(names)
            manipulated_object = load_egad_object(self.scene, model_id=object_name)
            target_object = load_egad_object(self.scene, model_id=object_name, visual_only=True)
            target_object.set_name("target_object")
            object_height = 0.04
        elif self.object_category.isnumeric():
            if self.object_category not in SHAPENET_CAT:
                raise ValueError(f"Object category not recognized: {self.object_category}")
            if self.object_name == "any_eval":
                names = CAT_DICT[self.object_category]["eval"]
                object_name = self.np_random.choice(names)
            if self.object_name == "any_train":
                names = CAT_DICT[self.object_category]["train"]
                object_name = self.np_random.choice(names)
            manipulated_object, object_height = load_shapenet_object(self.scene, cat_id=self.object_category,
                                                                     model_id=object_name)
            target_object, _ = load_shapenet_object(self.scene, cat_id=self.object_category, model_id=object_name,
                                                    visual_only=True)
            target_object.set_name("target_object")
        else:
            raise NotImplementedError
        if self.use_visual_obs:
            target_object.hide_visual()
        if self.renderer and not self.no_rgb:
            set_entity_color([target_object], [0, 1, 0, 0.6])
        return manipulated_object, target_object, object_height

    def generate_random_object_pose(self, randomness_scale):
        pos = self.np_random.uniform(low=-0.1, high=0.1, size=2) * randomness_scale
        if self.object_category == "ycb":
            orientation = YCB_ORIENTATION[self.object_name]
        else:
            orientation = np.array([1, 0, 0, 0])
        position = np.array([pos[0], pos[1], self.object_height])
        pose = sapien.Pose(position, orientation)
        return pose

    def generate_random_target_pose(self, randomness_scale):
        pos = self.np_random.uniform(low=-0.2, high=0.2, size=2) * randomness_scale
        height = 0.25
        position = np.array([pos[0], pos[1], height])
        # No randomness for the orientation. Keep the canonical orientation.
        if self.object_category == "ycb":
            orientation = YCB_ORIENTATION[self.object_name]
        else:
            orientation = np.array([1, 0, 0, 0])
        pose = sapien.Pose(position, orientation)
        return pose

    def reset_env(self):
        if "any" in self.object_name:
            self.scene.remove_actor(self.manipulated_object)
            self.scene.remove_actor(self.target_object)
            self.manipulated_object, self.target_object, self.object_height = self.load_object(self.object_name)

        pose = self.generate_random_object_pose(self.randomness_scale)
        self.manipulated_object.set_pose(pose)

        # Target pose
        pose = self.generate_random_target_pose(self.randomness_scale)
        self.target_object.set_pose(pose)
        self.target_pose = pose

        if self.object_category == "egad":
            for _ in range(100):
                self.scene.step()
            self.object_height = self.manipulated_object.get_pose().p[2]

    def create_lab_tables(self, table_height):
        # Build object table first
        builder = self.scene.create_actor_builder()
        table_thickness = 0.03

        # Top
        top_pose = sapien.Pose(np.array([lab.TABLE_ORIGIN[0], lab.TABLE_ORIGIN[1], -table_thickness / 2]))
        top_material = self.scene.create_physical_material(1, 0.5, 0.01)
        table_half_size = np.concatenate([lab.TABLE_XY_SIZE / 2, [table_thickness / 2]])
        builder.add_box_collision(pose=top_pose, half_size=table_half_size, material=top_material)
        # Leg
        if self.renderer and not self.no_rgb:
            table_visual_material = self.renderer.create_material()
            table_visual_material.set_metallic(0.0)
            table_visual_material.set_specular(0.3)
            table_visual_material.set_base_color(np.array([0.9, 0.9, 0.9, 1]))
            table_visual_material.set_roughness(0.3)

            leg_size = np.array([0.025, 0.025, (table_height / 2 - table_half_size[2])])
            leg_height = -table_height / 2 - table_half_size[2]
            x = table_half_size[0] - 0.1
            y = table_half_size[1] - 0.1

            builder.add_box_visual(pose=top_pose, half_size=table_half_size, material=table_visual_material)
            builder.add_box_visual(pose=sapien.Pose([x, y + lab.TABLE_ORIGIN[1], leg_height]), half_size=leg_size,
                                   material=table_visual_material, name="leg0")
            builder.add_box_visual(pose=sapien.Pose([x, -y + lab.TABLE_ORIGIN[1], leg_height]), half_size=leg_size,
                                   material=table_visual_material, name="leg1")
            builder.add_box_visual(pose=sapien.Pose([-x, y + lab.TABLE_ORIGIN[1], leg_height]), half_size=leg_size,
                                   material=table_visual_material, name="leg2")
            builder.add_box_visual(pose=sapien.Pose([-x, -y + lab.TABLE_ORIGIN[1], leg_height]), half_size=leg_size,
                                   material=table_visual_material, name="leg3")
        object_table = builder.build_static("object_table")

        # Build robot table
        table_half_size = np.array([0.3, 0.8, table_thickness / 2])
        robot_table_offset = -lab.DESK2ROBOT_Z_AXIS - 0.004
        table_height += robot_table_offset
        builder = self.scene.create_actor_builder()
        top_pose = sapien.Pose(
            np.array([lab.ROBOT2BASE.p[0] - table_half_size[0] + 0.08,
                      lab.ROBOT2BASE.p[1] - table_half_size[1] + 0.08,
                      -table_thickness / 2 + robot_table_offset]))
        top_material = self.scene.create_physical_material(1, 0.5, 0.01)
        builder.add_box_collision(pose=top_pose, half_size=table_half_size, material=top_material)
        if self.renderer and not self.no_rgb:
            table_visual_material = self.renderer.create_material()
            table_visual_material.set_metallic(0.0)
            table_visual_material.set_specular(0.5)
            table_visual_material.set_base_color(np.array([239, 212, 151, 255]) / 255)
            table_visual_material.set_roughness(0.1)
            builder.add_box_visual(pose=top_pose, half_size=table_half_size, material=table_visual_material)
        robot_table = builder.build_static("robot_table")
        return object_table, robot_table


def env_test():
    from sapien.utils import Viewer
    from constructor import add_default_scene_light
    env = LabRelocateEnv(object_category="02876657", object_name="9dff3d09b297cdd930612f5c0ef21eb8")
    viewer = Viewer(env.renderer)
    viewer.set_scene(env.scene)
    add_default_scene_light(env.scene, env.renderer)
    env.viewer = viewer

    env.reset_env()
    while not viewer.closed:
        env.simple_step()
        env.render()


if __name__ == '__main__':
    env_test()
