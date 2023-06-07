import numpy as np
import sapien.core as sapien


def generate_lab_door(scene: sapien.Scene, renderer: sapien.VulkanRenderer, wood_friction, handle_friction):
    # builder
    builder = scene.create_articulation_builder()
    door_frame = builder.create_link_builder()
    door_frame.set_name("frame")

    # Frame parameters
    base_size = np.array([0.61, 0.91, 0.053])
    left_frame_size = np.array([0.035, 0.15, 0.8])
    right_frame_size = np.array([0.15, 0.035, 0.8])

    # Offset parameters
    handle_y_offset = 0.06
    base_y_offset = 0.24
    door_y_offset = 0.005

    # Door parameters
    door_size = np.array([0.018, 0.61, 0.71])
    mount_size = np.array([0.015, 0.1, 0.08])
    mount_height = 0.325

    # Handle parameters
    handle_cylinder_size = np.array([0.06, 0.03, 0.03])
    handle_box_size = np.array([0.015, 0.1, 0.008])

    # Frame
    # table_physics_mat = scene.create_physical_material(1.0 * wood_friction, 0.5 * wood_friction, 0.01)
    door_frame.set_name("door_frame")
    door_frame.add_box_collision(
        pose=sapien.Pose(np.array([left_frame_size[0] / 2, left_frame_size[1] / 2 + handle_y_offset + door_y_offset,
                                   left_frame_size[2] / 2 + base_size[2]])),
        half_size=left_frame_size / 2,
    )
    door_frame.add_box_collision(
        pose=sapien.Pose(np.array([right_frame_size[0] / 2, -right_frame_size[1] / 2 - door_size[1] + handle_y_offset,
                                   right_frame_size[2] / 2 + base_size[2]])),
        half_size=right_frame_size / 2,
    )
    door_frame.add_box_collision(
        pose=sapien.Pose(np.array([0, -base_y_offset, base_size[2] / 2])),
        half_size=base_size / 2,
    )

    door = builder.create_link_builder(door_frame)
    door.set_name("door_board")
    joint_y_offset = door_size[1] - handle_y_offset
    door.set_joint_properties(
        joint_type="revolute",
        limits=np.array([[0, np.pi / 2]]),
        pose_in_parent=sapien.Pose(p=np.array([0, -joint_y_offset, 0]), q=np.array([0.707, 0, -0.707, 0])),
        pose_in_child=sapien.Pose(p=np.array([0, -joint_y_offset, 0]), q=np.array([0.707, 0, -0.707, 0])),
        friction=0.1,
    )
    # Door
    door.add_box_collision(
        pose=sapien.Pose(
            np.array([door_size[0] / 2, -door_size[1] / 2 + handle_y_offset, door_size[2] / 2 + base_size[2] / 2])),
        half_size=door_size / 2,
    )
    # Door handle mount
    door.add_box_collision(
        pose=sapien.Pose(np.array([-mount_size[0] / 2, 0, mount_size[2] / 2 + base_size[2] + mount_height])),
        half_size=mount_size / 2,
    )

    # Door handle
    handle_pose = sapien.Pose(
        np.array([-mount_size[0] - handle_cylinder_size[0] / 2, 0, mount_size[2] / 2 + base_size[2] + mount_height]))
    handle = builder.create_link_builder(door)
    handle.set_joint_properties(
        joint_type="revolute",
        limits=np.array([[0, np.pi / 2]]),
        pose_in_child=handle_pose,
        pose_in_parent=handle_pose,
        friction=0.1,
    )
    handle.add_capsule_collision(
        pose=handle_pose,
        half_length=handle_cylinder_size[0] / 2 - 0.005,
        radius=handle_cylinder_size[1] / 2,
    )
    handle.add_box_collision(
        pose=sapien.Pose(np.array(
            [-mount_size[0] - handle_cylinder_size[0] + handle_box_size[0] / 2, -handle_box_size[1] / 2,
             mount_size[2] / 2 + base_size[2] + mount_height])),
        half_size=handle_box_size / 2,
    )

    door_articulation = builder.build(fix_root_link=True)
    door_joint, handle_joint = door_articulation.get_active_joints()[:2]
    # door_joint.set_drive_property(0, 1, 5)
    handle_joint.set_drive_property(10, 1, 10)
    return door_articulation
