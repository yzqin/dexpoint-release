import numpy as np
import open3d as o3d

class SimplePointCloud():
    def __init__(self):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("Pointcloud",640,480)
        self.pointcloud = o3d.geometry.PointCloud()
        self.geometrie_added = False
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        self.vis.add_geometry(coordinate)

    def render(self,obs,is_imitation=False):
        self.pointcloud.clear()
        pc = obs["relocate-point_cloud"]
        pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc))
        self.pointcloud += pcd
        if is_imitation:
            goal_robot = obs["imagination_robot"]
            goal_robot_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(goal_robot))
            self.pointcloud += goal_robot_pcd

        if not self.geometrie_added:
            self.vis.add_geometry(self.pointcloud)
            self.geometrie_added = True
        self.vis.update_geometry(self.pointcloud)
        self.vis.poll_events()
        self.vis.update_renderer()

