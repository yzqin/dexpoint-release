import os
from time import time

import numpy as np

from dexpoint.env.rl_env.relocate_env import AllegroRelocateRLEnv
from dexpoint.real_world import task_setting

from simple_pc import SimplePointCloud
import time as t

if __name__ == '__main__':
    def create_env_fn():
        object_names = ["mustard_bottle", "tomato_soup_can", "potted_meat_can"]
        object_name = np.random.choice(object_names)
        rotation_reward_weight = 0  # whether to match the orientation of the goal pose
        use_visual_obs = True
        env_params = dict(robot_name="xarm7_allegro_v2",object_name=object_name, rotation_reward_weight=rotation_reward_weight,
                          randomness_scale=1, use_visual_obs=use_visual_obs, use_gui=True,
                          no_rgb=True)

        # If a computing device is provided, designate the rendering device.
        # On a multi-GPU machine, this sets the rendering GPU and RL training GPU to be the same,
        # based on "CUDA_VISIBLE_DEVICES".
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env_params["device"] = "cuda"
        environment = AllegroRelocateRLEnv(**env_params)

        # Create camera
        environment.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])

        # Specify observation
        environment.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate_noise"])

        # Specify imagination
        environment.setup_imagination_config(task_setting.IMG_CONFIG["relocate_robot_only"])
        return environment


    env = create_env_fn()
    print("Observation space:")
    print(env.observation_space)
    print("Action space:")
    print(env.action_space)

    obs = env.reset()
    simple_pc = SimplePointCloud()

    tic = time()
    rl_steps = 10000
    from EigenGrasp.eigengrasp import EigenGrasp

    executor = EigenGrasp(16,7).load_from_file("/home/wyk/Dex/dexpoint-v2/EigenGrasp/grasp_model.pkl")
    feature=[6,0,0,0,0,0,0]
    grasp_action = executor.compute_grasp(feature)
    #grasp_action=executor._pca.components_[]
    print('grasp_action:',grasp_action)
    action=np.zeros(22)
    action[-16:] = grasp_action
    for i in range(200):
        #action = np.zeros(env.action_space.shape)
        # action[0] = 0.0  # Moving forward ee link in x-axis
        obs, reward, done, info = env.step(action)
        simple_pc.render(obs,is_imitation=True)
        env.render()
        # t.sleep(5)
    for i in range(rl_steps):
        #action = np.zeros(env.action_space.shape)

        # hand_action_res = [-0.0,-0.707,-0.767,-0.695,-0.0,-0.707,-0.767 ,-0.695 , -0.0,-0.707,-0.767,-0.695,-0.829,-0.529,-0.0,-0.778]
        # hand_action_res = [-0.0,-0.78539815,-0.78539815,-0.78539815,-0.0,-0.78539815,-0.78539815 ,-0.78539815 , -0.0,-0.78539815,-0.78539815,-0.78539815,-0.78539815,-0.78539815,-0.0,-0.78539815]
        # for i in range(16):
        #     # action[i+6] += hand_action_res[i] + 1
        #     action[i+6] +=  1
        
        # action[0] = 1  # Moving forward ee link in x-axis
        # action[6:] = [-0.0,-0.78539815,-0.78539815,-0.78539815,-0.0,-0.78539815,-0.78539815 ,-0.78539815 , -0.0,-0.78539815,-0.78539815,-0.78539815,-0.78539815,-0.78539815,-0.78539815,-0.78539815]


        #action[20] = 0.5 # Moving forward ee link in x-axis
        obs, reward, done, info = env.step(action)
        # print(np.round(obs['state'][7:23], 3))
        simple_pc.render(obs,is_imitation=True)
        t.sleep(0.05)
        
        # print(obs)
        env.render()
    elapsed_time = time() - tic

    simulation_steps = rl_steps * env.frame_skip
    print(f"Single process for point-cloud environment with {rl_steps} RL steps "
          f"(= {simulation_steps} simulation steps) takes {elapsed_time}s.")
    print("Keep in mind that using multiple processes during RL training can significantly increase the speed.")
    env.scene = None
