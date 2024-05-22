
scp -P22 -r ./assets/checkpoints/20240506-1651 jzy@192.168.1.192:/home/jzy/study/ws/dexpoint-release/assets/checkpoints/

scp -P22 -r ./assets/checkpoints/20240511-1010 jzy@192.168.1.192:/home/jzy/study/ws/dexpoint-release/assets/checkpoints/
scp -P22 -r /home/jzy/work/IsaacGymEnvs/isaacgymenvs/runs/AllegroKukaLSTMPPO_07-17-29-21 jzy@192.168.1.192:/home/jzy/study/github_program/IsaacGymEnvs/isaacgymenvs/runs/

CUDA_VISIBLE_DEVICES=2 python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/pointnext-s.yaml

CUDA_VISIBLE_DEVICES=2 python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/pointnext-s.yaml mode=test --pretrained_path log/shapenetpart/shapenetpart-train-pointnext-s-ngpus1-seed46-20240417-103244-jBhJjAgXZCQZKBZPNmfxEA/checkpoint/shapenetpart-train-pointnext-s-ngpus1-seed46-20240417-103244-jBhJjAgXZCQZKBZPNmfxEA_ckpt_best.pth

mamba activate dexpoint

mamba activate openpoints

swanlab watch -l ./logs

CUDA_VISIBLE_DEVICES=1 python example/train.py --n 100 --workers 30 --iter 600 --lr 0.0003 --bs 500

watch -n 0.2 nvidia-smi

CUDA_VISIBLE_DEVICES=2
