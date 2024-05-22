import swanlab
import random

class SwanLog():
    def __init__(robot_name,):
        run = swanlab.init(
            experiment_name="dexpoint" + robot_name,
            description="dexpoint实验",
            config={
                "learning_rate": 0.01,
                "epochs": 20,
            },
            logdir="./logs"
        )

    def log(self, loss , acc):
        swanlab.log({"loss": loss, "accuracy": acc})
