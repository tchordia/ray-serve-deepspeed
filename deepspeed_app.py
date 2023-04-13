from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel
from ray import serve
import os
from dataclasses import dataclass
import asyncio

from pathlib import Path


import os
from argparse import ArgumentParser

import pandas as pd
import ray
import ray.util
from ray.air import Checkpoint, ScalingConfig
from ray.train.batch_predictor import BatchPredictor

import subprocess


from deepspeed_predictor import DeepSpeedPredictor, PredictionWorker, initialize_node

from dataclasses import dataclass

import os
from collections import defaultdict
from typing import List, Optional, Tuple

import pandas as pd
import ray
import ray.util
from ray.air import ScalingConfig
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

import yaml

app = FastAPI()


class Prompt(BaseModel):
    prompt: str


class Args(BaseModel):
    # bucket_uri: str = "s3://large-dl-models-mirror/models--anyscale--opt-66b-resharded/main/"
    # name: str = "facebook/opt-66b"
    # hf_home: str = "/nvme/cache"
    # checkpoint_path: str = "/nvme/model"
    bucket_uri: str
    name: str
    hf_home: str
    checkpoint_path: str
    batch_size: int = 20
    ds_inference: bool = True
    use_kernel: bool = True
    use_meta_tensor: bool = True
    num_worker_groups: int = 1
    num_gpus_per_worker_group: int = 8
    reshard_checkpoint_path: Optional[str] = None
    use_cache: bool = True

    max_new_tokens: int = 50
    max_tokens: int = 1024
    replace_method: int = False
    dtype: str = "float16"
    save_mp_checkpoint_path: Optional[str] = None


raw_args = os.getenv("APPLICATION_ARGS")
dict_args = yaml.load(raw_args, Loader=yaml.SafeLoader) if raw_args else None
args = Args.parse_obj(dict_args) if dict_args else Args()


@serve.deployment(
    route_prefix="/", num_replicas=1,
)
@serve.ingress(app)
class DeepspeedApp:
    def __init__(self, args: Args) -> None:
        self.args = args

        scaling_config = ScalingConfig(
            use_gpu=True,
            num_workers=args.num_gpus_per_worker_group,
            trainer_resources={"CPU": 0},
        )

        self.scaling_config = scaling_config
        self.init_worker_group(scaling_config)

    @app.post("/")
    async def generate_text(self, prompt: Prompt):
        return await self.generate_text_batch(prompt)

    @serve.batch(max_batch_size=args.batch_size)
    async def generate_text_batch(self, prompts: List[Prompt]):
        """Generate text from the given prompts in batch """

        print("Received prompts", prompts)
        input_column = "predict"
        #  Wrap in pandas
        data = pd.DataFrame(
            [prompt.prompt for prompt in prompts], columns=[input_column]
        )
        data_ref = ray.put(data)
        prediction = (
            await asyncio.gather(
                *[
                    worker.generate.remote(
                        data_ref,
                        column=input_column,
                        do_sample=True,
                        temperature=0.9,
                        max_length=100,
                    )
                    for worker in self.prediction_workers
                ]
            )
        )[0]
        print("Predictions", prediction)
        return prediction[: len(prompts)]

    def init_worker_group(self, scaling_config: ScalingConfig):
        """Create the worker group.

        Each worker in the group communicates with other workers through the
        torch distributed backend. The worker group is inelastic (a failure of
        one worker will destroy the entire group). Each worker in the group
        recieves the same input data and outputs the same generated text.
        """
        args = self.args

        # Start a placement group for the workers.
        self.pg = scaling_config.as_placement_group_factory().to_placement_group()
        prediction_worker_cls = PredictionWorker.options(
            num_cpus=scaling_config.num_cpus_per_worker,
            num_gpus=scaling_config.num_gpus_per_worker,
            resources=scaling_config.additional_resources_per_worker,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_capture_child_tasks=True
            ),
        )
        # Create the prediction workers.
        self.prediction_workers = [
            prediction_worker_cls.remote(args, i, scaling_config.num_workers)
            for i in range(scaling_config.num_workers)
        ]
        # Get the IPs and ports of the workers.
        self.prediction_workers_ips_ports = ray.get(
            [
                prediction_worker.get_address_and_port.remote()
                for prediction_worker in self.prediction_workers
            ]
        )
        # Rank 0 worker will be set as the master address for torch distributed.
        rank_0_ip, rank_0_port = self.prediction_workers_ips_ports[0]

        # Map from node ip to the workers on it
        ip_dict = defaultdict(list)
        for i, ip_port in enumerate(self.prediction_workers_ips_ports):
            ip_dict[ip_port[0]].append(i)

        # Configure local ranks and start the distributed backend on each worker.
        # This assumes that there cannot be a situation where 2 worker groups use the
        # same node.
        tasks = []
        for rank in range(len(self.prediction_workers)):
            worker = self.prediction_workers[rank]
            local_world_size = len(ip_dict[self.prediction_workers_ips_ports[rank][0]])
            local_rank = ip_dict[self.prediction_workers_ips_ports[rank][0]].index(rank)
            tasks.append(
                worker.init_distributed.remote(
                    local_rank, local_world_size, rank_0_ip, rank_0_port
                )
            )
        ray.get(tasks)

        # Initialize the model itself on each worker.
        ray.get([worker.init_model.remote() for worker in self.prediction_workers])


entrypoint = DeepspeedApp.bind()

# The following block will be executed if the script is run by Python directly
if __name__ == "__main__":
    serve.run(entrypoint)
