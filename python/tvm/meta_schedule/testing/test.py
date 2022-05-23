# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import logging
import tempfile
from os import path as osp
from typing import List

import numpy as np  # type: ignore
import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm.meta_schedule import TuneConfig
from tvm.meta_schedule.database import JSONDatabase
from tvm.meta_schedule.testing.relay_workload import _get_network
from tvm.meta_schedule.tune import tune_relay
from tvm.target.target import Target

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


def test_pytorch_model(
    work_dir: str,
    model_name: str,
    input_shape: List[int],
    target: str,
):
    dev = tvm.cpu() if str(target).startswith("llvm") else tvm.cuda()
    if model_name.startswith("bert"):
        data = tvm.nd.array(np.random.randint(0, 30521, size=input_shape), dev)  # embedding size
    else:
        data = tvm.nd.array(np.random.randn(*input_shape).astype("float32"), dev)

    mod, params, (input_name, _, _) = _get_network([model_name, input_shape])
    target = Target(target)

    rt_mod1: tvm.runtime.Module = tune_relay(
        mod=mod,
        params=params,
        target=target,
        config=TuneConfig(
            strategy="evolutionary",
            num_trials_per_iter=32,
            max_trials_per_task=32,
            max_trials_global=20000,
            search_strategy_config={
                "genetic_num_iters": 10,
            },
        ),
        work_dir=work_dir,
        database=JSONDatabase(
            osp.join(work_dir, "workload.json"),
            osp.join(work_dir, "records.json"),
        ),
    )
    # Compile without meta-scheduler for correctness check
    with tvm.transform.PassContext(opt_level=0):
        rt_mod2 = relay.build(mod, target=target, params=params)

    def get_output(data, lib):
        module = graph_executor.GraphModule(lib["default"](dev))
        module.set_input(input_name, data)
        module.run()
        return module.get_output(0).numpy()

    # Check correctness
    actual_output = get_output(data, rt_mod1)
    expected_output = get_output(data, rt_mod2)
    assert np.allclose(actual_output, expected_output, rtol=1e-4, atol=2e-4)


if __name__ == """__main__""":
    work_dir = "/home/zxybazh/test"
    # test_pytorch_model(work_dir, "resnet_18", [1, 3, 224, 224], "llvm --num-cores=12")
    # test_pytorch_model(work_dir, "resnet_18", [1, 3, 224, 224], "nvidia/geforce-rtx-3070")
    # test_pytorch_model(work_dir, "mobilenet_v2", [1, 3, 224, 224], "llvm --num-cores=12")
    # test_pytorch_model(work_dir, "mobilenet_v2", [1, 3, 224, 224], "nvidia/geforce-rtx-3070")
    # test_pytorch_model(work_dir, "bert_base", [1, 64], "llvm --num-cores=12")
    # test_pytorch_model(work_dir, "bert_base", [1, 64], "nvidia/geforce-rtx-3070")
    test_pytorch_model(work_dir, "mobilenet_v2", [1, 3, 224, 224], "llvm --num-cores=12")