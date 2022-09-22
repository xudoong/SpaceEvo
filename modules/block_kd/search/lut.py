from collections import defaultdict
import os
from typing import Dict, List

import numpy as np


class LUT:

    def __init__(self, lut_dir: str):
        self.lut_dir = lut_dir
        self.lut = self._build_lut()

    def _convert_resolution(self, hw):
        hw = int(hw)
        rv_list = [224, 192, 160]
        for rv in rv_list:
            if rv % hw == 0:
                return rv 
        raise ValueError(hw)

    @property
    def resolution_list(self):
        return list(self.lut.keys())

    @property
    def avail_stage_list(self):
        return list(self.lut[self.resolution_list[0]].keys())

    def sample_block(self, stage_name, n=1, resolution=224):
        total_samples = len(self.lut[resolution][stage_name])
        idx_list = np.random.choice(total_samples, n)
        rv = []
        for i in idx_list:
            rv.append(self.lut[resolution][stage_name][i])
        return rv

    def sample_indiv(self, arch: List[int], n=1):
        stage_samples_list = []
        resolution = np.random.choice(self.resolution_list)
        for i, c in enumerate(arch, start=1):
            stage_choice = f'stage{i}_{c}'
            stage_samples = self.sample_block(stage_choice, n=n, resolution=resolution)
            stage_samples_list.append(stage_samples)  

        # sum stage latency and loss
        rv = []
        for i in range(n):
            loss = 0
            latency = 0
            for stage_idx in range(len(arch)):
                loss += stage_samples_list[stage_idx][i]['loss']
                latency += stage_samples_list[stage_idx][i]['latency']
            rv.append({'loss': loss, 'latency': latency})
        return rv     

    def _build_lut(self) -> Dict:
        rv = defaultdict(lambda: defaultdict(lambda: []))
        for file_name in os.listdir(self.lut_dir):
            if file_name.endswith('.csv'):
                stage_name = file_name.replace('.csv', '')
                with open(os.path.join(self.lut_dir, file_name), 'r') as f:
                    for line in f.readlines():
                        sample_str, input_shape, loss, flops, params, latency = line.split(',')
                        v = dict(
                            sample_str=sample_str,
                            input_shape=input_shape,
                            loss=float(loss),
                            flops=float(flops),
                            params=float(params),
                            latency=float(latency)
                        )
                        hw = int(input_shape.split('x')[-1])
                        resolution = self._convert_resolution(hw)
                        rv[resolution][stage_name].append(v)
        return rv