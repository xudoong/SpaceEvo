import copy
from json.encoder import INFINITY
from logging import warning
from typing import List

import numpy as np

from .lut import LUT


class BlockKDSearcher:

    def __init__(self, lut_dir: str, num_stages=6, num_choices_per_stage=5, num_samples_per_indiv=10000, top=20, 
                    latency_constraint=25, lat_loss_t=0.5, lat_loss_a=0.5) -> None:
        self.lut = LUT(lut_dir=lut_dir)
        self.num_stages = num_stages
        self.num_choices_per_stage = num_choices_per_stage
        self.num_samples_per_indiv = num_samples_per_indiv
        self.top = top
        if isinstance(latency_constraint, List):
            self.latency_constraint_list = latency_constraint
        else:
            self.latency_constraint_list = [latency_constraint]
        self.lat_loss_t = lat_loss_t
        self.lat_loss_a = lat_loss_a

    def sample_individual(self, arch: List[int], n=50000):
        indiv_list = []
        for _ in range(n // 500):
            indiv_list.extend(self.lut.sample_indiv(arch, n=500))
        return indiv_list

    def eval_individual(self, arch: List[int]):
        score, latency_loss, acc_loss = (0, 0, 0)
        for latency_constraint in self.latency_constraint_list:
            _score, _latency_loss, _acc_loss = self._eval_individual_one_latency(arch, latency_constraint)
            score += _score 
            acc_loss += _acc_loss
            latency_loss += _latency_loss
        return score, latency_loss, acc_loss

    # arch [1, 3, 2, 1, 2, 3]
    def _eval_individual_one_latency(self, arch: List[int], latency_constraint):
        assert len(arch) == self.num_stages
        
        indiv_sample_list = []
        for _ in range(self.num_samples_per_indiv // 500):
            indiv_sample_list.extend(self.lut.sample_indiv(arch, n=500))

        # eval score
        # latency
        indiv_sample_list = sorted(indiv_sample_list, key=lambda x: x['latency'])
        end = 0
        for indiv in indiv_sample_list:
            if indiv['latency'] <= latency_constraint:
                end += 1
            else:
                break
        p_latency_smaller_than_constraint = end / self.num_samples_per_indiv + 1e-5
        # loss
        avg_top_loss = 0
        if end:
            indiv_sample_list = indiv_sample_list[:end]
            indiv_sample_list = sorted(indiv_sample_list, key=lambda x: x['loss'])
            for sample in indiv_sample_list[:self.top]:
                avg_top_loss += sample['loss']
            avg_top_loss /= len(indiv_sample_list[:self.top]) + 1e-5
        else:
            warning('Sample 0 indv that satisfy constraint.')
            avg_top_loss = 1e5

        # calculate score
        latency_loss = np.clip((self.lat_loss_t / p_latency_smaller_than_constraint) ** self.lat_loss_a, a_min=1, a_max=INFINITY)
        score = 1 / (latency_loss * avg_top_loss)
        return score, latency_loss, avg_top_loss

    # ===== functions for traverse =====

    def list_all_indiv_arch(self):
        def f_recursive(idx):
            if idx == self.num_stages - 1:
                for i in range(self.num_choices_per_stage):
                    rv.append(buffer + [i])
            else:
                for i in range(self.num_choices_per_stage):
                    buffer.append(i)
                    f_recursive(idx + 1)
                    buffer.pop()
        rv = []
        buffer = []
        f_recursive(0)
        return rv

    def list_all_indiv(self):
        rv = []
        for arch in self.list_all_indiv_arch():
            rv.append((arch, self.eval_individual(arch)))
        rv.sort(key=lambda x: x[1], reverse=True)
        return rv

    # ===== functions for evolution =====

    def random_sample(self):
        rv = []
        for _ in range(self.num_stages):
            rv.append(np.random.choice(list(range(self.num_choices_per_stage))))
        return rv 

    def mutate(self, arch: List[int]) -> List[int]:
        arch = copy.deepcopy(arch)
        stage_idx = np.random.choice(list(range(self.num_stages)))
        choice = np.random.choice(list(range(self.num_choices_per_stage)))
        while choice == arch[stage_idx]:
            choice = np.random.choice(list(range(self.num_choices_per_stage)))
        arch[stage_idx] = choice
        return arch
