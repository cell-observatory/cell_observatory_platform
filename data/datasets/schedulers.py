import sys
import logging
import warnings
from typing import Dict, List

import ray
import torch

from utils.context import (list_numa_nodes,
                            cpus_for_node,
                            node_id,
                            read_numa_distance_row,
                            get_world_size,
                            torch_gpu_to_numa)

logger = logging.getLogger(__name__)


@ray.remote(namespace="schedulers",
            lifetime="detached",
            num_cpus=0)
class NumaNodeAffinityScheduler:
    """
    Returns a NUMA node id to bind actor's process to.

    Currently only supports the following policy:
        Local-first placement with distance-aware fallback and 
        optional oversubscription.
    """
    
    def __init__(self, 
                 node_id: int,
                 gpu_numa_nodes: List[int],
                 policy: str = "distance", 
                 oversub_factor: float = 1.0,
    ):  
        # set policy
        self.policy = policy
        self.oversub_factor = max(1.0, float(oversub_factor))

        # discover topology
        self.numa_nodes = list_numa_nodes()
        self.gpu_numa_nodes = gpu_numa_nodes
        self.cpu_only_nodes = [n for n in self.numa_nodes \
                                if n not in self.gpu_numa_nodes]
        self.cpus_for_node = {n: cpus_for_node(n) for n in self.numa_nodes}
        self.node_capacity = {
            n: int(len(self.cpus_for_node[n]) * self.oversub_factor) for n in self.numa_nodes
        }

        # distance map: lower is closer (10=self, 12=20% slower, etc.)
        self.distance = self._build_distance_map(self.numa_nodes)

        ray.logger.info(f"NUMA Node Affinity Scheduler initialized on node {node_id} "
                    f"with policy '{self.policy}' and oversub_factor={self.oversub_factor}. "
                    f"Discovered NUMA nodes: {self.numa_nodes}, "
                    f"GPU NUMA nodes: {self.gpu_numa_nodes}, "
                    f"CPU-only NUMA nodes: {self.cpu_only_nodes}, "
                    f"Node capacities: {self.node_capacity}, "
                    f"Distance map: {self.distance}")

        # current allocations per NUMA (how many loader actors placed)
        self.allocations = {n: 0 for n in self.numa_nodes}
        self.remote_for_requested = {n: 0 for n in self.numa_nodes}
        self.total_for_requested = {n: 0 for n in self.numa_nodes}

    # --------- discovery helpers ----------

    def _build_distance_map(self, nodes: List[int]) -> Dict[int, Dict[int, int]]:
        all_rows = {}
        for n in nodes:
            row = read_numa_distance_row(n)
            if row is not None and len(row) == len(nodes):
                all_rows[n] = row

        dist = {a: {} for a in nodes}
        if all_rows:
            for a in nodes:
                row = all_rows.get(a)
                if row is None:
                    warnings.warn("NUMA distance information not available, using fallback.")
                    for b in nodes:
                        dist[a][b] = 10 if a == b else 20
                else:
                    for b in nodes:
                        dist[a][b] = row[b]

        # fallback if no distance info available
        else:
            warnings.warn("NUMA distance information not available, using fallback.")
            for a in nodes:
                for b in nodes:
                    dist[a][b] = 10 if a == b else 20
        return dist

    # --------- capacity / accounting ----------

    def _has_capacity(self, node: int) -> bool:
        return self.allocations[node] < self.node_capacity[node]

    def _record_allocation(self, chosen_node: int, requested_node: int):
        self.allocations[chosen_node] += 1
        self.total_for_requested[requested_node] += 1
        if chosen_node != requested_node:
            self.remote_for_requested[requested_node] += 1
        
        logger.info(f"Placed actor on NUMA node {chosen_node} "
                    f"(requested {requested_node}). "
                    f"Current allocations: {self.allocations}, "
                    f"remote for requested: {self.remote_for_requested}, "
                    f"total for requested: {self.total_for_requested}")

    def _record_free(self, freed_node: int):
        self.allocations[freed_node] = max(0, self.allocations[freed_node] - 1)
        logger.info(f"Freed one allocation on NUMA node {freed_node}. "
                    f"Current allocations: {self.allocations}")

    # --------- API ----------

    def list_cpu_only_nodes(self):
        return list(self.cpu_only_nodes)

    def schedule_actor_for_gpu(self, torch_gpu_index: int) -> int:        
        info = torch_gpu_to_numa(torch_gpu_index)
        gpu_numa = int(info["numa_node"])
        return self.schedule_actor(gpu_numa)

    def schedule_actor(self, requested_numa: int) -> int:
        if self.policy == "distance":
            if requested_numa not in self.numa_nodes:
                raise ValueError(f"Requested NUMA node {requested_numa} is not valid. "
                                 f"Available NUMA nodes: {self.numa_nodes}")

            # if local node has capacity, use it
            if self._has_capacity(requested_numa):
                self._record_allocation(requested_numa, requested_numa)
                return requested_numa

            # otherwise, find the closest node with capacity
            candidates = sorted(
                (n for n in self.numa_nodes if n != requested_numa),
                # sort by distance, then by current load
                key=lambda n: (self.distance[requested_numa].get(n, 9999), self.allocations[n] / max(1, self.node_capacity[n]))
            )

            # only support CPU-only nodes as fallback
            for n in [c for c in candidates if c in self.cpu_only_nodes]:
                if self._has_capacity(n):
                    self._record_allocation(n, requested_numa)
                    return n

            # if no CPU-only node available, use local node again (oversubscribe if needed)
            self._record_allocation(requested_numa, requested_numa)
            return requested_numa
        
        else:
            raise ValueError(f"Unsupported scheduling policy: {self.policy}")

    def free(self, placed_numa: int):
        if placed_numa in self.numa_nodes:
            self._record_free(placed_numa)

    def snapshot(self) -> Dict:
        return {
            "numa_nodes": self.numa_nodes,
            "gpu_numa_nodes": self.gpu_numa_nodes,
            "cpu_only_nodes": self.cpu_only_nodes,
            "capacity": self.node_capacity,
            "allocations": self.allocations,
            "remote_for_requested": self.remote_for_requested,
            "total_for_requested": self.total_for_requested,
            "distance": self.distance,
            "oversub_factor": self.oversub_factor,
        }