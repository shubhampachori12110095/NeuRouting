# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from instances import VRPSolution
from experimental.neurewriter.manager import VRPManager, VRPNode


def parse_solution(solution: VRPSolution) -> VRPManager:
    instance = solution.instance
    dm = VRPManager(instance)
    dm.nodes.append(VRPNode(x=instance.depot[0], y=instance.depot[1], demand=0, px=instance.depot[0],
                            py=instance.depot[1], capacity=instance.capacity, dis=0.0))
    for position, demand in zip(instance.customers, instance.demands):
        dm.nodes.append(VRPNode(x=position[0], y=position[1], demand=demand,
                                px=position[0], py=position[1],
                                capacity=instance.capacity, dis=0.0))
    dm.num_nodes = len(dm.nodes)

    for route in solution.routes:
        dm.add_route_node(0)
        for c in route[1:-1]:
            dm.add_route_node(c)
    dm.add_route_node(0)

    return dm
