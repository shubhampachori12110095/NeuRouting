from typing import List, Tuple

import numpy as np

from lns.destroy import DestroyProcedure
from lns.initial import nearest_neighbor_solution
from lns.repair import RepairProcedure


class LNSOperatorPair:
    def __init__(self, destroy_operator: DestroyProcedure, repair_operator: RepairProcedure):
        self.destroy = destroy_operator
        self.repair = repair_operator


class LargeNeighborhoodSearch:
    def __init__(self,
                 operator_pairs: List[LNSOperatorPair],
                 initial=nearest_neighbor_solution,
                 adaptive=False):
        self.initial = initial
        self.destroy = [op.destroy for op in operator_pairs]
        self.repair = [op.repair for op in operator_pairs]
        self.n_operators = len(operator_pairs)
        self.adaptive = adaptive
        self.performances = [np.inf] * self.n_operators if adaptive else None

    def select_operator_pair(self) -> Tuple[DestroyProcedure, RepairProcedure, int]:
        if self.adaptive:
            idx = np.argmax(self.performances)
        else:
            idx = np.random.randint(0, self.n_operators)
        return self.destroy[idx], self.repair[idx], idx
