from typing import List

import numpy as np
import torch

from generators.nazari_generator import generate_nazari_instances
from generators.uchoa_generator import generate_uchoa_instances
from instances import VRPInstance


def generate_instance(n_customers: int,
                      distribution: str = 'nazari') -> VRPInstance:
    return generate_multiple_instances(1, n_customers, distribution)[0]


def generate_multiple_instances(n_instances: int,
                                n_customers: int,
                                distribution: str = 'nazari',
                                seed=42) -> List[VRPInstance]:
    if distribution == 'nazari':
        np.random.seed(seed)
        return generate_nazari_instances(n_instances, n_customers)
    elif distribution == 'uchoa':
        torch.manual_seed(seed)
        return generate_uchoa_instances(n_instances, n_customers)
    else:
        Exception(f"{distribution} is unknown.")
