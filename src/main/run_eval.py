import sys

sys.path.append("src")

from baselines import LKHSolver, SCIPSolver, OrToolsSolver
from generators import generate_multiple_instances
from main.evaluator import Evaluator
from nlns.builder import nlns_builder

if __name__ == "__main__":
    n_customers = 50
    n_instances = 100
    max_steps = n_customers
    time_limit = 60
    neigh_size = 256
    ckpt_path = "./pretrained/"

    eval_instances = generate_multiple_instances(n_instances=n_instances, n_customers=n_customers, seed=0)

    destroy_operators = {"point": [0.15, 0.25], "tour": [0.15, 0.25]}

    greedy_env = nlns_builder(destroy_operators, ["greedy"], neigh_size,
                              simulated_annehaling=True, ckpt_path=ckpt_path, name="Greedy")
    rlagent_env = nlns_builder(destroy_operators, ["rlagent"], neigh_size,
                               simulated_annehaling=False, ckpt_path=ckpt_path, name="RL Agent")
    scip_solver = SCIPSolver()
    ortools_solver = OrToolsSolver()
    lkh_solver = LKHSolver("./executables/LKH3")

    evaluator = Evaluator([greedy_env, rlagent_env, scip_solver, ortools_solver, lkh_solver])

    stats = evaluator.compare(eval_instances, max_steps=max_steps, time_limit=time_limit)
    print(stats.mean_cost())
