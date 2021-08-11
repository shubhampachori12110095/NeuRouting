from generators import generate_multiple_instances
from main.evaluator import Evaluator
from nlns.factory import nlns_builder

if __name__ == "__main__":
    n_customers = 50
    n_instances = 5
    max_steps = n_customers
    time_limit = 60
    neighborhood_size = 10
    ckpt_path = "../../pretrained/"

    eval_instances = generate_multiple_instances(n_instances=n_instances, n_customers=n_customers, seed=0)

    random_env = nlns_builder("random", "greedy", neighborhood_size, ckpt_path=ckpt_path, name="RANDOM")
    egate_env = nlns_builder("egate", "greedy", neighborhood_size, ckpt_path=ckpt_path, name="EGATE")
    rlagent_env = nlns_builder("point", "rlagent", neighborhood_size, ckpt_path=ckpt_path, name="RL AGENT")

    evaluator = Evaluator([random_env, egate_env, rlagent_env])

    stats = evaluator.compare(eval_instances, max_steps=max_steps, time_limit=time_limit)
    print(stats.mean_cost())
