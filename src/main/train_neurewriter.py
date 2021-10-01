import sys
import argparse
import random
import time
import numpy as np
import torch

sys.path.append("src")

from neurewriter.data_utils import process_batch
from neurewriter.neurewriter_model import NeuRewriterModel
from neurewriter.supervisor import VRPSupervisor
from generators import generate_multiple_instances
from nlns.initial import nearest_neighbor_solution
from utils.logging import ConsoleLogger

parser = argparse.ArgumentParser(description="NeuRewriter")
parser.add_argument('--cpu', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=False)
parser.add_argument('-n', '--n_customers', type=int, required=True)
parser.add_argument('-ts', '--train_samples', type=int, default=100000)
parser.add_argument('-vs', '--val_samples', type=int, default=100)
parser.add_argument('-dist', '--distribution', type=str, default="nazari")
parser.add_argument('--processes', type=int, default=1)
parser.add_argument('--load_model', type=str, default=None)

parser.add_argument('--LSTM_hidden_size', type=int, default=512)
parser.add_argument('--MLP_hidden_size', type=int, default=256)
parser.add_argument('--param_init', type=float, default=0.1)
parser.add_argument('--num_LSTM_layers', type=int, default=1)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--num_sample_rewrite_pos', type=int, default=10)
parser.add_argument('--num_sample_rewrite_op', type=int, default=10)
parser.add_argument('--cont_prob', type=float, default=0.5)

data_group = parser.add_argument_group('data')

data_group.add_argument('--lr', type=float, default=5e-5)
data_group.add_argument('--value_loss_coef', type=float, default=0.01)
data_group.add_argument('--gamma', type=float, default=0.9)
data_group.add_argument('-bs', '--batch_size', type=int, default=256)
data_group.add_argument('--num_MLP_layers', type=int, default=2)
data_group.add_argument('--embedding_size', type=int, default=7)
data_group.add_argument('--attention_size', type=int, default=16)

output_trace_group = parser.add_argument_group('output_trace_option')
output_trace_group.add_argument('--output_trace_flag', type=str, default='nop',
                                choices=['succeed', 'fail', 'complete', 'nop'])
output_trace_group.add_argument('--output_trace_option', type=str, default='both', choices=['pred', 'both'])
output_trace_group.add_argument('--output_trace_file', type=str, default=None)

train_group = parser.add_argument_group('train')
train_group.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'])
train_group.add_argument('--lr_decay_steps', type=int, default=500)
train_group.add_argument('--lr_decay_rate', type=float, default=0.9)
train_group.add_argument('--gradient_clip', type=float, default=5.0)
train_group.add_argument('-e', '--epochs', type=int, default=10)
train_group.add_argument('--dropout_rate', type=float, default=0.0)
train_group.add_argument('-li', '--log_interval', type=int, required=False)

args = parser.parse_args()


def create_model(args):
    model = NeuRewriterModel(args)
    if model.cuda_flag:
        model = model.cuda()
    model.share_memory()
    model_supervisor = VRPSupervisor(model, args)
    if args.load_model:
        model_supervisor.load_pretrained(args.load_model)
    else:
        print('Created model with fresh parameters.')
        model_supervisor.model.init_weights(args.param_init)
    return model_supervisor


if __name__ == "__main__":
    args.cuda = not args.cpu and torch.cuda.is_available()
    device = "cuda:0" if args.cuda else "cpu"
    print(f"Using {device} for training.")

    # logger = MultipleLogger(loggers=[ConsoleLogger(), WandBLogger()])
    logger = ConsoleLogger()
    run_name = f"model_neurewriter_n{args.n_customers}"
    logger.new_run(run_name=run_name[:run_name.rfind('.')])

    train_instances = generate_multiple_instances(n_instances=args.train_samples,
                                                  n_customers=args.n_customers,
                                                  distribution=args.distribution,
                                                  seed=42)
    train_data = [nearest_neighbor_solution(inst) for inst in train_instances]

    val_instances = generate_multiple_instances(n_instances=args.val_samples,
                                                n_customers=args.n_customers,
                                                distribution=args.distribution,
                                                seed=73)
    val_data = [nearest_neighbor_solution(inst) for inst in val_instances]

    model_supervisor = create_model(args)

    print(f"Starting NeuRewriter training...")

    losses = []
    rewards = []
    incumbent_cost = np.inf

    for epoch in range(args.epochs):
        random.shuffle(train_data)
        for batch_idx in range(0, args.train_samples, args.batch_size):
            batch_data = process_batch(train_data, args.batch_size, batch_idx)
            train_loss, train_reward = model_supervisor.train(batch_data)
            losses.append(train_loss)
            rewards.append(train_reward)

            if (batch_idx + 1) % args.log_interval == 0:
                logger.log({
                    "epoch": epoch + 1,
                    "batch_idx": batch_idx // args.batch_size + 1,
                    "loss": np.mean(losses),
                    "mean_reward": np.mean(rewards)
                }, phase="train")
                rewards = losses = []

            if args.lr_decay_steps and model_supervisor.global_step % args.lr_decay_steps == 0:
                model_supervisor.model.lr_decay(args.lr_decay_rate)
                if model_supervisor.model.cont_prob > 0.01:
                    model_supervisor.model.cont_prob *= 0.5

        start_eval_time = time.time()
        eval_loss, eval_reward, eval_cost = model_supervisor.eval(val_data)
        runtime = time.time() - start_eval_time

        logger.log({
            "epoch": epoch + 1,
            "loss": eval_loss,
            "mean_reward": eval_reward,
            "mean_cost": eval_cost,
            "runtime": runtime
        }, phase="val")

        if eval_cost < incumbent_cost:
            incumbent_cost = eval_cost
            model_supervisor.save_model(ckpt_path="./pretrained/", ckpt_name=run_name + ".pt")

    print(f"Training completed successfully.")
