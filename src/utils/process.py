import argparse
import ast
from copyreg import pickle
from enum import Enum
import os
import random
import time
import torch
import pickle

from problems.cvrp import VRPDataset, make_instance


class Mode(str, Enum):
    CGP = "cgp_search"
    RANDOM_SEARCH = "random_search"

    GENOME_EVALUATION = "genome_evaluation" 
    SCORING = "scoring"
    GENERATE_VALIDATION_DATA = "generate_validation_data"

def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Model of evolving architecture of GNN with CGP for CVRP")
    parser.add_argument('--problem', default='cvrp', help="The problem to solve, default 'tsp'")
    parser.add_argument('--model', default='attention', help="Model, 'attention' (default) or 'pointer'")
    parser.add_argument('--mode', choices = [m.value for m in Mode], default=Mode.CGP.value)

    parser.add_argument('--genome_name', type=str, help='Name of predefined architecture')
    parser.add_argument('--genome_path', type=str, help='Path to dir with candidates metadata')
    parser.add_argument('--id', type=int, default=None, help='Identifier of candidate to use')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--epoch_size', type=int, default=1280000, help='Number of instances per epoch during training')
    parser.add_argument('--budget', type=int, default=200, help='Computational budget of architecture search')
    parser.add_argument('--start_from_transformer', action='store_true', help='Indicates if evolution starts from transformer architecture'
                        ' instead of randomly generated parents')
    parser.add_argument('--deep_neural_connection', action='store_true', help='Disable limiting the depth of nerual connections')
    parser.add_argument('--x_dim', type=int, default=8, help='Size of X dimension of the grid')
    parser.add_argument('--y_dim', type=int, default=5, help='Size of Y dimension of the grid')
    parser.add_argument('--validation_set_path', type=str, help='Path to validation set')
    parser.add_argument('--test_set_path', type=str, help='Path to test set')
    parser.add_argument('--val_test_size', type=int, default=10000,
                        help='Number of instances used for validation and test set')
    parser.add_argument('--checkpoint_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--n_epochs', type=int, default=100, help='The number of epochs to train')
    parser.add_argument('--graph_size', type=int, default=10, help="The size of the problem graph")
    parser.add_argument('--batch_size', type=int, default=512, help='Number of instances per batch during training')
    # # Model
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3,
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Clip the parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")

    # # Training
    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=1.0, help='Learning rate decay per epoch')


    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--exp_beta', type=float, default=0.8,
                        help='Exponential moving average baseline decay (default 0.8)')
    parser.add_argument('--baseline', default=None,
                        help="Baseline to use: 'rollout', 'critic' or 'exponential'. Defaults to no baseline.")
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help='Significance in the t-test for updating rollout baseline')
    parser.add_argument('--bl_warmup_epochs', type=int, default=None,
                        help='Number of epochs to warmup the baseline, default None means 1 for rollout (exponential '
                             'used for warmup phase), 0 otherwise. Can only be used with rollout baseline.')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                         help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help='Set to decrease memory usage by checkpointing encoder')
    parser.add_argument('--shrink_size', type=int, default=None,
                        help='Shrink the batch size if at least this many instances in the batch are finished'
                             ' to save memory (default None means no shrinking)')
    parser.add_argument('--data_distribution', type=str, default=None,
                        help='Data distribution to use during training, defaults and options depend on problem.')

    # # Misc
    parser.add_argument('--log_step', type=int, default=50, help='Log info every log_step steps')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--no_save_model', action='store_true', help='Don\'t save model weights after training')

    parser.add_argument('--log_dir', default='../logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--run_name', default='', type=str, help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    
    parser.add_argument('--genome', type=str, default=None, help='Genome of architecture to evaluate')

    opts = parser.parse_args(args)
    if opts.seed == -1:
        opts.seed = random.randint(0, 9999)
        
    if opts.mode == "full_evaluation":
        if opts.genome_path is None:
            parser.error("--genome_path is required in full_evaluation mode")
        if opts.id is None:
            parser.error("--id is required in full_evaluation mode")
    
    if opts.validation_set_path is not None:
        data = torch.load(opts.validation_set_path)
        opts.validation_set = VRPDataset(size=opts.graph_size, num_samples=opts.val_test_size, data=data)
        
    if opts.test_set_path is not None:
        if opts.test_set_path.endswith(".pkl"):
            with open(opts.test_set_path, "rb") as f:
                data = pickle.load(f)
            data = [make_instance(instance) for instance in data]
        else:
            data = torch.load(opts.test_set_path)
        opts.test_set = VRPDataset(size=opts.graph_size, num_samples=opts.val_test_size, data=data)
        
    if opts.genome:
        opts.genome = ast.literal_eval(opts.genome)
    # CUSTOM SENEGAS
    opts.baseline = 'rollout'
    # opts.no_progress_bar = False
    opts.n_heads = 8

    opts.epoch_time_limit = 10*60 # 10 minut

    # MUTATION
    opts.struct_mutation_p = 0.1
    opts.output_mutation_p = 0.2
    opts.debug_mutation = True
    opts.debug = False
    
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")
    opts.run_name = "run_{}_{}".format(time.strftime("%Y%m%dT%H%M%S"), opts.run_name)
    opts.save_dir = os.path.join(
        opts.output_dir,
        opts.run_name
    )
    opts.reproducible_seed = True
    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == 'rollout' else 0
    assert (opts.bl_warmup_epochs == 0) or (opts.baseline == 'rollout')
    assert opts.epoch_size % opts.batch_size == 0, "Epoch size must be integer multiple of batch size!"
    return opts

def initial_setup(opts):
    os.makedirs(opts.save_dir)
    if opts.mode == "cgp_search" or opts.mode == "random_search" or opts.mode == "evolve_transformer":    
        os.makedirs(os.path.join(opts.save_dir,"genomes_full"))
        opts.genomes_full_out_dir = os.path.join(opts.save_dir, "genomes_full")
        os.makedirs(os.path.join(opts.save_dir,"genomes_active"))
        opts.genomes_active_out_dir = os.path.join(opts.save_dir, "genomes_active")
        os.makedirs(os.path.join(opts.save_dir,"parents"))
        opts.parents_out_dir = os.path.join(opts.save_dir, "parents")
    
