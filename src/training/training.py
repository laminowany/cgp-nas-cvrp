from dataclasses import dataclass
import os
import time
import numpy as np
import torch
import math
from torch.nn import DataParallel
import torch.optim as optim
import random

from tqdm import tqdm
from torch.utils.data import DataLoader
from models.attention_model import AttentionModel
from models.reinforce_baselines import RolloutBaseline, WarmupBaseline, get_inner_model
from nas.cgp import CGP_Encoder
from models.encoders.graph_encoder import GraphAttentionEncoder
from utils.logger import Logger
from utils.misc import move_to
from problems.cvrp import CVRP

@dataclass
class EvaluationResult:
    model: AttentionModel
    scores: list
    snapshots: list
    
def reset_seeds(opts):
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    
def produce_transformer_genome(opts):
    if opts.x_dim < 8:
        raise Exception("For transformer to fit it genotype the x_dim must have at least 8 length")
    to_global_idx = lambda x, y, opts: opts.x_dim * opts.y_dim + 1 if x == opts.x_dim else y * opts.x_dim + x + 1
    length = opts.x_dim * opts.y_dim
    genome = [None] * (length + 2)
    for y in range(opts.y_dim):
        genome[to_global_idx(0, y, opts)] = ((1, 0))
        for x in range(1, opts.x_dim):
            pos = to_global_idx(x, y, opts)
            genome[pos] = ((1, to_global_idx(x - 1, y, opts)))
    main_row = opts.y_dim // 2
    genome[-1] = (5, (to_global_idx(opts.x_dim - 1, main_row, opts)))    
    
    prev_pos = 0
    x = 0
    while x <= opts.x_dim - 8:
        pos = to_global_idx(x, main_row, opts)
        genome[pos] = ((4, prev_pos))
        genome[to_global_idx(x, main_row - 1, opts)] = ((1, prev_pos))
        genome[pos + 1] = ((5, (to_global_idx(x, main_row - 1, opts), pos)))   
        genome[pos + 2] = ((2, pos + 1)) 
        genome[pos + 3] = ((3, pos + 2, 1))
        genome[to_global_idx(x + 3, main_row - 1, opts)] = ((1, pos + 2))
        genome[pos + 4] = ((7, pos + 3))
        genome[pos + 5] = ((3, pos + 4, -1))
        genome[pos + 6] = ((5, (to_global_idx(x + 5, main_row - 1, opts), pos + 5)))  
        genome[pos + 7] = ((2, pos + 6))
        prev_pos = pos + 7
        x = x + 8
    
    return genome

def verify_sanity(opts, logger: Logger):
    orig_epochs = opts.n_epochs
    orig_epoch_size = opts.epoch_size
    orig_graph_size = opts.graph_size
    orig_y_dim = opts.y_dim
    orig_x_dim = opts.x_dim
    orig_no_progress_bar = opts.no_progress_bar
    orig_no_save_model = opts.no_save_model
    opts.n_epochs = 1
    opts.epoch_size = 1
    opts.graph_size = 10
    opts.x_dim = 8
    opts.no_progress_bar = True
    opts.no_save_model = True

    validation_set = getattr(opts, "validation_set", None)
    if not validation_set:
        validation_set = CVRP.make_dataset(size=opts.graph_size, num_samples=opts.val_test_size)
    
    reset_seeds(opts)
    original_encoder = GraphAttentionEncoder(
        n_heads=opts.n_heads,
        embed_dim=opts.embedding_dim,
        n_layers=1,
        normalization=opts.normalization
    )
    model_original = AttentionModel(opts, original_encoder)
    baseline = RolloutBaseline(model_original, opts)
    baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)
    optimizer = optim.Adam(
        [{'params': model_original.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)
    scores_orig = []
    for i in range(3):
        scores_orig.append(float(train_epoch(
                        model_original,
                        optimizer,
                        baseline,
                        lr_scheduler,
                        i,
                        validation_set,
                        opts
                    )))
        logger.record(
                    epoch=i,
                    id=-2,
                    score=scores_orig[-1],
                    time=0)
    
    reset_seeds(opts)
    encoder = CGP_Encoder(opts, produce_transformer_genome(opts))
    model = AttentionModel(opts, encoder)
    baseline = RolloutBaseline(model, opts)
    baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)
    scores_cgp = []
    for i in range(3):
        scores_cgp.append(float(train_epoch(
                        model,
                        optimizer,
                        baseline,
                        lr_scheduler,
                        i,
                        validation_set,
                        opts
                    )))
        logger.record(
                    epoch=i,
                    id=-1,
                    score=scores_cgp[-1],
                    time=0
                )

    if scores_orig != scores_cgp:
        raise Exception("CARAMBA!")
    
    opts.n_epochs = orig_epochs
    opts.epoch_size = orig_epoch_size
    opts.graph_size = orig_graph_size
    opts.y_dim = orig_y_dim
    opts.x_dim = orig_x_dim
    opts.no_progress_bar = orig_no_progress_bar
    opts.no_save_model = orig_no_save_model

def evaluate(opts, logger: Logger, encoder = None, candidate_id = None) -> EvaluationResult:
    if not candidate_id:
        candidate_id = 0
        
    torch.manual_seed(opts.seed)

    model = AttentionModel(opts, encoder)
    baseline = RolloutBaseline(model, opts)
    baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)
    validation_set = getattr(opts, "validation_set", None)
    if not validation_set:
        validation_set = CVRP.make_dataset(size=opts.graph_size, num_samples=opts.val_test_size)
    
    snapshots = []
    scores = []
    for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
        start = time.perf_counter()
        try:
            score = train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                validation_set,
                opts
            )
            scores.append(score)
        except TimeoutError:
            return None
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
        end = time.perf_counter()
        logger.record(
                epoch=epoch,
                id=candidate_id,
                score=float(score),
                time=end-start
        )
    return EvaluationResult(model, scores, snapshots)

def validate(model, dataset, opts):
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    model.set_decode_type("greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)

def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, opts):
    if not opts.no_progress_bar:
        print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(CVRP.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model.train()
    model.set_decode_type("sampling")
    start = time.perf_counter()
    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        train_batch(
            model,
            optimizer,
            baseline,
            batch,
            opts
        )
        step += 1
    epoch_duration = time.time() - start_time
    if not opts.no_progress_bar:
        print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))
    
    if epoch == opts.n_epochs - 1 and not opts.no_save_model:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, val_dataset, opts)
    if not opts.no_progress_bar:
        print(f'epoch {epoch}, score {avg_reward}')
    baseline.epoch_callback(model, epoch)
    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()
    return avg_reward

def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)

def train_batch(
        model,
        optimizer,
        baseline,
        batch,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

