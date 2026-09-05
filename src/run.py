import ast
import csv
import gc
import os
from pathlib import Path
import numpy as np
import torch
import random

from models.attention_model import AttentionModel
from problems.cvrp import CVRP
from utils.graph import export_cgp_to_graphviz
from utils.misc import compare_floats
from utils.process import get_options, initial_setup
from nas.cgp import CGP_Encoder
from training.training import evaluate, produce_transformer_genome, reset_seeds, validate, verify_sanity 
from utils.logger import Logger

def train_and_validate_encoder(opts, logger, encoder, encoder_result):
    children_limit = 4
    scores = {}
    result_cache = {}
    osobnik_id = 0
    generation = 1
    
    hashkey = hash(encoder)   
    scores[osobnik_id] =  encoder_result.scores[-1]
    result_cache[hashkey] = scores[osobnik_id]      
    best_score = scores[osobnik_id]
    best_encoder = encoder
    best_id = osobnik_id
    best_weights = encoder.save_snapshot()
    osobnik_id += 1
    
    parent_weights = best_weights
    parent_score = best_score
    parent_encoder = best_encoder

    logger.record(key="candidates", id=best_id, genome=encoder.genome)
    logger.record(key="candidates_scores", id=best_id, final_score=scores[best_id])
    export_cgp_to_graphviz(parent_encoder.genes, opts, os.path.join(opts.genomes_full_out_dir, f"_PARENT_{best_id}"), only_active=False)
    export_cgp_to_graphviz(parent_encoder.genes, opts, os.path.join(opts.genomes_active_out_dir, f"_PARENT_{best_id}"), only_active=True)
    export_cgp_to_graphviz(parent_encoder.genes, opts, os.path.join(opts.parents_out_dir, f"generation_{0}_genome_{best_id}"), only_active=True)
    export_cgp_to_graphviz(parent_encoder.genes, opts, os.path.join(opts.parents_out_dir, f"generation_{0}_genome_{best_id}_FULL"), only_active=False)
    
    budget = opts.budget
    while budget > 0:
        encoders = parent_encoder.produce_offspring(children_limit, opts, budget)
        export_cgp_to_graphviz(parent_encoder.genes, opts, os.path.join(opts.genomes_full_out_dir, f"{generation}__PARENT"), only_active=False)
        export_cgp_to_graphviz(parent_encoder.genes, opts, os.path.join(opts.genomes_active_out_dir, f"{generation}__PARENT"), only_active=True)
        
        for encoder in encoders:
            logger.record(key="candidates", id=osobnik_id, genome=encoder.genome)
            hashkey = hash(encoder) 
            parent_equivalent = CGP_Encoder.are_equivalent(parent_encoder, encoder)
            
            if parent_equivalent:
                scores[osobnik_id] = result_cache[hashkey]
                export_cgp_to_graphviz(encoder.genes, opts, os.path.join(opts.genomes_full_out_dir, f"{generation}_genome_{osobnik_id}_EQ"), only_active=False)
                export_cgp_to_graphviz(encoder.genes, opts, os.path.join(opts.genomes_active_out_dir, f"{generation}_genome_{osobnik_id}_EQ"), only_active=True)
                if compare_floats(parent_score, best_score) == 0: # neutral drift
                    best_encoder = encoder
                    best_id = osobnik_id
            else:
                export_cgp_to_graphviz(encoder.genes, opts, os.path.join(opts.genomes_full_out_dir, f"{generation}_genome_{osobnik_id}_NEW"), only_active=False)
                export_cgp_to_graphviz(encoder.genes, opts, os.path.join(opts.genomes_active_out_dir, f"{generation}_genome_{osobnik_id}_NEW"), only_active=True)
                encoder.load_snapshot(parent_weights)
                if hashkey in result_cache:
                    scores[osobnik_id] = result_cache[hashkey]
                else:
                    result = evaluate(opts, logger, encoder, osobnik_id)
                    if result:
                        scores[osobnik_id] = result.scores[-1]
                        result_cache[hashkey] = scores[osobnik_id]
                        if compare_floats(scores[osobnik_id], best_score) == -1:
                            best_encoder = encoder
                            best_score = scores[osobnik_id]
                            best_weights = encoder.save_snapshot()
                            best_id = osobnik_id
                            print(f'Found better offspring {osobnik_id} with score: {best_score}.')
                            export_cgp_to_graphviz(encoder.genes, opts, os.path.join(opts.parents_out_dir, f"generation_{generation}_genome_{osobnik_id}"), only_active=True)
                    else:
                        scores[osobnik_id] = None
                    budget -= 1
                    if budget == 0:
                        break
            logger.record(key="candidates_scores", id=osobnik_id, final_score=scores[osobnik_id])
            osobnik_id += 1
        parent_encoder = best_encoder
        parent_score = best_score
        parent_weights = best_weights
            
        print(f'Generation {generation}, the best score {best_score}')    
        logger.record(key="evolution", generation=generation, best_id=best_id, score=best_score)
        logger.record(key="budget", budget=(opts.budget - budget), best_id=best_id, score=best_score)
        generation += 1

def run_random_search(opts, logger):
    result_cache = {}
    best_score = None
    best_model = None
    candidate_id = 1
    budget = opts.budget
    while budget > 0:
        encoder = CGP_Encoder.random_genome(opts)
        model = AttentionModel(opts, CGP_Encoder.random_genome(opts))
        export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.genomes_full_out_dir, f"CANDIDATE_{candidate_id}"), only_active=False)
        export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.genomes_active_out_dir, f"CANDIDATE_{candidate_id}"), only_active=True)
        logger.record(key="candidates", id=candidate_id, genome=model.get_encoder().genome)
        hashkey = hash(model.get_encoder()) 
        if hashkey in result_cache: # duplicate
            continue
        result = evaluate(opts, logger, encoder, candidate_id)
        if result:    
            score =  result.scores[-1]
            logger.record(key="scores", id=candidate_id, final_score=score)
            result_cache[hashkey] = score
            
            if not best_score or compare_floats(score, best_score) == -1:
                best_score = score
                del best_model
                best_model = model
                best_id = candidate_id
                print(f'Found better candidate {candidate_id} with cost: {best_score}.')
                export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.parents_out_dir, f"genome_{candidate_id}"), only_active=True)
            else:
                del model
        logger.record(key="budget", budget=(opts.budget - budget), best_id=best_id, score=best_score)
        print(f'Iteration {opts.budget - budget + 1}, best so far {best_score}') 
        candidate_id += 1
        budget -= 1
    logger.record(key="winner", id=candidate_id, x_dim = opts.x_dim, y_dim = opts.y_dim, genome=best_model.get_encoder().genome)

def run_genome_evaluation(opts, logger):
    if not opts.genome:
        raise Exception("Please specify genome with --genome")
    genome = opts.genome
    encoder = CGP_Encoder(opts, genome)
    logger.record(key="genomes", name = opts.genome_name, genome = genome)
    export_cgp_to_graphviz(encoder.genes, opts, os.path.join(opts.save_dir, f"candidate"), only_active=True)
    
    evaluate(opts, logger, encoder)


def run_transformer_evolution(opts, logger):
    parent_encoder = CGP_Encoder(opts, produce_transformer_genome(opts))
    osobnik_id = 1
    result = evaluate(opts, logger, parent_encoder, osobnik_id)  
    train_and_validate_encoder(opts, logger, parent_encoder, result)
        
        
def run_cgp_search(opts, logger):
    osobnik_id = 0
    result = None
    
    if opts.start_from_transformer:
        encoder = CGP_Encoder(opts, produce_transformer_genome(opts))
        result = evaluate(opts, logger, encoder, osobnik_id)
    else:
        encoder = CGP_Encoder.random_genome(opts)
        while not result:
            encoder = CGP_Encoder.random_genome(opts)
            result = evaluate(opts, logger, encoder, osobnik_id)
    train_and_validate_encoder(opts, logger, encoder, result)

def run_generate_validation_data(opts):
    validation_set = CVRP.make_dataset(size=opts.graph_size, num_samples=opts.val_test_size)
    torch.save(validation_set.data, os.path.join(opts.save_dir, f'dataset_{opts.graph_size}CVRP_seed_{opts.seed}.pt'))
    
    
def run_scoring(opts, logger):
    if not opts.test_set_path:
            raise Exception("Please specify name of test set with --test_set_path")
    
    logger.record(key="genomes", genome = opts.genome)
    encoder = CGP_Encoder(opts, (opts.genome))
    model = AttentionModel(opts, encoder)
    export_cgp_to_graphviz(encoder.genes, opts, os.path.join(opts.save_dir, f"candidate"), only_active=True, paper_style=True)
    checkpoint = torch.load(opts.checkpoint_path, map_location=opts.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    score = validate(model, opts.test_set, opts)
    logger.record(key="scores", score=score)
    print(f"Final score {score}")
    print(f"Active encoder parameters: {encoder.count_active_parameters():,}")


if __name__ == "__main__":
    opts = get_options()
    initial_setup(opts)
    logger = Logger(opts)
    #verify_sanity(opts, logger)
    reset_seeds(opts)
    
    if opts.mode == "cgp_search":    
        run_cgp_search(opts, logger)
    elif opts.mode == "random_search":
        run_random_search(opts, logger)
    elif opts.mode == "generate_validation_data":
        run_generate_validation_data(opts)
    elif opts.mode == "genome_evaluation":
        run_genome_evaluation(opts, logger)
    elif opts.mode == "scoring":
        run_scoring(opts, logger)