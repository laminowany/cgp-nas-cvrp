import copy
import math
import torch
import random
import time 
from collections import defaultdict, deque
from dataclasses import dataclass

from torch import nn

class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # Alternative:
        # headst = heads.transpose(0, 1)  # swap the dimensions for batch and heads to align it for the matmul
        # # proj_h = torch.einsum('bhni,hij->bhnj', headst, self.W_out)
        # projected_heads = torch.matmul(headst, self.W_out)
        # out = torch.sum(projected_heads, dim=1)  # sum across heads

        # Or:
        # out = torch.einsum('hbni,hij->bnj', heads, self.W_out)

        return out

class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input
    
class Add(nn.Module):
    """Simple adding for making skip connection possible"""
    def __init__(self, input_dims, embed_dim):
        super(Add, self).__init__()
        self.embed_dim = embed_dim

        self.projections = nn.ModuleList([
            nn.Linear(dim, embed_dim)
            if dim != embed_dim
            else nn.Identity()
            for dim in input_dims
        ])

    def forward(self, xs):
        projected = [ proj(x) for x, proj in zip(xs, self.projections) ]
        return sum(projected)
    
@dataclass
class Gene:
    pos: int
    type: int
    inputs: list[int]
    args: list[int]
    active: bool = False
    
    def encode(self):
        inputs = self.inputs
        if len(inputs) == 1:
            inp_repr = inputs[0]
        else:
            inp_repr = tuple(inputs)
        return (self.type, inp_repr, *self.args)
    
@dataclass
class CGP_Element:
    nn: nn.Module
    dim: int
    active: bool = False

def parse_gene(t, pos):
    if not t:
        return None
    type_ = t[0]
    rest = list(t[1:])
    if not rest:
        raise ValueError(f"Gene {t} has no inputs")
    first = rest[0]
    if isinstance(first, tuple):
        inputs = list(first)
        args = rest[1:]
    else:
        inputs = [first]
        args = rest[1:]
    return Gene(pos, type_, inputs, args)

def parse_genes(data):
    return [parse_gene(t, idx) for idx, t in enumerate(data)]

# 1 - Identity
# 2 - Normalization
# 3 - Linear scaling
# 4 - MultiHeadAttention
# 5 - Add
# 6 - Gelu
# 7 - Relu
GENE_TYPES_LEN = 7

# (TYP, (INPUTY), (PARAMS))
class CGP_Encoder(nn.Module):
    def __init__(self, opts, genome): 
        super().__init__()
        self.num_heads = 8
        self.feed_forward_hidden = 512
        self.x_dim = opts.x_dim
        self.y_dim = opts.y_dim
        self.len = self.x_dim * self.y_dim + 2
        self.opts = opts
        self.embed_dim = opts.embedding_dim
        self.debug = opts.debug
        self.genome = genome
        assert len(self.genome) == self.len
        self.genes = parse_genes(self.genome)
        
        self.nets = [CGP_Element(None, self.embed_dim), *[None] * (self.len-1)]
        for x in range(self.x_dim):
            for y in range(self.y_dim):
                idx = self.get_global_idx(x, y)
                if not self.genes[idx]:
                    continue
                self.nets[idx] = self.produce_net(self.genes[idx])
        self.nets[self.len - 1] = self.produce_net(self.genes[self.len - 1])
        
        assert len(self.genes) == self.len
        assert len(self.nets) == self.len
        self.mark_active_paths()
        self.build_propagation_order()


    def forward(self, x):
        outputs = [None]*self.len
        outputs[0] = x    
        if self.debug:
            print("\n========== CGP FORWARD ==========")
            print(
                f"[INPUT] "
                f"shape={x.shape} "
                f"mean={x.mean().item():.4f} "
                f"std={x.std().item():.4f} "
                f"first={x.flatten()[0].item():.4f}"
            )
        for idx in self.propagation_order:
            if idx == 0:
                continue
            inputs = self.genes[idx].inputs
            in_vals = [outputs[i] for i in inputs]
            if self.genes[idx].type == 5:
                outputs[idx] = self.nets[idx].nn(in_vals)
            elif len(in_vals) == 1:
                outputs[idx] = self.nets[idx].nn(in_vals[0])
            else:
                outputs[idx] = self.nets[idx].nn(in_vals)
            if self.debug:
                out = outputs[idx] 
                if isinstance(out, torch.Tensor):
                    print(
                        f"TYP {self.genes[idx].type} output "
                        f"shape={out.shape} "
                        f"mean={out.mean().item():.4f} "
                        f"std={out.std().item():.4f} "
                        f"first={out.flatten()[0].item():.4f}"
                    )

                    if torch.isnan(out).any():
                        print(" !!! NAN DETECTED !!! ")

                else:
                    print(f" output type={type(out)}")
        final_h = outputs[self.len-1]
        graph_embedding = final_h.mean(dim=1)
        if self.debug:
            print("\n========== FINAL ==========")
            print(
                f"final_h "
                f"shape={final_h.shape} "
                f"mean={final_h.mean().item():.4f} "
                f"std={final_h.std().item():.4f} "
            )
            print(
                f"graph_embedding "
                f"shape={graph_embedding.shape} "
                f"mean={graph_embedding.mean().item():.4f} "
                f"std={graph_embedding.std().item():.4f} "
                f"norm={graph_embedding.norm().item():.4f}"
            )
            print("=================================\n")
        return (final_h, graph_embedding)

    def mark_active_paths(self):
        visited = set()
        queue = deque()
        queue.append(self.len - 1)

        while queue:
            pos = queue.popleft()
            if pos == 0 or pos in visited:
                continue
            self.nets[pos].active  = True
            self.genes[pos].active  = True
            visited.add(pos)
            queue.extend(self.genes[pos].inputs)

    def build_propagation_order(self):
        order =[]
        for x in range(self.x_dim):
            for y in range(self.y_dim): 
                idx = self.get_global_idx(x, y)
                if self.nets[idx] and self.nets[idx].active:
                    order.append(idx)
        order.append(self.len - 1)
        self.propagation_order = order
        
    def get_global_idx(self, x, y):
        return CGP_Encoder.to_global_idx(x, y, self.x_dim, self.y_dim)
    
    @staticmethod
    def to_global_idx(x, y, x_dim, y_dim):
        if x == x_dim:
            return x_dim*y_dim + 1
        return y*x_dim + x + 1

    def to_xy(self, pos):
        if pos == self.len - 1:
            return (self.x_dim, 0)
        pos -= 1
        x = pos % self.x_dim
        y = pos // self.x_dim
        return (x, y)
    
    def spawn_random_gene(self, x, y):
        pos = self.get_global_idx(x, y)
        type = random.randint(1, GENE_TYPES_LEN)
        args = []
        inputs = []
        
        if x == 0:     
            possible_inputs = [0]
        else:
            possible_inputs = list(map(lambda py: self.get_global_idx(x - 1, py) ,range(self.y_dim)))
            
        prob_input = 1
        available = possible_inputs.copy()
        while random.random() < prob_input and available:
            inp = random.choice(available)
            inputs.append(inp)
            available.remove(inp)
            if type != 5:
                prob_input = 0
            else:
                prob_input *= 0.5
            
        if type == 3:
            args = [random.randint(-1, 1)]
            
        return Gene(pos, type, inputs, args)
    
    
    def mutate_gene_inputs(self, x, y, opts):
        pos = self.get_global_idx(x, y)
        type = self.genes[pos].type
        args = self.genes[pos].args
        
        inputs = []
        if x == 0:     
            possible_inputs = [0]
        else:
            if opts.deep_neural_connection:
                possible_inputs = [
                    self.get_global_idx(px, py)
                    for px in range(x)
                    for py in range(self.y_dim)
                ]
            else:
                possible_inputs = list(map(lambda py: self.get_global_idx(x - 1, py), range(self.y_dim)))
        
        inputs = []  
        prob_input = 1
        available = possible_inputs.copy()
        while random.random() < prob_input and available:
            inp = random.choice(available)
            inputs.append(inp)
            available.remove(inp)
            if type != 5:
                prob_input = 0
            else:
                prob_input *= 0.5
        return Gene(pos, type, inputs, args)
    
    def mutate_gene_type(self, x, y):
        pos = self.get_global_idx(x, y)
        inputs = self.genes[pos].inputs
        possible_types = range(1, GENE_TYPES_LEN + 1)
        possible_types = [x for x in possible_types if x !=  self.genes[pos].type]
        type = random.choice(possible_types)
        args = []
        if type == 3:
            args = [random.randint(-1, 1)]
        if type != 5 and len(inputs) > 1:
            inputs = inputs[:1]
            
        return Gene(pos, type, inputs, args)

    def produce_net(self, gene: Gene):
        first_input_dim = self.embed_dim
        if gene.inputs[0] == 0:
            first_input_dim = self.embed_dim
        else:
            first_input_dim = self.nets[gene.inputs[0]].dim
        output_dim = first_input_dim
        if gene.type == 1:
            net = nn.Identity()
        elif gene.type == 2:
            net = Normalization(first_input_dim)
        elif gene.type == 3:
            scaling = gene.args[0]
            if scaling == 1 and first_input_dim <= 1024:
                output_dim = first_input_dim * 4
                net = nn.Linear(first_input_dim, output_dim)
            elif scaling == -1 and first_input_dim >= 32:
                output_dim = first_input_dim // 4
                net = nn.Linear(first_input_dim, output_dim)
            else:
                net = nn.Linear(first_input_dim, first_input_dim)
        elif gene.type == 4:
            net = MultiHeadAttention(self.num_heads, first_input_dim, first_input_dim)
        elif gene.type == 5:
            input_dims = [
                self.nets[i].dim
                for i in gene.inputs
            ]
            net = Add(input_dims, self.embed_dim)
            output_dim = self.embed_dim
        elif gene.type == 6:
            net = nn.GELU()
        elif gene.type == 7:
            net = nn.ReLU()
        elif gene.type == 8:
            net = nn.LayerNorm()
        else:
            raise f'unknown layer type {gene.type}'
        
        self.add_module(f"node_{gene.pos}", net)
        return CGP_Element(net, output_dim)
    
    def produce_offspring(self, n, opts, remaining_budget):
        children = []
        parent_genome = self.genome

        for _ in range(n):
            genome = copy.deepcopy(parent_genome)
  
            for pos in range(1, len(self.genes) - 1):
                x, y = self.to_xy(pos)
                if not self.genes[pos]:
                    new_gene = self.spawn_random_gene(x, y)
                    genome[pos - 1] = new_gene.encode()
                    
            k = 3  # decay speed
            t = remaining_budget / opts.budget
            decay = math.exp(-k * (1 - t))
            mutations_num = max(1, int(0.5 * (len(parent_genome) - 2) * decay))
            mutations = random.sample(range(1, len(parent_genome)), mutations_num)
            for pos in mutations:
                x, y = self.to_xy(pos)
                if pos == len(parent_genome) - 1:
                    genome[pos] = self.mutate_gene_inputs(x, y, opts).encode()
                else:    
                    if random.random() < (self.y_dim / (self.y_dim + GENE_TYPES_LEN)) and x != 0:
                        genome[pos] = self.mutate_gene_inputs(x, y, opts).encode()
                    else:
                        genome[pos] = self.mutate_gene_type(x, y).encode()
        
            child = CGP_Encoder(
                opts=self.opts,
                genome=genome
            )
            children.append(child)
        return children
    
    @staticmethod
    def random_genome(opts):
        dummy = CGP_Encoder.__new__(CGP_Encoder)
        x_dim = opts.x_dim
        y_dim = opts.y_dim
        dummy.x_dim = x_dim
        dummy.y_dim = y_dim
        length = x_dim * y_dim
        genome = [None] * (length + 2) 
        for x in range(x_dim):
            for y in range(y_dim):
                gene = dummy.spawn_random_gene(x, y)
                genome[CGP_Encoder.to_global_idx(x, y, x_dim, y_dim)] = gene.encode()
                
        possible_inputs = list(map(lambda py: CGP_Encoder.to_global_idx(x_dim - 1, py, x_dim, y_dim), range(y_dim)))
        outputs = []
        prob_input = 1
        while random.random() < prob_input and possible_inputs:
            inp = random.choice(possible_inputs)
            outputs.append(inp)
            possible_inputs.remove(inp)
            prob_input *= 0.5
        genome[length + 1] = (5, tuple(outputs))
        
        return CGP_Encoder(
            opts=opts,
            genome=genome
        )
        
    def save_snapshot(self):
        snapshot = {}
        for pos, net_element in enumerate(self.nets):
            if net_element and net_element.nn:
                snapshot[f"net_{pos}"] = {
                    "type": type(net_element.nn).__name__,
                    "state_dict": net_element.nn.state_dict()
                }
        return snapshot

    def load_snapshot(self, snapshot):
        for pos, net_element in enumerate(self.nets):
            key = f"net_{pos}"
            if not net_element or not net_element.nn or key not in snapshot:
                continue
            saved = snapshot[key]
            if saved["type"] != type(net_element.nn).__name__:
                continue
            current = net_element.nn.state_dict()
            compatible = {}
            for k, v in saved["state_dict"].items():
                if k not in current:
                    continue
                if current[k].shape != v.shape:
                    continue
                compatible[k] = v
            net_element.nn.load_state_dict(compatible, strict=False)
    
    def count_active_parameters(self, trainable_only=True):
        total = 0

        for net_element in self.nets:
            if not net_element or not net_element.nn or not net_element.active:
                continue

            for param in net_element.nn.parameters():
                if trainable_only and not param.requires_grad:
                    continue

                total += param.numel()

        return total
    
    def print_active_parameters(self):
        total = 0

        for pos, net_element in enumerate(self.nets):
            if not net_element or not net_element.nn or not net_element.active:
                continue

            params = sum(p.numel() for p in net_element.nn.parameters())
            total += params

            gene_type = self.genes[pos].type if self.genes[pos] else None

            print(
                f"pos={pos:2d} "
                f"type={gene_type} "
                f"module={type(net_element.nn).__name__:20s} "
                f"params={params:,}"
            )

        print(f"TOTAL ACTIVE PARAMS: {total:,}")

    def export_genome(self):
        genome = []

        for pos in range(1, self.len - 1):
            gene = self.genes[pos]
            if gene is None:
                genome.append(None)
                continue
            if len(gene.inputs) == 1:
                inp_repr = gene.inputs[0]
            else:
                inp_repr = tuple(gene.inputs)
            genome.append(
                tuple([gene.type, inp_repr, *gene.args])
            )
        return genome
            
    def __hash__(self):
        res = []
        for pos in range(1, self.len):
            gene = self.genes[pos]
            if not gene or not gene.active:
                continue
            res.append((
                pos,
                gene.type,
                tuple(sorted(gene.inputs)),
                tuple(gene.args)
            ))     
        return hash((
            tuple(res)
        ))
            
    @staticmethod
    def are_equivalent(net_a, net_b):
        return hash(net_a) == hash(net_b)