import os

os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
from transformers import LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import datasets
import transformers
from datasets import load_dataset
import pandas as pd
import time
from collections import OrderedDict
from composer.models import ComposerModel
from transformers import PreTrainedTokenizerBase

from typing import Any, Callable, Optional, Union, cast
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

MASTER_CONFIG = {
    'n_layers': 24,  # 16,
    'epochs': 10,
    'log_interval': 10,
    'vocab_size': 50368,  # len(vocab),
    'd_model': 2048,
    'context_window': 16,  # 256,
    'n_heads': 16,
    'batch_size': 1,
}


def get_config(
        conf_path: str = '/home/jupyter/PS/new3/llm-foundry/scripts/train/yamls/pretrain/gpt-neo-125m.yaml',
) -> DictConfig:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    with open(conf_path) as f:
        test_cfg = om.load(f)

    return cast(DictConfig, test_cfg)

om_cfg = get_config(conf_path='/home/jupyter/PS/new3/llm-foundry/scripts/train/yamls/pretrain/gpt-neo-125m.yaml')    

def build_tokenizer(
        tokenizer_name: str,
        tokenizer_kwargs: dict[str, Any],
) -> PreTrainedTokenizerBase:
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    signal_file_path = dist.get_node_signal_file_name()

    if dist.is_available() and dist.is_initialized(
    ) and dist.get_world_size() > 1:
        # Make sure the tokenizer files are downloaded and cached first by local rank 0
        with dist.local_rank_zero_download_and_wait(signal_file_path):
            pass

    if tokenizer_name in registry.tokenizers:
        tokenizer = construct_from_registry(
            name=tokenizer_name,
            registry=registry.tokenizers,
            partial_function=True,
            pre_validation_function=PreTrainedTokenizerBase,
            post_validation_function=None,
            kwargs=tokenizer_kwargs,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            **tokenizer_kwargs,
        )

        # HuggingFace does not respect the model_max_length kwarg, and overrides it with
        # min(kwargs['model_max_length'], original_config['model_max_length']), so we
        # explicitly set it here
        tokenizer.model_max_length = tokenizer_kwargs.get(
            'model_max_length',
            int(1e30),
        )

    if not hasattr(tokenizer, 'eos_token') or tokenizer.eos_token is None:
        raise ValueError(
            f'The tokenizer {tokenizer_name} must have an eos_token.',
        )

    if dist.is_available() and dist.is_initialized(
    ) and dist.get_world_size() > 1:
        if dist.get_local_rank() == 0:
            with open(signal_file_path, 'wb') as f:
                f.write(b'local_rank0_completed_tokenizer_setup')

        dist.barrier()

        if dist.get_local_rank() == 0:
            os.remove(signal_file_path)

    return tokenizer


def tokenize_function(sample):
    return tokenizer(
        text=sample['text'],
        padding="max_length",
        max_length=2048,
        truncation=True,
    )


def get_batches(data, split, batch_size, context_window, config=MASTER_CONFIG):
    train = data[:int(.8 * len(data))]
    val = data[int(.8 * len(data)): int(.9 * len(data))]
    test = data[int(.9 * len(data)):]

    batch_data = train
    if split == 'val':
        batch_data = val

    if split == 'test':
        batch_data = test

    # pick random starting points
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))
    x = torch.stack([batch_data[i:i + context_window] for i in ix]).long()
    y = torch.stack([batch_data[i + 1:i + context_window + 1] for i in ix]).long()
    return x.get('input_ids'), y.get('input_ids')  # ASM


def evaluate_loss(model, config=MASTER_CONFIG):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = []
        for _ in range(10):
            xb, yb = get_batches(dataset, split, config['global_train_batch_size'], config['model']['max_seq_len'])
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out


def train(model, optimizer, scheduler=None, config=MASTER_CONFIG, print_logs=False):
    losses = []
    start_time = time.time()
    for epoch in range(5):
        optimizer.zero_grad()

        xs, ys = get_batches(dataset, 'train', config['global_train_batch_size'], config['model']['max_seq_len'])
        print(type(xs))
        print(xs)
        logits, loss = model(xs, targets=ys)
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        if epoch % 1 == 0:
            batch_time = time.time() - start_time
            x = evaluate_loss(model)
            losses += [x]
            if print_logs:
                print(
                    f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (5 - epoch) :.3f}")
            start_time = time.time()

            if scheduler:
                print("lr: ", scheduler.get_lr())

    print("validation loss: ", losses[-1]['val'])
    return pd.DataFrame(losses).plot()


import datasets
import transformers
from datasets import load_dataset
import pandas as pd
import time
from torch_sparse import SparseTensor, transpose

# importing sys
import sys

# adding Folder_2/subfolder to the system path
# sys.path.insert(0, '/home/jupyter/llm-foundry3/')
# print(sys.path)
# from structured_nets import pytorch
# from structured_nets.pytorch.structure import toeplitz as toep
# from structured_nets.pytorch.structure import krylov as kry
# from structured_nets.pytorch.structure import circulant as circ
# from structured_nets.pytorch.structure import fastfood as ff
# from structured_nets.pytorch import *
# from structured_nets.pytorch.structure import layer as sl
# from structured_nets.pytorch import utils

# sys.path.insert(0, '../../../../structured-nets/pytorch/')
sys.path.insert(0, '/home/jupyter/llm-foundry3/structured-nets/pytorch/')
print(sys.path)
# import layer as sl

# from llmfoundry.models import toeplitz as toep


import toeplitz as toep
# from structure import krylov as kry
# from structure import circulant as circ
# from structure import fastfood as ff
# from structure import layer as sl
import layer as sl

import utils


class RMSNorm(nn.Module):
    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))

    def forward(self, x):
        """
        assumes shape is (batch, seq_len, d_model)
        """
        # frob norm is not the same as RMS. RMS = 1/sqrt(N) * frob norm
        print("x:")
        print(x.size())
        ff_rms = torch.linalg.norm(x, dim=(1, 2)) * x[0].numel() ** -.5
        print("ff_rms:")
        print(ff_rms.size())
        raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)
        print("raw:")
        print(raw.size())
        # mul=self.scale[:x.shape[1], :].unsqueeze(0) * raw
        # print("mul:")
        # print(mul.size())
        # return self.scale[:x.shape[1], :].unsqueeze(0) * raw
        return raw


def get_rotary_matrix(context_window, embedding_dim):
    R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False).to('cuda')
    for position in range(context_window):
        for i in range(embedding_dim // 2):
            theta = 10000. ** (-2. * (i - 1) / embedding_dim)
            m_theta = position * theta
            R[position, 2 * i, 2 * i] = np.cos(m_theta)
            R[position, 2 * i, 2 * i + 1] = - np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i] = np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i + 1] = np.cos(m_theta)
    return R


class RoPEMaskedAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.class_type = None
        self.config = om_cfg

        diags = config['model']['d_model']
        band = config['model']['d_model']//4
        # offsets_positive = torch.arange(0, diags)
        offsets_negative = torch.arange(0, -band - 2, -1)
        # offsets  = torch.cat([offsets_positive, offsets_negative])
        offsets = offsets_negative
        # print(offsets)
        # print(torch.randn(diags, diags))
        self.s_edge_index = torch.sparse.spdiags(torch.randn(band + 2, diags), offsets,
                                                 (diags, diags)).coalesce().indices()
        self.register_buffer('s_row', self.s_edge_index[0])
        self.register_buffer('s_col', self.s_edge_index[1])
        qval = torch.nn.Parameter(torch.zeros(self.s_edge_index.shape[1]), requires_grad=True)
        kval = torch.nn.Parameter(torch.zeros(self.s_edge_index.shape[1]), requires_grad=True)
        self.w_q = SparseTensor(row=self.s_row, col=self.s_col, value=qval).to('cuda')
        self.w_k = SparseTensor(row=self.s_row, col=self.s_col, value=kval).to('cuda')

        self.w_v = sl.StructuredLinear(class_type='low_rank', layer_size=config['model']['d_model'], r=1).to(
            'cuda')  # int(config['d_model']/2))
        # nn.Linear(config['d_model'], config['d_model'], bias=False)

        self.R = get_rotary_matrix(config['model']['max_seq_len'], config['model']['d_model'])

    #     def get_rotary_matrix(context_window, embedding_dim):
    #         R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False).to('cuda')
    #         for position in range(context_window):
    #             for i in range(embedding_dim//2):
    #                 theta = 10000. ** (-2.*(i - 1) / embedding_dim)
    #                 m_theta = position * theta
    #                 R[position, 2*i,2*i] = np.cos(m_theta)
    #                 R[position, 2*i,2*i+1] = - np.sin(m_theta)
    #                 R[position, 2*i+1,2*i] = np.sin(m_theta)
    #                 R[position, 2*i+1,2*i+1] = np.cos(m_theta)
    #         return R

    def forward(self, x, return_attn_weights=False):
        # print(x.shape)
        b, m, d = x.shape

        # first, make sure the diagonal is zero
        # This improves loss
        #         with torch.no_grad():
        #             self.w_q.weight = nn.Parameter(torch.ones(config['d_model'], config['d_model']).tril())
        #             self.w_k.weight = nn.Parameter(torch.ones(config['d_model'], config['d_model']).triu())

        #             self.w_q.weight.fill_diagonal_(0.)
        #             self.w_k.weight.fill_diagonal_(0.)
        #             self.w_v.weight.fill_diagonal_(0.)

        # q = self.w_q.repeat(x)
        # k = self.w_k.repeat(x)
        v = self.w_v(x)
        q = x @ self.w_q.to_dense()  # self.w_q.repeat(x)
        k = x @ self.w_k.to_dense()  # self.w_k.repeat(x)
        # v = x @self.w_v# self.w_v.repeat(x)

        q_rotated = (torch.bmm(q.transpose(0, 1).to('cuda'), self.R[:m])).transpose(0, 1)
        k_rotated = (torch.bmm(k.transpose(0, 1).to('cuda'), self.R[:m])).transpose(0, 1)

        activations = F.scaled_dot_product_attention(
            q_rotated, k_rotated, v, dropout_p=.1, is_causal=True
        )

        if return_attn_weights:
            attn_mask = torch.tril(torch.ones((m, m)), diagonal=0)
            attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1, 2)).to('cuda') / np.sqrt(d) + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            return activations, attn_weights
        return activations


# definitely there's an optimization we could make where we cache the rotation matrices, but skip.
class RoPEMaskedMultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = om_cfg
        self.heads = nn.ModuleList([
            RoPEMaskedAttentionHead(config) for _ in range(config['model']['n_heads'])
        ])
        self.linear = nn.Linear(config['model']['n_heads'] * config['model']['d_model'], config['model']['d_model'])
        self.dropout = nn.Dropout(.1)

    def forward(self, x):
        heads = [h(x) for h in self.heads]
        x = torch.cat(heads, dim=-1)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit
    https://arxiv.org/pdf/2002.05202v1.pdf
    """

    def __init__(self, size):
        super().__init__()
        # self.config = config
        self.linear_gate = nn.Linear(size, size)
        self.linear = nn.Linear(size, size)
        self.beta = torch.randn(1, requires_grad=True)

        self.beta = nn.Parameter(torch.ones(1))
        self.register_parameter("beta", self.beta)

    def forward(self, x):
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        out = swish_gate * self.linear(x)
        return out


# add RMSNorm and residual conncection
class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = om_cfg
        self.n = config['model']['d_model'] / 4
        self.off_diag = config['model']['d_model'] / 10

        self.rms = RMSNorm(config['model']['max_seq_len'], config['model']['d_model'])

        self.attention = RoPEMaskedMultiheadAttention(config)

        # k = self.n*(2*self.off_diag+1) - self.off_diag*(self.off_diag+1)
        # rows = torch.randint(0, self.n, (k,))
        # cols = torch.randint(0, self.n, (k,))
        # self.f = torch.nn.Parameter(torch.zeros(k), requires_grad=True)
        # self.f_edge_index = torch.stack([rows, cols])
        # self.register_buffer('f_row', self.f_edge_index[0])
        # self.register_buffer('f_col', self.f_edge_index[1])
        # self.linear = SparseTensor(row=self.f_row, col=self.f_col, value=self.f)

        self.linear = sl.StructuredLinear(class_type='low_rank', layer_size=config['model']['d_model'],
                                          r=1)  # int(config['d_model']/2))

        self.feedforward = nn.Sequential(
            self.linear,  # nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['model']['d_model']),
        )

    def forward(self, x):
        x = self.rms(x)  # rms pre-normalization
        x = x + self.attention(x)

        x = self.rms(x)  # rms pre-normalization
        x = x + self.feedforward(x)
        return x


# class Llama(nn.Module):
class Llama(ComposerModel):

    def __init__(self, config, tokenizer_name):
        super().__init__()
        self.config = om_cfg
        # print(type(config))
        # print(config)
        # self.embeddings = nn.Embedding(config.get('vocab_size'), config.get('d_model'))
        self.embeddings = nn.Embedding(config['model']['vocab_size'], config['model']['d_model'])
        print(tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_eos_token=True)  # , token=True)

        # ADDED 20.12.24 ASM
        DEFAULT_PAD_TOKEN = "[PAD]"
        DEFAULT_EOS_TOKEN = "</s>"
        DEFAULT_BOS_TOKEN = "<s>"
        DEFAULT_UNK_TOKEN = "<unk>"

        self.tokenizer.add_special_tokens(
            {
                "pad_token": DEFAULT_PAD_TOKEN,
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.llama_blocks = nn.Sequential(
            OrderedDict([(f"llama_{i}", LlamaBlock(config)) for i in range(config['model']['n_layers'])])
        )

        self.ffn = nn.Sequential(
            sl.StructuredLinear(class_type='low_rank', layer_size=config['model']['d_model'], r=1),
            # nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['model']['d_model']),
            # sl.StructuredLinear(class_type = 'low_rank', layer_size=config['vocab_size'],  r=1),
            # ASM needed to convert to token
            nn.Linear(config['model']['d_model'], config['model']['vocab_size']),
        )

        print("model params:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
        # print(type(idx))
        # print(idx)
        idx = torch.tensor(idx.get('input_ids')).clone().detach()  # ASM
        print("input shape: {}".format(idx.shape))
        x = self.embeddings(idx)

        print("embedding shape: {}".format(x.shape))
        x = self.llama_blocks(x)
        print("post llamablock shape: {}".format(x.shape))
        logits = self.ffn(x)
        print("logits shape :{}".format(x.shape))

        # lg_shape = logits.shape
        # labels = inputs['labels']
        # lb_cpu = labels.cpu()

        if targets is None:
            return logits

        else:
            targets = torch.tensor(targets.get('input_ids')).clone().detach()  # ASM
            print("targets shape :{}".format(targets.shape))
            loss = F.cross_entropy(logits.view(-1, self.config['model']['vocab_size']),
                                   targets.view(-1))  # ASM replaced by below
            # loss = F.cross_entropy(logits.view(-1, self.config.get('d_model')), targets.view(-1))
            return logits, loss

    def loss(self, outputs, targets):
        targets = torch.tensor(targets.get('input_ids')).clone().detach()  # ASM
        print("targets shape :{}".format(targets.shape))

        return F.cross_entropy(outputs.view(-1, self.config['model']['vocab_size']), targets.view(-1), reduction="none")
        # return F.cross_entropy(outputs, targets, reduction="none")


