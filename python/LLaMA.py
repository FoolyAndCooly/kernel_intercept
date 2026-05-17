import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(12046)

hidden_size = 512
num_heads = 8
num_kv_heads = 8
num_layers = 12
intermediate_size = 1376
sequence_len = 128
vocab_size = 65
rms_norm_eps = 1e-5
rope_theta = 10000.0


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def precompute_freqs_cis(dim, seq_len, theta=10000.0, device=None):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# PLACEHOLDER_LLAMA_CLASSES

class LlamaMLP(nn.Module):

    def __init__(self, dim, intermediate_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaBlock(nn.Module):

    def __init__(self, dim, n_heads, n_kv_heads, intermediate_dim, seq_len, eps, theta):
        super().__init__()
        self.attention = LlamaAttention(dim, n_heads, n_kv_heads, seq_len, theta)
        self.feed_forward = LlamaMLP(dim, intermediate_dim)
        self.attention_norm = RMSNorm(dim, eps)
        self.ffn_norm = RMSNorm(dim, eps)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class CharLLaMA(nn.Module):

    def __init__(self, vs):
        super().__init__()
        self.token_embedding = nn.Embedding(vs, hidden_size)
        self.layers = nn.Sequential(*[
            LlamaBlock(
                hidden_size, num_heads, num_kv_heads,
                intermediate_size, sequence_len, rms_norm_eps, rope_theta
            ) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size, rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vs, bias=False)

    def forward(self, x):
        B, T = x.shape
        x = self.token_embedding(x)
        x = self.layers(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


def create_llama(vocab_sz, device='cuda'):
    model = CharLLaMA(vocab_sz).to(device)
    model.eval()
    return model


def llama_dummy_loop(batchsize, train, num_iters, rps, uniform, dummy_data,
                     local_rank, barriers, client_barrier, tid, input_file='',
                     model_name=None):
    import ctypes
    del rps, uniform, dummy_data, local_rank, client_barrier, input_file, train
    try:
        print(f"[llama_dummy {tid}] starting loop (num_iters={num_iters})", flush=True)
        lib_path = (os.environ.get("ORION_HOME", os.path.expanduser("~") + "/orion")
            + "/src/cuda_capture/libinttemp.so")
        RTLD_NOLOAD = 4
        try:
            backend_lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL | RTLD_NOLOAD)
        except OSError:
            backend_lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)

        barriers[0].wait()

        device = "cuda"
        model = create_llama(vocab_size, device)
        dummy = torch.randint(0, vocab_size, (batchsize, sequence_len), device=device)

        torch.cuda.nvtx.range_push(f"llama_warmup_{tid}")
        _ = model(dummy)
        torch.cuda.nvtx.range_pop()
        time.sleep(0.1)
        backend_lib.block(0)
        torch.cuda.synchronize()

        barriers[0].wait()

        main_iters = max(num_iters - 1, 1)
        for i in range(main_iters):
            torch.cuda.nvtx.range_push(f"llama_iter_{i}_{tid}")
            _ = model(dummy)
            torch.cuda.nvtx.range_pop()
            backend_lib.block(i + 1 if num_iters > 1 else 0)
            torch.cuda.synchronize()

        barriers[0].wait()
        print(f"[llama_dummy {tid}] done", flush=True)
    except Exception as exc:
        print(f"[llama_dummy {tid}] exception: {exc}", flush=True)
        raise


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_llama(vocab_size, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CharLLaMA: {n_params/1e6:.1f}M parameters")
    dummy = torch.randint(0, vocab_size, (2, sequence_len), device=device)
    with torch.no_grad():
        out = model(dummy)
    print(f"Input: {dummy.shape}, Output: {out.shape}")


class LlamaAttention(nn.Module):

    def __init__(self, dim, n_heads, n_kv_heads, seq_len, theta=10000.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.register_buffer(
            'freqs_cis', precompute_freqs_cis(self.head_dim, seq_len, theta))
        self.register_buffer(
            'mask', torch.tril(torch.ones(seq_len, seq_len)))

    def forward(self, x):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, self.freqs_cis[:T])

        # GQA: repeat KV heads
        if self.n_rep > 1:
            k = k.unsqueeze(3).expand(B, T, self.n_kv_heads, self.n_rep, self.head_dim)
            k = k.reshape(B, T, self.n_heads, self.head_dim)
            v = v.unsqueeze(3).expand(B, T, self.n_kv_heads, self.n_rep, self.head_dim)
            v = v.reshape(B, T, self.n_heads, self.head_dim)

        # (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)

