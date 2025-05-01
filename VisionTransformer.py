import jittor as jt
import jittor.nn as nn
from jittor import init
from functools import partial
from jt_extension import Mlp, PatchEmbed, apply_recursive
from jittor.init import trunc_normal_
from prompt_pool import PromptPool


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        assert (dim % num_heads == 0)
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias = qkv_bias)
        self.scale = (self.head_dim) ** (-0.5)
        self.proj = nn.Linear(dim, dim)

    def execute(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1,
                                                                                      4)  # (3, B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = nn.softmax(attn, dim=-1)

        output = (attn @ v)  # (B, num_heads, N, head_dim)
        output = output.transpose(1, 2).reshape(B, N, -1)

        output = self.proj(output)

        return output


class Block(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, mlp_ratio=4.0, init_values=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def execute(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, mlp_ratio=4.0, num_heads=8, depth=12,
                 embed_layer=PatchEmbed, block_fn=Block, qkv_bias=True, use_prompt_pool=False,
                 use_prompt_key=False, pool_size=10, top_k=5, prompt_len=5, init_values=None,
                 use_class_token=True, num_classes=1000, prompt_init='uniform', prompt_key_init='uniform',
                 batchwise_prompt=False):
        super().__init__()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)

        self.blocks = nn.Sequential(*[block_fn(dim=embed_dim, num_heads=num_heads, init_values=init_values,
                                               qkv_bias=qkv_bias, mlp_ratio=mlp_ratio) for i in range(depth)])

        if use_class_token:
            self.cls_token = jt.zeros((1, 1, embed_dim))

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        embed_len = num_patches

        if use_class_token:
            embed_len += 1

        if use_prompt_pool:
            embed_len += top_k * prompt_len

        self.pos_embed = jt.randn((1, embed_len, embed_dim)) * .02

        if use_prompt_pool:
            self.prompt_pool = PromptPool(prompt_len=prompt_len, embed_dim=embed_dim,
                                          use_prompt_key=use_prompt_key, pool_size=pool_size,
                                          top_k=top_k, prompt_init=prompt_init,
                                          prompt_key_init=prompt_key_init, batchwise_prompt=batchwise_prompt)

        self.head = nn.Linear(embed_dim, num_classes)

        self.init_weights()

    def forward_features(self, x, cls_features=None):
        x_embed = self.patch_embed(x)  # (B, N, C)
        out = dict()

        if hasattr(self, 'prompt_pool'):
            out = self.prompt_pool(x_embed, cls_features=cls_features)
            self.total_prompt_len = out['total_prompt_len']
            x_embed = out['prompted_embedding']

        if self.cls_token is not None:
            x_embed = jt.concat([self.cls_token.expand(x_embed.shape[0], -1, -1), x_embed], dim=1)

        x_embed = x_embed + self.pos_embed

        x = self.blocks(x_embed)
        # x = self.norm(x)

        out['x'] = x

        return out

    def forward_head(self, res):
        x = res['x']

        if hasattr(self, 'prompt_pool'):
            prompts = x[:, 1:self.total_prompt_len + 1]
            prompts_mean = prompts.mean(dim=1)
            head_input = prompts_mean
        else:
            head_input = x[:, 0]

        res['pre_logits'] = head_input

        logits = self.head(head_input)

        res['logits'] = logits

        return res

    def execute(self, x, task_id=-1, cls_features=None, train=True):
        res = self.forward_features(x, cls_features)
        res = self.forward_head(res)

        return res

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)

        if hasattr(self, 'cls_token') and self.cls_token is not None:
            init.gauss_(self.cls_token, std=1e-6)

        apply_recursive(init_weights_vit, self)


def init_weights_vit(module: nn.Module, name: str = ''):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zero_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()