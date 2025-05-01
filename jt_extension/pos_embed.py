import math
import jittor as jt
import jittor.nn as nn


def resize_pos_embed(pos_embed, pos_embed_new, num_prefix=1, new_grid_size=()):
    new_num_tokens = pos_embed_new.shape[1]

    pos_embed_prefix, pos_embed_grid = pos_embed[0, :num_prefix], pos_embed[0, num_prefix:]

    old_grid_size = int(math.sqrt(len(pos_embed_grid)))

    if new_num_tokens > old_grid_size ** 2:
        new_num_tokens -= old_grid_size ** 2

        pos_embed_prefix = pos_embed_prefix.unsqueeze(0)
        pos_embed_prefix = pos_embed_prefix.expand(-1, new_num_tokens, -1)

    pos_embed_grid = pos_embed_grid.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
    pos_embed_grid = nn.interpolate(pos_embed_grid, size=new_grid_size, mode='bicubic', align_corners=False)
    pos_embed_grid = pos_embed_grid.permute(0, 2, 3, 1)
    pos_embed_grid = pos_embed_grid.reshape(1, new_grid_size[0] * new_grid_size[1], -1)

    pos_embed = jt.concat([pos_embed_prefix, pos_embed_grid], dim=1)
    return pos_embed
