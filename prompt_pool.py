import jittor as jt
import jittor.nn as nn

class PromptPool(nn.Module):
    def __init__(self, pool_size=10, prompt_len=5, embed_dim=768, use_prompt_key=False,
                 top_k=5, prompt_init='uniform', prompt_key_init='uniform', batchwise_prompt = False):
        super().__init__()
        self.pool_size = pool_size
        self.prompt_len = prompt_len
        self.embed_dim = embed_dim
        self.use_prompt_key = use_prompt_key
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        pool_shape = (pool_size, prompt_len, embed_dim)

        if prompt_init == 'uniform':
            self.prompts = jt.init.uniform(pool_shape, low=-1.0, high=1.0)

        if self.use_prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'uniform':
                self.prompt_key = jt.init.uniform(key_shape, low=-1.0, high=1.0)
        else:
            self.prompt_key = self.prompts.mean(dim = 1)

    def execute(self, x_embed, cls_features):
        out = dict()

        prompt_query = cls_features

        query_norm = prompt_query / jt.norm(prompt_query, p=2, dim=1, keepdims=True)  # (B, embed_dim)
        key_norm = self.prompt_key / jt.norm(self.prompt_key, p=2, dim=1, keepdims=True)  # (pool_size, embed_dim)

        similarity = query_norm @ key_norm.transpose(0, 1)  # (B, pool_size)

        selected_sim, indices = jt.topk(similarity, k=self.top_k, dim=1)

        if self.batchwise_prompt:
            prompt_id, _, id_counts = jt.unique(indices, return_inverse=True, return_counts=True)
            _, idx = jt.topk(prompt_id, k = self.top_k)
            indices = prompt_id[idx]

            # selected_indices = indices

            indices = indices.expand(prompt_query.shape[0], -1)

            prompts = self.prompts[indices]
            prompts = prompts.reshape(prompts.shape[0], -1, prompts.shape[-1])

            query_norm = query_norm.unsqueeze(1)
            selected_key_norm = key_norm[indices]

            sim = jt.sum(query_norm * selected_key_norm) / prompt_query.shape[0]


        else:
            prompts = self.prompts[indices]
            prompts = prompts.reshape(prompts.shape[0], -1, prompts.shape[-1])
            sim = jt.sum(selected_sim) / selected_sim.shape[0]

            # selected_indices = indices.flatten()

        out['prompted_embedding'] = jt.concat([prompts, x_embed], dim = 1)
        out['pull_loss'] = sim
        out['total_prompt_len'] = self.top_k * self.prompt_len
        # out['selected_indices'] = selected_indices

        return out
