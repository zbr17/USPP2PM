import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ClsEmb(nn.Module):
    multiply = 1
    output_hidden_states = False
    trainable = False
    def __init__(self, config):
        super().__init__()

    def forward(self, outputs, data_info):
        last_hidden_state = outputs[0]
        cls_embeddings = last_hidden_state[:, 0]
        return cls_embeddings

class MeanPooling(nn.Module):
    multiply = 1
    output_hidden_states = False
    trainable = False
    def __init__(self, config):
        super().__init__()

    def forward(self, outputs, data_info):
        last_hidden_state = outputs[0]
        attention_mask = data_info["inputs"]["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.sum(input_mask_expanded, dim=1).clamp(min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class MaxPooling(nn.Module):
    multiply = 1
    output_hidden_states = False
    trainable = False
    def __init__(self, config):
        super().__init__()

    def forward(self, outputs, data_info):
        last_hidden_state = outputs[0].clone()
        attention_mask = data_info["inputs"]["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        last_hidden_state[input_mask_expanded == 0] = -1e9
        max_embeddings = torch.max(last_hidden_state, dim=1)[0]
        return max_embeddings

class MeanMaxPooling(nn.Module):
    multiply = 2
    output_hidden_states = False
    trainable = False
    def __init__(self, config):
        super().__init__()

    def forward(self, outputs, data_info):
        last_hidden_state = outputs[0]
        attention_mask = data_info["inputs"]["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # compute mean embeddings
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.sum(input_mask_expanded, dim=1).clamp(min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        # compute max embeddings
        state_clone = last_hidden_state.clone()
        state_clone[input_mask_expanded == 0] = -1e9
        max_embeddings = torch.max(state_clone, dim=1)[0]
        
        embeddings = torch.cat([mean_embeddings, max_embeddings], dim=-1)
        return embeddings

class Identity(nn.Module):
    multiply = 1
    output_hidden_states = False
    trainable = False
    def __init__(self, config):
        super().__init__()
        
    def forward(self, outputs, data_info):
        return outputs[0]


class HiddenCatClsEmb(nn.Module):
    multiply = 4
    output_hidden_states = True
    trainable = False
    def __init__(self, config):
        super().__init__()

    def forward(self, outputs, data_info):
        all_hidden_states = torch.stack(outputs[1])
        cat_pool_states = torch.cat([all_hidden_states[-idx] for idx in range(1, 5)], dim=-1)
        cls_embeddings = cat_pool_states[:, 0]
        return cls_embeddings

class Hidden2MeanMaxPooling(nn.Module):
    multiply = 2
    output_hidden_states = True
    trainable = False
    def __init__(self, config):
        super().__init__()

    def forward(self, outputs, data_info):
        all_hidden_states = torch.stack(outputs[1])
        hidden_state = all_hidden_states[-2]
        attention_mask = data_info["inputs"]["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        # compute mean embeddings
        sum_embeddings = torch.sum(hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.sum(input_mask_expanded, dim=1).clamp(min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        # compute max embeddings
        state_clone = hidden_state.clone()
        state_clone[input_mask_expanded == 0] = -1e9
        max_embeddings = torch.max(state_clone, dim=1)[0]
        
        embeddings = torch.cat([mean_embeddings, max_embeddings], dim=-1)
        return embeddings

class Hidden2ClsMeanMaxPooling(nn.Module):
    multiply = 3
    output_hidden_states = True
    trainable = False
    def __init__(self, config):
        super().__init__()

    def forward(self, outputs, data_info):
        all_hidden_states = torch.stack(outputs[1])
        hidden_state = all_hidden_states[-2]
        cls_state = hidden_state[:, 0]
        other_hidden_state = hidden_state[:, 1:]
        attention_mask = data_info["inputs"]["attention_mask"]
        input_mask_expanded = attention_mask[:, 1:].unsqueeze(-1).expand(other_hidden_state.size()).float()
        # compute mean embeddings
        sum_embeddings = torch.sum(other_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.sum(input_mask_expanded, dim=1).clamp(min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        # compute max embeddings
        state_clone = other_hidden_state.clone()
        state_clone[input_mask_expanded == 0] = -1e9
        max_embeddings = torch.max(state_clone, dim=1)[0]
        
        embeddings = torch.cat([cls_state, mean_embeddings, max_embeddings], dim=-1)
        return embeddings

class HiddenWeightedClsEmb(nn.Module):
    multiply = 1
    output_hidden_states = True
    trainable = True
    def __init__(self, config):
        super().__init__()
        self.layer_start = 9
        self.num_hidden_layers = 12
        self.weights = nn.Parameter(torch.tensor([1] * (
            self.num_hidden_layers + 1 - self.layer_start
        ), dtype=torch.float))
    
    def forward(self, outputs, data_info):
        all_hidden_states = torch.stack(outputs[1])
        cat_pool_states = all_hidden_states[self.layer_start:, :, :, :]
        weights = self.weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(cat_pool_states.size())

        weighted_states = (weights * cat_pool_states).sum(dim=0) / self.weights.sum()

        cls_embeddings = weighted_states[:, 0]
        return cls_embeddings

class HiddenLSTMClsEmb(nn.Module):
    multiply = 1
    output_hidden_states = True
    trainable = True
    def __init__(self, config):
        super().__init__()
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.output_dim
        self.lstm_dim = config.output_dim
        self.lstm = nn.LSTM(self.hidden_size, self.lstm_dim, batch_first=True)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, outputs, data_info):
        all_hidden_states = torch.stack(outputs[1])
        hidden_states = torch.stack([
            all_hidden_states[idx][:, 0].squeeze()
            for idx in range(1, self.num_hidden_layers+1)
        ], dim=-1)
        hidden_states = hidden_states.permute(0, 2, 1)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        return out

class HiddenAttentionClsEmb(nn.Module):
    multiply = 1
    output_hidden_states = True
    trainable = True
    def __init__(self, config):
        super().__init__()
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.output_dim
        self.att_dim = config.output_dim
        self.dropout = nn.Dropout(0.1)

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.att_dim))
        self.q = nn.Parameter(torch.from_numpy(q_t).float())
        self.w_h = nn.Parameter(torch.from_numpy(w_ht).float())
        # self.q = nn.Parameter(torch.randn(1, self.hidden_size))
        # self.w_h = nn.Parameter(torch.randn(self.hidden_size, self.att_dim))
    
    def attention(self, h):
        att_mat = torch.matmul(self.q, h.transpose(-1, -2)).squeeze(1)
        att_mat = F.softmax(att_mat, dim=-1)
        v = torch.matmul(att_mat.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v).squeeze(2)
        return v
    
    def forward(self, outputs, data_info):
        all_hidden_states = torch.stack(outputs[1])
        hidden_states = torch.stack([
            all_hidden_states[idx][:, 0].squeeze()
            for idx in range(1, self.num_hidden_layers+1)
        ], dim=-1)
        hidden_states = hidden_states.permute(0, 2, 1)
        out = self.attention(hidden_states)
        out = self.dropout(out)
        return out

class HiddenBranchMeanMaxPooling(nn.Module):
    multiply = 2
    output_hidden_states = True
    trainable = False
    def __init__(self, config):
        super().__init__()
        self.num_block = config.num_block
    
    def compute_meanmax(self, data, mask):
        # compute mean embeddings
        sum_embeddings = torch.sum(data * mask, dim=1)
        sum_mask = torch.sum(mask, dim=1).clamp(min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        # compute max embeddings
        state_clone = data.clone()
        state_clone[mask == 0] = -1e9
        max_embeddings = torch.max(state_clone, dim=1)[0]
        
        embeddings = torch.cat([mean_embeddings, max_embeddings], dim=-1)
        return embeddings

    def forward(self, outputs, data_info):
        all_hidden_states = torch.stack(outputs[1])
        attention_mask = data_info["inputs"]["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(all_hidden_states[-1].size()).float()
        output_list = []
        for idx in range(-self.num_block, 0):
            hidden_state = all_hidden_states[idx]
            embeddings = self.compute_meanmax(hidden_state, input_mask_expanded)
            output_list.append(embeddings)
        return output_list

_handler_dict = {
    "cls_emb": ClsEmb,
    "mean_pooling": MeanPooling,
    "max_pooling": MaxPooling,
    "mean_max_pooling": MeanMaxPooling,
    "hidden_cat_cls_emb": HiddenCatClsEmb,
    "hidden_2_mean_max_pooling": Hidden2MeanMaxPooling,
    "hidden_2_cls_mean_max_pooling": Hidden2ClsMeanMaxPooling,
    "hidden_weighted_cls_emb": HiddenWeightedClsEmb,
    "hidden_lstm_cls_emb": HiddenLSTMClsEmb,
    "hidden_attention_cls_emb": HiddenAttentionClsEmb,
    "hidden_branch_mean_max_pooling": HiddenBranchMeanMaxPooling
}

def give_handler(config):
    handler_name = config.handler_name
    _meta_handler = _handler_dict[handler_name]
    return _meta_handler