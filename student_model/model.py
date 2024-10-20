import torch
from torch import nn, optim
from transformers import GPT2Model, GPT2Config

class GPT2FeaturePrediction(nn.Module):
    def __init__(self, config):
        super(GPT2FeaturePrediction, self).__init__()
        # self.gpt2 = GPT2Model.from_pretrained("gpt2")
        self.gpt2 = GPT2Model(GPT2Config())

        hidden_dim = self.gpt2.config.hidden_size
        VISION_DIM = config["vision_dim"]
        VISION_POOL_DIM = config["vision_pool_dim"]
        EMBED_DIM = config["embed_dim"]
        FEAT_DIM = config["feat_dim"]
        self.project_text = nn.Linear(EMBED_DIM, hidden_dim)
        self.project_feature = nn.Linear(FEAT_DIM, hidden_dim)
        self.project_vision = nn.Linear(VISION_DIM, hidden_dim)
        self.project_vision_pool = nn.Linear(VISION_POOL_DIM, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, FEAT_DIM)

    def forward(self, text, image, image_pool, prev_feature, mask):
        text = self.project_text(text)
        image = self.project_vision(image)
        image_pool = self.project_vision_pool(image_pool)
        prev_feature = self.project_feature(prev_feature)
        combined_input = torch.cat([text, image_pool, image, prev_feature], dim=1)  # (B, seq_L, hidden_dim)
        B, seq_len = combined_input.shape[0], combined_input.shape[1]
        assert mask.shape == (B, seq_len), (mask.shape, B, seq_len, text.shape, image.shape, image_pool.shape, prev_feature.shape)
        gpt2_output = self.gpt2(inputs_embeds=combined_input, attention_mask=mask).last_hidden_state
        output_feature = self.linear_out(gpt2_output)
        return output_feature


from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerFeaturePrediction(nn.Module):
    def __init__(self, config):
        super(TransformerFeaturePrediction, self).__init__()
        
        VISION_DIM = config["vision_dim"]
        VISION_POOL_DIM = config["vision_pool_dim"]
        EMBED_DIM = config["embed_dim"]
        FEAT_DIM = config["feat_dim"]
        
        hidden_dim = FEAT_DIM
        num_layers = 12
        num_heads = 8
        dim_feedforward = 4096
        dropout = 0.1

        self.project_text = nn.Linear(EMBED_DIM, hidden_dim)
        self.project_feature = nn.Linear(FEAT_DIM, hidden_dim)
        self.project_vision = nn.Linear(VISION_DIM, hidden_dim)
        self.project_vision_pool = nn.Linear(VISION_POOL_DIM, hidden_dim)
        
        encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # self.linear_out = nn.Linear(hidden_dim, FEAT_DIM)

    def forward(self, text, image, image_pool, prev_feature, mask):
        text = self.project_text(text)
        image = self.project_vision(image)
        image_pool = self.project_vision_pool(image_pool)
        prev_feature = self.project_feature(prev_feature)
        
        combined_input = torch.cat([text, image_pool, image, prev_feature], dim=1)  # (B, seq_L, hidden_dim)
        combined_input = combined_input.permute(1, 0, 2)  # Transformer expects input shape (seq_L, B, hidden_dim)
        transformer_output = self.transformer_encoder(combined_input, src_key_padding_mask=mask).permute(1, 0, 2)
        
        # output_feature = self.linear_out(transformer_output)
        output_feature = transformer_output
        
        return output_feature