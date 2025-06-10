class MultiHeadAttentionEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.embed(x).unsqueeze(1)  # [B, 1, D]
        attn_output, _ = self.attn(x, x, x)
        return self.norm(attn_output.squeeze(1))