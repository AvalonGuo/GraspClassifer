import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FilmLayer(Module):
    """FiLM layer: y = gamma * x + beta"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, gamma, beta):
        # x: [B, N, D]
        # gamma, beta: [B, D]
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)

class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FilmLayer(dim),                     # ← 新增 FiLM 层
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))
    def forward(self, x, film_params=None):
        """
        x: [B, N, D]
        film_params: list of (gamma, beta) tuples, length = depth
                     each gamma/beta: [B, D]
        """
        if film_params is None:
            # 默认无调制（兼容旧接口）
            B, _, D = x.shape
            film_params = [(torch.ones(B, D, device=x.device), torch.zeros(B, D, device=x.device)) for _ in range(len(self.layers))]

        for i, (attn, film, ff) in enumerate(self.layers):
            x = attn(x) + x
            gamma, beta = film_params[i]
            x = film(x, gamma, beta)   # ← FiLM 调制
            x = ff(x) + x

        return self.norm(x)



class ViLT(Module):
    """
    Vision-Language transformer
        :a Variance of ViT, supports multimodel input (image,language) pair. 
    """
    def __init__(self, *, image_size, patch_size, num_classes, lang_dim, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_cls_tokens = 1 if pool == 'cls' else 0

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),          #token embedding
            nn.LayerNorm(dim),
        )

        self.cls_token = nn.Parameter(torch.randn(num_cls_tokens, dim))
        self.pos_embedding = nn.Parameter(torch.randn(num_patches + num_cls_tokens, dim))
        self.dropout = nn.Dropout(emb_dropout)

        if lang_dim is not None:
            self.use_film = True
            self.lang_gamma_projs = ModuleList([nn.Linear(lang_dim, dim) for _ in range(depth)])
            self.lang_beta_projs = ModuleList([nn.Linear(lang_dim, dim) for _ in range(depth)])
            
            # 🔥 关键：初始化 gamma 为 1，beta 为 0
            for proj in self.lang_gamma_projs:
                nn.init.zeros_(proj.weight)
                nn.init.ones_(proj.bias)
            for proj in self.lang_beta_projs:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
        else:
            self.use_film = False

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img, lang_embed=None):
        """
        img: [B, C, H, W]
        lang_embed: [B, L]  (optional, language embedding)
        """
        batch = img.shape[0]
        x = self.to_patch_embedding(img)

        cls_tokens = repeat(self.cls_token, '... d -> b ... d', b=batch)
        x = torch.cat((cls_tokens, x), dim=1)

        seq = x.shape[1]
        x = x + self.pos_embedding[:seq]
        x = self.dropout(x)

        # --- 生成 FiLM 参数 ---
        film_params = None
        if self.use_film:
            assert lang_embed is not None, "lang_embed must be provided when use_film=True"
            film_params = []
            for i in range(len(self.lang_gamma_projs)):
                # 在 forward 中
                gamma = self.lang_gamma_projs[i](lang_embed)
                beta = self.lang_beta_projs[i](lang_embed)

                # 微调模式：小扰动
                gamma = 1.0 + 0.1 * torch.tanh(gamma)
                beta = 0.1 * torch.tanh(beta)
                film_params.append((gamma, beta))

        x = self.transformer(x, film_params=film_params)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


if __name__ == "__main__":

    # 假设语言模型输出维度为 512
    vit = ViLT(
        image_size=224,
        patch_size=16,
        num_classes=2,
        dim=768,
        depth=6,
        heads=12,
        mlp_dim=3072,
        lang_dim=512,  # ← 启用 FiLM
        dropout=0.1,
        emb_dropout=0.1
    )

    img = torch.randn(4, 3, 224, 224)
    lang = torch.randn(4, 512)  # 例如 T5 或 CLIP 文本编码器的输出

    net = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=224, p2=224)
    out = net(img)
    net2 = nn.Linear(3*224*224, 768)
    out2 = net2(out)
    cls_token = nn.Parameter(torch.randn(1,768))
    cls_tokens = repeat(cls_token, '... d -> b ... d', b=4)
    x = torch.cat((cls_tokens, out2), dim=1)
    pos_embedding = nn.Parameter(torch.randn(196 + 1, 768))
