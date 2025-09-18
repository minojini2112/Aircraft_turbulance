#Self-supervised MAE for flight data pretraining
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    """
    Convert time series segments into patch embeddings for transformer processing
    """
    def __init__(self, seq_len, patch_size, in_channels, embed_dim):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        
        self.projection = nn.Sequential(
            Rearrange('b (n p) c -> b n (p c)', p=patch_size),
            nn.Linear(patch_size * in_channels, embed_dim)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, in_channels)
        x = self.projection(x)
        return x

class FlightDataMAE(nn.Module):
    """
    Masked Autoencoder for flight sensor data pretraining
    Based on the AeroTurb-RL paper architecture
    """
    def __init__(self, 
                 seq_len=100, 
                 patch_size=4, 
                 in_channels=50, 
                 embed_dim=256, 
                 depth=6, 
                 num_heads=8,
                 decoder_embed_dim=128,
                 decoder_depth=4,
                 decoder_num_heads=8,
                 mask_ratio=0.75):
        super().__init__()
        
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = False  # Add this line
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(seq_len, patch_size, in_channels, embed_dim)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Decoder embeddings
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False
        )
        
        # Decoder blocks
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_embed_dim,
            nhead=decoder_num_heads,
            dim_feedforward=decoder_embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)
        
        # Reconstruction head
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * in_channels)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize positional embeddings and other parameters"""
        # Initialize positional embeddings for 1D time series (not 2D images)
        pos_embed = self._get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        decoder_pos_embed = self._get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_patches)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        # Initialize other parameters
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _get_1d_sincos_pos_embed(self, embed_dim, num_patches):
        """Generate 1D sin-cos positional embeddings for time series"""
        assert embed_dim % 2 == 0
        
        # Create position indices (including CLS token, so num_patches + 1)
        pos = np.arange(num_patches + 1, dtype=np.float32)
        
        # Use half of dimensions to encode position
        half_dim = embed_dim // 2
        
        # Create sinusoidal embeddings
        emb = np.log(10000) / (half_dim - 1)
        emb = np.exp(np.arange(half_dim, dtype=np.float32) * -emb)
        emb = pos[:, None] * emb[None, :]
        
        # Concatenate sin and cos
        pos_embed = np.concatenate([np.sin(emb), np.cos(emb)], axis=1)
        
        return pos_embed
    
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # apply Transformer blocks
        x = self.encoder(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        
        # add pos embed
        x = x + self.decoder_pos_embed
        
        # apply Transformer blocks
        memory = x[:, :1, :]  # Use cls token as memory for decoder
        tgt = x[:, 1:, :]     # Patch tokens as target
        x = self.decoder(tgt, memory)
        
        # predictor projection
        x = self.decoder_pred(x)
        
        # remove cls token
        x = x[:, 1:, :]
        
        return x
    
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, L, p*p*3]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        
        # Ensure pred and target have the same size
        if pred.shape[1] != target.shape[1]:
            # If pred is smaller, pad it or slice target to match
            min_patches = min(pred.shape[1], target.shape[1])
            pred = pred[:, :min_patches, :]
            target = target[:, :min_patches, :]
            mask = mask[:, :min_patches]
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**0.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def patchify(self, imgs):
        """
        imgs: (N, seq_len, channels)
        x: (N, num_patches, patch_size * channels)
        """
        p = self.patch_size
        assert imgs.shape[1] % p == 0
        
        h = imgs.shape[1] // p
        x = imgs.reshape(shape=(imgs.shape[0], h, p, imgs.shape[2]))
        x = x.reshape(shape=(imgs.shape[0], h, p * imgs.shape[2]))
        return x
    
    def unpatchify(self, x):
        """
        x: (N, num_patches, patch_size * channels)
        imgs: (N, seq_len, channels)
        """
        p = self.patch_size
        h = x.shape[1]
        assert h * p == self.seq_len
        
        x = x.reshape(shape=(x.shape[0], h, p, -1))
        imgs = x.reshape(shape=(x.shape[0], h * p, x.shape[3]))
        return imgs
    
    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        
        # Make sure mask matches pred size
        if mask.shape[1] != pred.shape[1]:
            mask = mask[:, :pred.shape[1]]
        
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def build_flight_mae_model():
    """Factory function to create FlightDataMAE model"""
    model = FlightDataMAE(
        seq_len=100,
        patch_size=4, 
        in_channels=18,  # Changed from 50 to 18 (matches your data)
        embed_dim=256,
        depth=6,
        num_heads=8,
        decoder_embed_dim=128,
        decoder_depth=4,
        decoder_num_heads=8,
        mask_ratio=0.75
    )
    return model