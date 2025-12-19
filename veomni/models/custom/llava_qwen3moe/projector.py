import torch
import torch.nn as nn
import numpy as np
import math
from functools import lru_cache
    
@lru_cache(maxsize=100)
def get_adaptive_pool_size(M, N, scale=20):
    r = 1 / math.sqrt(scale)
    Mh = max(1, int(np.round(M * r)))
    Nw = max(1, int(np.round(N * r)))
    return Mh, Nw


class AudioUpScaleProjector(nn.Module):
    def __init__(self, encoder_hidden, out_hidden, downsample_ratio):
        super().__init__()

        self.hidden_size = out_hidden
        self.audio_hidden_size = encoder_hidden
        self.audio_downsample_ratio = downsample_ratio
        self.linear_compress_ratio = downsample_ratio // 2 # conv已经压缩了2倍
        self.afeat_1d_conv = nn.Conv1d(in_channels=encoder_hidden, out_channels=encoder_hidden, kernel_size=2, stride=2, padding=0) # 50Hz -> 25Hz
        self.linear1 = nn.Linear(int(self.audio_hidden_size * self.linear_compress_ratio), self.hidden_size, bias=True)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def forward(self, x, feature_length):

        x = self.afeat_1d_conv(x.transpose(1, 2)).transpose(1, 2) # Process Whisper features with 1D conv: (B x T x D) -> (B x T//2 x D')
        bs, seq_len, audio_hidden_size = x.size()
        # 计算目标长度（向上取整到 compress_ratio 的整数倍）
        target_seq_len = math.ceil((seq_len + self.linear_compress_ratio - 1) / self.linear_compress_ratio) * self.linear_compress_ratio
        pad_len = target_seq_len - seq_len

        if pad_len > 0:
            pad_tensor = torch.zeros(bs, pad_len, audio_hidden_size, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad_tensor], dim=1)  # 在时间维度 padding
        new_seq_len = target_seq_len // self.linear_compress_ratio
        x = x.reshape(bs, new_seq_len, audio_hidden_size * self.linear_compress_ratio)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        num_tokens = [(l + self.audio_downsample_ratio - 1)// self.audio_downsample_ratio for l in feature_length]
        
        valid_outputs = []
        for i in range(bs):
            valid_segment = x[i, :num_tokens[i], :]
            valid_outputs.append(valid_segment)
        
        x = torch.cat(valid_outputs, dim=0)
        return x, num_tokens

class DynamicAvgPoolProjector(nn.Module):
    def __init__(self, encoder_hidden, out_hidden, downsample_ratio):
        super().__init__()
        self.mm_downsample_ratio = downsample_ratio
        self.hidden_size = encoder_hidden
        self.merge_size = 2
        in_hidden = self.hidden_size
    
        self.mlp = nn.Sequential(
            nn.Linear(in_hidden, in_hidden),
            nn.GELU(),
            nn.Linear(in_hidden, out_hidden),
        )

    def forward(self, images_feature, images_thw, merge_size=None):
        """
        Args:
            images: Tensor of shape [N, hidden_size]
            images_thw: Tensor of shape [m, 3], each row is [t, h, w]
                       There are m sequences; each sequence corresponds to t*h*w vectors in `images`.
        Returns:
            Tensor of shape [m, out_hidden]
        """
        outputs = []
        start = 0
        seq_len = []
        
        if merge_size is None:
            merge_size = torch.tensor([self.merge_size]*images_thw.shape[0])
        
       
        for thw, each_merge_size in zip(images_thw, merge_size):
            t, h, w = thw.cpu().tolist()
            h, w = int(h / each_merge_size.item()), int(w / each_merge_size.item())
            length = int(t * h * w)
            img_seq = images_feature[start:start + length]  # [t*h*w, hidden]
            start += length

            # reshape to [t, h, w, hidden] -> permute to [hidden, t, h, w]
            img_feat = img_seq.view(t, h, w, -1).permute(3, 0, 1, 2)  # [hidden, t, h, w]

            Mh, Nw = get_adaptive_pool_size(h, w, scale=self.mm_downsample_ratio)
            pool = nn.AdaptiveAvgPool2d((Mh, Nw))  # pool on H, W

            pooled = pool(img_feat)  # [hidden, t, Mh, Nw]
            pooled = pooled.permute(1, 2, 3, 0).contiguous()  # [t, Mh, Nw, hidden]
          
            pooled = pooled.view(-1, self.hidden_size)  # flatten: [t * Mh * Nw, hidden]
            
            tokens = [pooled.shape[0]//t]*t # calculate each image
            outputs.append(pooled)
            seq_len.extend(tokens)
        
        outputs = torch.cat(outputs, dim=0)  # [m, hidden_size]
        projected = self.mlp(outputs)  # [m, out_hidden]
        
        return projected, seq_len

        


def build_image_projector(projector_type, encoder_hidden, out_hidden, downsample_ratio):
    print("image encoder", encoder_hidden, out_hidden, downsample_ratio)
    if projector_type== "dynamic_avgpool":
        return DynamicAvgPoolProjector(encoder_hidden, out_hidden, downsample_ratio)
    else:
        raise NotImplementedError


def build_audio_projector(projector_type, encoder_hidden, out_hidden, downsample_ratio):
    print("audio encoder", encoder_hidden, out_hidden, downsample_ratio)
    if projector_type == "channel_upscale":
        return AudioUpScaleProjector(encoder_hidden, out_hidden, downsample_ratio)

    else:
        raise NotImplementedError





