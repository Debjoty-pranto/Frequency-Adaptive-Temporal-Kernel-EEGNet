import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyAdaptiveTemporalKernel(nn.Module):
    """
    Vectorized temporal conv + small adaptive gating based on band powers.
    Stable: base_conv is shared; per-trial modulation is small.
    Input: x (B, C, T)
    Output: (B, out_channels, C, T)
    """
    def __init__(self,
                 in_channels=1,
                 out_channels=24,
                 kernel_size=128,
                 sample_rate=512,
                 freq_bands=[(8,12),(13,30)],  # tweak bands to match your preprocessing
                 dropout=0.2):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.freq_bands = freq_bands
        self.num_bands = len(freq_bands)

        # base conv applied per channel (we will reshape B*C -> conv)
        self.base_conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                   padding=kernel_size//2, bias=False)

        # small MLP turning band-powers (B, num_bands) -> (B, out_channels) modulation in (0,1)
        hidden = max(32, self.num_bands * 8)
        self.mod_mlp = nn.Sequential(
            nn.Linear(self.num_bands, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_channels),
            nn.Sigmoid()
        )

        self.bn2d = nn.BatchNorm2d(out_channels)

        # init
        nn.init.xavier_uniform_(self.base_conv.weight)

    def compute_band_powers(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        # rFFT across time
        fft = torch.fft.rfft(x, dim=-1)
        power = (fft.abs() ** 2)  # (B, C, Tfreq)
        freqs = torch.fft.rfftfreq(n=T, d=1.0 / self.sample_rate).to(x.device)  # (Tfreq,)

        band_pows = []
        for (low, high) in self.freq_bands:
            mask = (freqs >= low) & (freqs < high)
            if mask.any():
                bp = power[..., mask].mean(dim=(-1, -2))  # mean over freqs and channels -> (B,)
            else:
                bp = torch.zeros(B, device=x.device)
            band_pows.append(bp.unsqueeze(1))
        return torch.cat(band_pows, dim=1)  # (B, num_bands)

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape

        # 1) band powers per trial
        band_pows = self.compute_band_powers(x)  # (B, num_bands)

        # 2) modulation per trial -> (B, out_channels); values in (0,1)
        modulation = self.mod_mlp(band_pows)

        # 3) apply base_conv per channel by reshaping
        x_reshaped = x.view(B * C, 1, T)      # (B*C, 1, T)
        conv_out = self.base_conv(x_reshaped) # (B*C, out_channels, T)
        conv_out = conv_out.view(B, C, self.out_channels, -1)  # (B, C, out, T)
        conv_out = conv_out.permute(0, 2, 1, 3).contiguous()   # (B, out, C, T)

        # 4) gentle gating: convert modulation to scale ~[0.8,1.2] (tunable)
        scale = 0.8 + modulation.unsqueeze(-1).unsqueeze(-1) * 0.4  # (B, out, 1, 1)
        out = conv_out * scale  # broadcast over C,T

        out = self.bn2d(out)
        return out


class AdaptiveEEGNet(nn.Module):
    """
    Simplified AdaptiveEEGNet tuned for stability / learning.
    Expects either:
      - x: [B, C, T]  (Braindecode windows)
      - x: [B, 1, C, T]
    """
    def __init__(self,
                 nb_classes=2,
                 Chans=12,
                 Samples=1024,
                 kernLength=128,
                 F1=24,
                 D=2,
                 F2=48,
                 dropoutRate=0.25,
                 sample_rate=512,
                 freq_bands=[(8,12),(13,30)]):
        super().__init__()
        self.Chans = Chans
        self.Samples = Samples

        self.freq_adaptive = FrequencyAdaptiveTemporalKernel(
            in_channels=1,
            out_channels=F1,
            kernel_size=kernLength,
            sample_rate=sample_rate,
            freq_bands=freq_bands,
            dropout=0.2
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise spatial conv (grouped conv): spatial mixing across channels
        self.depthwise = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.bn_dw = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropoutRate)

        # Separable (pointwise) conv
        self.sep = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn_sep = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropoutRate)

        self.flatten = nn.Flatten()

        # compute flattened feature dim with dummy
        with torch.no_grad():
            dummy = torch.randn(1, 1, Chans, Samples)
            n_feat = self._forward_features(dummy).shape[1]

        # simple linear classifier (no big MLP)
        self.classifier = nn.Linear(n_feat, nb_classes)

    def _forward_features(self, x):
        # x: [B, 1, Chans, Samples]
        B, _, Chans, Samples = x.shape
        x1d = x.view(B, Chans, Samples)  # (B, C, T)

        out = self.freq_adaptive(x1d)    # (B, F1, C, T)
        out = self.bn1(out)

        out = self.depthwise(out)        # (B, F1*D, 1, T')
        out = self.bn_dw(out)
        out = self.elu(out)
        out = self.pool1(out)
        out = self.drop1(out)

        out = self.sep(out)              # (B, F2, 1, T'')
        out = self.bn_sep(out)
        out = self.elu(out)
        out = self.pool2(out)
        out = self.drop2(out)

        return self.flatten(out)

    def forward(self, x):
        # accept [B, C, T] or [B,1,C,T]
        if x.ndim == 3:
            x = x.unsqueeze(1)
        feats = self._forward_features(x)
        return self.classifier(feats)
