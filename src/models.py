"""The four architectures plus a linear reference, all with one interface.

EN: Every model here takes (batch, lookback, n_features) and returns
    (batch, 1): a single standardised temperature for t + horizon. Keeping the
    interface identical is what lets the training loop and the evaluation code
    stay completely model-agnostic.
TR: Buradaki her model (batch, lookback, n_features) alıp (batch, 1) döndürüyor:
    t + horizon anı için tek bir standartlaştırılmış sıcaklık. Arayüzü birebir
    aynı tutmam, eğitim döngüsünün ve değerlendirme kodunun modelden tamamen
    bağımsız kalmasını sağlıyor.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# 0. Linear reference / Doğrusal referans
# --------------------------------------------------------------------------- #
class LinearFlatten(nn.Module):
    """Flatten the window and fit one linear layer.

    EN: This is the "is a deep model even necessary?" control. It has no notion
        of time order at all, yet on this dataset it is a surprisingly strong
        competitor. If an LSTM only ties with this, the LSTM is not earning its
        complexity.
    TR: Bu, "derin modele gerçekten gerek var mı?" kontrolü. Zaman sırası diye
        bir kavramı yok ama bu veri setinde şaşırtıcı derecede güçlü bir rakip.
        Bir LSTM yalnızca bununla başa baş gidiyorsa, karmaşıklığını hak
        etmiyordur.
    """

    def __init__(self, n_features: int, lookback: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features * lookback, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# 1. LSTM
# --------------------------------------------------------------------------- #
class LSTMForecaster(nn.Module):
    """Single/multi-layer LSTM, prediction read from the final hidden state.

    EN: I take the last layer's final hidden state rather than pooling over
        time, because for a next-value forecast the most recent context is what
        matters and the gates have already carried the older information
        forward.
    TR: Zaman üzerinden havuzlama yapmak yerine son katmanın nihai gizli
        durumunu alıyorum; çünkü bir sonraki değeri tahmin ederken en güncel
        bağlam belirleyicidir ve kapılar eski bilgiyi zaten ileri taşımıştır.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.rnn(x)
        return self.head(h_n[-1])


# --------------------------------------------------------------------------- #
# 2. GRU
# --------------------------------------------------------------------------- #
class GRUForecaster(nn.Module):
    """Same idea as the LSTM with one gate fewer and ~25% fewer parameters."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.rnn(x)
        return self.head(h_n[-1])


# --------------------------------------------------------------------------- #
# 3. 1D CNN
# --------------------------------------------------------------------------- #
class CNN1DForecaster(nn.Module):
    """Dilated causal convolutions over the time axis.

    EN: Dilations of 1-2-4-8 give the stack a receptive field of ~60 hours with
        only four layers, and convolutions parallelise across time, so this
        trains far faster than the RNNs. I use causal padding (left-side only)
        so no filter can peek at a timestep that comes after the one it is
        producing.
    TR: 1-2-4-8 genişlemeleri, yalnızca dört katmanla yığına ~60 saatlik bir
        algı alanı kazandırıyor; ayrıca konvolüsyonlar zaman ekseninde
        paralelleşiyor, dolayısıyla bu model RNN'lerden çok daha hızlı eğitiliyor.
        Nedensel dolgu (yalnızca sol taraf) kullanıyorum ki hiçbir filtre,
        ürettiği adımdan sonraki bir adıma bakamasın.
    """

    def __init__(
        self,
        n_features: int,
        channels: int = 64,
        n_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = n_features
        for i in range(n_blocks):
            dilation = 2**i
            layers += [
                _CausalConv1d(in_ch, channels, kernel_size, dilation),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = channels
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, F) -> (B, F, L) for Conv1d
        h = self.conv(x.transpose(1, 2))
        # EN: take the last timestep: it has seen the whole window.
        # TR: son zaman adımını al: tüm pencereyi görmüş olan adım odur.
        return self.head(h[:, :, -1])


class _CausalConv1d(nn.Module):
    """Conv1d that pads only on the left, so the output stays causal."""

    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int, dilation: int
    ) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


# --------------------------------------------------------------------------- #
# 4. Transformer
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    """Classic sinusoidal position encoding.

    EN: Self-attention is permutation invariant, so without this the model would
        see the 120-hour window as an unordered bag of readings. For a weather
        forecast that would throw away exactly the information I care about.
    TR: Self-attention permütasyona duyarsızdır; bu olmadan model 120 saatlik
        pencereyi sırasız bir ölçüm torbası olarak görürdü. Hava tahmini için bu,
        tam da önemsediğim bilgiyi çöpe atmak olurdu.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    """Encoder-only transformer with mean pooling over time.

    EN: I keep d_model deliberately small (64). With ~35k training windows and
        14 input features, a big transformer would memorise the training years
        long before it learned anything transferable.
    TR: d_model'i bilerek küçük tutuyorum (64). ~35 bin eğitim penceresi ve 14
        girdi değişkeniyle, büyük bir transformer aktarılabilir bir şey öğrenmeden
        çok önce eğitim yıllarını ezberlerdi.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos_encoding(self.input_proj(x))
        h = self.encoder(h)
        return self.head(h.mean(dim=1))


# --------------------------------------------------------------------------- #
# Registry / Kayıt defteri
# --------------------------------------------------------------------------- #
MODEL_REGISTRY = {
    "Linear": LinearFlatten,
    "LSTM": LSTMForecaster,
    "GRU": GRUForecaster,
    "CNN1D": CNN1DForecaster,
    "Transformer": TransformerForecaster,
}


def build_model(name: str, n_features: int, lookback: int, **kwargs) -> nn.Module:
    """Create a model by name with the arguments it actually accepts."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Known: {list(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[name]
    if cls is LinearFlatten:
        return cls(n_features=n_features, lookback=lookback, **kwargs)
    return cls(n_features=n_features, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
