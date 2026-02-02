"""敵対的摂動ジェネレータネットワーク (UNet ベース)。

NSFW画像を入力として受け取り、知覚的に微小な摂動を加えた画像を出力する。
出力画像は元画像に残差（ノイズ）を加算した形式で、
摂動の大きさは max_perturbation で L-inf 制約される。

アーキテクチャ:
    エンコーダ: 段階的にダウンサンプリング (ストライド2畳み込み)
    ボトルネック: オプションでSelf-Attention付き残差ブロック
    デコーダ: スキップ接続付きアップサンプリング (転置畳み込み)
    出力層: tanh で [-1, 1] に制約 -> max_perturbation でスケール

入出力:
    入力: (N, 3, H, W) float32 [0, 1] NSFW画像
    出力: (N, 3, H, W) float32 [0, 1] 摂動済み画像
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """チャネル方向のSelf-Attentionモジュール。

    特徴マップ内の長距離依存関係を捉えるために使用する。
    ボトルネック部分で空間的に離れた領域間の関係を学習する。
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, c, h, w = x.shape
        # クエリ、キー、バリューを計算
        q = self.query(x).view(batch, -1, h * w).permute(0, 2, 1)
        k = self.key(x).view(batch, -1, h * w)
        v = self.value(x).view(batch, -1, h * w)

        # アテンションマップの計算
        attention = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch, c, h, w)

        # 残差接続（学習初期はγ≈0でスキップ接続に近い）
        return self.gamma * out + x


class ResidualBlock(nn.Module):
    """残差ブロック。

    2層の畳み込み + バッチ正規化 + 残差接続。
    入出力チャネル数が異なる場合は1x1畳み込みでチャネル数を調整する。
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # チャネル数が異なる場合のショートカット
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        return F.leaky_relu(out + residual, 0.2)


class EncoderBlock(nn.Module):
    """UNetエンコーダブロック。

    ストライド2の畳み込みで空間解像度を半分にダウンサンプリングし、
    残差ブロックで特徴抽出を行う。
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.res_block = ResidualBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.bn(self.down(x)), 0.2)
        x = self.res_block(x)
        return x


class DecoderBlock(nn.Module):
    """UNetデコーダブロック。

    転置畳み込みで空間解像度を2倍にアップサンプリングし、
    スキップ接続からの特徴マップと結合した後、残差ブロックで処理する。
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        # スキップ接続結合後のチャネル数 = out_channels + skip_channels
        self.res_block = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.bn(self.up(x)), 0.2)
        # アップサンプル結果とスキップ接続の空間サイズを合わせる
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x)
        return x


class AdversarialGenerator(nn.Module):
    """敵対的摂動ジェネレータ (UNetアーキテクチャ)。

    NSFW画像を入力として、知覚的に微小な摂動を残差として出力する。
    元画像 + 摂動 = 摂動済み画像 として、分類器をSFWと誤判定させる。

    特徴:
        - エンコーダ-デコーダ構造でマルチスケール特徴を利用
        - スキップ接続で入力の細部情報を保持
        - tanh出力 + L-infクリッピングで摂動の大きさを制約
        - オプションのSelf-Attentionで大域的な依存関係をモデル化

    Args:
        in_channels: 入力チャネル数（RGB=3）。
        base_channels: 最初のエンコーダ層のチャネル数。
        depth: エンコーダ/デコーダの段数。
        max_perturbation: 摂動のL-inf最大値 [0, 1]スケール。
        use_attention: ボトルネックにSelf-Attentionを使用するか。
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        depth: int = 4,
        max_perturbation: float = 16 / 255,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.max_perturbation = max_perturbation
        self.depth = depth

        # 入力の初期畳み込み（ダウンサンプリングなし）
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2),
        )

        # エンコーダ（段階的にダウンサンプリング）
        self.encoders = nn.ModuleList()
        ch = base_channels
        encoder_channels = [base_channels]  # スキップ接続用のチャネル数記録
        for i in range(depth):
            out_ch = min(ch * 2, 512)  # 最大512チャネルに制限
            self.encoders.append(EncoderBlock(ch, out_ch))
            ch = out_ch
            encoder_channels.append(ch)

        # ボトルネック
        bottleneck_layers = [ResidualBlock(ch, ch)]
        if use_attention:
            bottleneck_layers.append(SelfAttention(ch))
        bottleneck_layers.append(ResidualBlock(ch, ch))
        self.bottleneck = nn.Sequential(*bottleneck_layers)

        # デコーダ（段階的にアップサンプリング + スキップ接続）
        self.decoders = nn.ModuleList()
        for i in range(depth):
            # スキップ接続のチャネル数（エンコーダの逆順）
            skip_ch = encoder_channels[depth - 1 - i]
            out_ch = skip_ch  # スキップと同じチャネル数に復元
            self.decoders.append(DecoderBlock(ch, skip_ch, out_ch))
            ch = out_ch

        # 出力層: 3チャネルの摂動マップを生成（tanh で [-1, 1] に制約）
        self.output_conv = nn.Sequential(
            nn.Conv2d(ch, ch // 2, 3, padding=1),
            nn.BatchNorm2d(ch // 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ch // 2, in_channels, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播: NSFW画像 -> 摂動δ。

        Args:
            x: (N, 3, H, W) float32 [0, 1] 元のNSFW画像。

        Returns:
            (N, 3, H, W) 摂動δ [-max_perturbation, +max_perturbation]。

        注意:
            元画像との合成 (x + perturbation) は呼び出し側で行う。
            これにより、摂動のみの取得や可視化が容易になる。
        """
        # 初期特徴抽出
        feat = self.input_conv(x)

        # エンコーダ（スキップ接続用に各段の出力を保存）
        skips = [feat]
        for encoder in self.encoders:
            feat = encoder(feat)
            skips.append(feat)

        # ボトルネック
        feat = self.bottleneck(feat)

        # デコーダ（スキップ接続と結合）
        for i, decoder in enumerate(self.decoders):
            # スキップはエンコーダの逆順（最後のエンコーダ出力は使わない）
            skip = skips[self.depth - 1 - i]
            feat = decoder(feat, skip)

        # 摂動マップの生成 [-max_pert, +max_pert]
        # tanh で [-1, 1] に制限し、max_perturbation でスケール
        perturbation = self.output_conv(feat) * self.max_perturbation

        # 摂動δのみを返す（元画像との加算は呼び出し側で行う）
        return perturbation
