# NSFW Adversarial Noise Generator

pNSFWMedia分類器に対する**画像レベル**の敵対的ノイズ生成システム。NSFW画像に知覚困難な摂動を加え、分類器をSFWと誤判定させます。

## 概要

NudeNet YOLOv8バックボーンからpNSFWMedia分類器までの**エンドツーエンドパイプライン**を微分可能なTensorFlowモデルとして構築し、画像ピクセルに対する勾配を計算して敵対的ノイズを生成します。

### パイプライン

```
画像 (320x320x3, [0,1])
  -> NudeNet backbone (ONNX->TF変換)
  -> Global Average Pooling
  -> 直交射影 (C -> 256次元)
  -> L2正規化
  -> pNSFWMedia分類器
  -> NSFW確率 (0.0~1.0)
```

### 攻撃手法

| 攻撃手法 | 種別 | 特徴 |
|---------|------|------|
| **FGSM** | 単一ステップ勾配攻撃 | 高速、ε1パラメータで制御 |
| **PGD** | 反復勾配攻撃 | FGSMの反復版、高い成功率 |
| **C&W** | 最適化ベース攻撃 | 最小L2摂動を探索 |
| **DeepFool** | 最小摂動攻撃 | 決定境界への最短距離を計算 |

## セットアップ

### 必要環境

- WSL2 (Ubuntu 22.04+)
- Python 3.10+
- NVIDIA GPU + CUDA Toolkit 12.4+
- TensorFlow 2.16.1+ / PyTorch 2.2+
- ONNX Runtime GPU

### WSL2 前提のシステム依存パッケージ

```bash
# WSL2 Ubuntu で必要な最小限のパッケージ
sudo apt update && sudo apt install -y \
    python3-dev python3-pip python3-venv \
    libglib2.0-0 libsm6 libxrender1 libxext6

# NVIDIA ドライバはWindows側にインストール済みであること
# WSL2 内では CUDA Toolkit のみ必要
# https://developer.nvidia.com/cuda-downloads (WSL-Ubuntu を選択)
nvidia-smi  # GPU が認識されていることを確認
```

### インストール

```bash
git clone https://github.com/Kataragi/nsfw-adversarial.git
cd nsfw-adversarial
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> **注意**: `requirements.txt` は `--extra-index-url` で PyTorch CUDA 12.4 wheels を参照しています。
> 異なる CUDA バージョンを使用する場合は `cu124` 部分を `cu121` 等に書き換えてください。

### 必要ファイルの配置

#### 1. pNSFWMedia分類器

```bash
# 方法1: 直接コピー
cp /path/to/pNSFWMedia/models/pnsfwmedia_classifier.keras models/target_classifier/

# 方法2: gitサブモジュール
git submodule add https://github.com/Kataragi/pNSFWMedia.git models/pNSFWMedia
cp models/pNSFWMedia/models/pnsfwmedia_classifier.keras models/target_classifier/
```

#### 2. 射影行列

pNSFWMediaの埋め込み抽出時に生成される射影行列を配置:

```bash
cp /path/to/pNSFWMedia/models/nudenet_projection.npy models/
```

#### 3. NudeNet ONNXモデル

`pip install nudenet` でインストールすると自動的に `best.onnx` が配置されます。
初回実行時にONNXバックボーンがTensorFlow形式に自動変換され、`models/tf_backbone/` にキャッシュされます。

#### 4. 攻撃対象画像

```bash
mkdir -p dataset/nsfw_images
# NSFW画像を配置 (.jpg, .png, .webp 等)
```

## 使い方

### FGSM攻撃

```bash
python src/noise_generator.py \
    --method fgsm \
    --images-dir dataset/nsfw_images \
    --epsilon 0.031 \
    --output-dir experiments/attack_results/fgsm_8px
```

### PGD攻撃

```bash
python src/noise_generator.py \
    --method pgd \
    --images-dir dataset/nsfw_images \
    --epsilon 0.031 \
    --alpha 0.008 \
    --iterations 20 \
    --output-dir experiments/attack_results/pgd_8px
```

### C&W攻撃

```bash
python src/noise_generator.py \
    --method cw \
    --images-dir dataset/nsfw_images \
    --cw-c 1.0 \
    --cw-kappa 0.0 \
    --iterations 500 \
    --output-dir experiments/attack_results/cw_c1.0
```

### DeepFool攻撃

```bash
python src/noise_generator.py \
    --method deepfool \
    --images-dir dataset/nsfw_images \
    --max-iterations 100 \
    --overshoot 0.02 \
    --output-dir experiments/attack_results/deepfool
```

### 一括ロバスト性評価

```bash
python src/evaluate_robustness.py \
    --images-dir dataset/nsfw_images \
    --methods fgsm pgd cw deepfool \
    --epsilon-range 0.016 0.031 0.063 0.125 \
    --output-dir experiments/robustness_analysis
```

### TensorBoardによるモニタリング

```bash
# TensorBoardを有効にして攻撃実行
python src/noise_generator.py \
    --method pgd \
    --images-dir dataset/nsfw_images \
    --tensorboard --tb-log-dir logs

# 別ターミナルでTensorBoard起動
tensorboard --logdir logs
```

ブラウザで http://localhost:6006 を開くと以下を確認できます:
- 各イテレーションでの平均NSFW確率
- 損失関数の推移
- 攻撃成功率の推移

### パイプラインオプション

すべてのコマンドで以下のオプションが使用可能:

```bash
--classifier-model PATH   # 分類器モデルパス
--projection-path PATH    # 射影行列パス
--nudenet-onnx PATH       # NudeNet ONNXパス (未指定時は自動検出)
--backbone-cache DIR      # TFバックボーンのキャッシュディレクトリ
--max-images N            # 処理する画像数の上限
```

## 出力形式

### JSON結果

```json
{
    "attack_method": "PGD",
    "parameters": {"epsilon": 0.031, "alpha": 0.008, "iterations": 20},
    "results": {
        "total_samples": 100,
        "nsfw_samples": 95,
        "success_rate_0.5": 0.92,
        "success_rate_0.4": 0.87,
        "success_rate_0.3": 0.75,
        "avg_prob_reduction": 0.45,
        "avg_l2_norm": 2.34,
        "avg_linf_norm": 0.031,
        "avg_linf_norm_pixel": 8.0,
        "avg_iterations": 18.3
    }
}
```

### 敵対的画像

攻撃成功した画像は `output_dir/images/` にPNG形式で保存されます。

### 評価メトリクス

| メトリクス | 説明 |
|-----------|------|
| `success_rate_0.5` | NSFW確率が0.5未満に低下した割合 |
| `success_rate_0.4` | NSFW確率が0.4未満に低下した割合 |
| `success_rate_0.3` | NSFW確率が0.3未満に低下した割合 |
| `avg_prob_reduction` | 攻撃前後の平均確率低下量 |
| `avg_l2_norm` | 摂動のL2ノルム平均 |
| `avg_linf_norm` | 摂動のL-inf ノルム平均 ([0,1]スケール) |
| `avg_linf_norm_pixel` | 摂動のL-inf ノルム平均 ([0,255]スケール) |
| `avg_iterations` | 反復攻撃の平均収束回数 |

## ジェネレータベース敵対的摂動 (PyTorch)

画像ごとの最適化（FGSM/PGD等）に加え、PyTorchベースのUNetジェネレータを学習させることで、**1回の順伝播で**NSFW画像に敵対的摂動を付与できます。

### アーキテクチャ

```
NSFW画像 (N, 3, 320, 320) [0, 1]
  -> ジェネレータ G (UNet)
  -> 摂動マップ δ (tanh -> max_perturbation でスケール)
  -> 摂動済み画像 x' = clamp(x + δ, 0, 1)
  -> 凍結済み分類器 f(x') -> NSFW確率
  -> 損失: L = λ_cls * BCE(f(x'), 0) + λ_l2 * ||x' - x||_2
```

### ジェネレータ学習

```bash
# 基本的な学習
python -m src.generator.train \
    --config config/generator_config.yaml \
    --train-dir dataset/nsfw_images/train \
    --val-dir dataset/nsfw_images/val

# パラメータをカスタマイズ
python -m src.generator.train \
    --config config/generator_config.yaml \
    --train-dir dataset/nsfw_images/train \
    --val-dir dataset/nsfw_images/val \
    --epochs 100 \
    --batch-size 8 \
    --lr 0.0002 \
    --lambda-cls 1.0 \
    --lambda-l2 10.0 \
    --max-perturbation 0.0627

# チェックポイントから学習を再開
python -m src.generator.train \
    --config config/generator_config.yaml \
    --train-dir dataset/nsfw_images/train \
    --resume checkpoints/generator/generator_epoch0050.pt
```

### ジェネレータ学習のTensorBoardモニタリング

```bash
tensorboard --logdir logs/generator
```

以下の指標を確認できます:
- エポック/バッチ単位の損失推移（分類損失、L2ペナルティ）
- 学習率の推移
- 検証セットでの攻撃成功率 (SR@0.5, SR@0.4, SR@0.3)
- サンプル画像（元画像・摂動済み画像・摂動マップ）

### ジェネレータの設定パラメータ

| パラメータ | 説明 | 推奨値 |
|-----------|------|--------|
| `max_perturbation` | 摂動のL-inf上限 [0,1]スケール | 8~16/255 |
| `lambda_cls` | 分類損失の重み | 1.0 |
| `lambda_l2` | L2摂動ペナルティの重み | 5.0~20.0 |
| `base_channels` | UNetの基本チャネル数 | 32 |
| `depth` | UNetのエンコーダ段数 | 4 |
| `learning_rate` | Adam学習率 | 1e-4~5e-4 |
| `scheduler` | 学習率スケジューラ | cosine (warmup=5) |

## プロジェクト構成

```
nsfw-adversarial/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   ├── attack_config.yaml       # 攻撃パラメータ設定
│   └── generator_config.yaml    # ジェネレータ学習設定
├── src/
│   ├── __init__.py
│   ├── pipeline.py              # エンドツーエンドパイプライン (ONNX->TF変換)
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── fgsm.py              # FGSM攻撃 (画像)
│   │   ├── pgd.py               # PGD攻撃 (画像)
│   │   ├── cw.py                # C&W攻撃 (画像)
│   │   └── deepfool.py          # DeepFool攻撃 (画像)
│   ├── generator/
│   │   ├── __init__.py
│   │   ├── model.py             # UNetジェネレータ (PyTorch)
│   │   ├── classifier_wrapper.py # 分類器のPyTorchラッパー
│   │   ├── dataset.py           # NSFW画像データセット
│   │   └── train.py             # ジェネレータ学習スクリプト
│   ├── image_attacker.py        # 攻撃オーケストレーション
│   ├── noise_generator.py       # CLI メインスクリプト
│   ├── evaluate_robustness.py   # ロバスト性評価・可視化
│   └── utils.py                 # ユーティリティ (画像I/O、メトリクス)
├── models/
│   ├── target_classifier/       # pNSFWMedia分類器
│   ├── tf_backbone/             # NudeNetバックボーン TF版 (自動生成)
│   └── torch_backbone/          # NudeNetバックボーン PyTorch版 (自動生成)
├── checkpoints/
│   └── generator/               # ジェネレータのチェックポイント
├── experiments/
│   ├── attack_results/
│   └── visualizations/
└── notebooks/
    └── attack_demo.ipynb
```

## パラメータチューニングガイド

εの単位は[0,1]ピクセルスケール。x/255で指定:

| 手法 | 推奨パラメータ |
|------|--------------|
| FGSM | ε = 8~16/255 |
| PGD  | ε = 8/255, α = ε/4, iterations = 20~40 |
| C&W  | c = 1.0~10.0, iterations = 500+ |
| DeepFool | overshoot = 0.02~0.05 |

## 初回実行時の動作

1. NudeNet ONNXモデルを自動検出
2. バックボーン部分をONNXから抽出
3. `onnx2tf` でTensorFlow SavedModelに変換
4. `models/tf_backbone/` にキャッシュ (2回目以降は高速起動)
5. パイプライン全体をTFモデルとして構築
6. 指定された攻撃を実行

## ライセンス

Research use only.
