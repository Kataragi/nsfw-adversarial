# NSFW Adversarial Noise Generator

pNSFWMedia分類器に対する敵対的ノイズ生成システム。NSFW画像の埋め込みベクトルに摂動を加え、分類器をSFWと誤判定させます。

## 概要

NudeNet YOLOv8バックボーンから抽出された256次元埋め込みベクトルに対して、以下の4つの攻撃手法を実装しています:

| 攻撃手法 | 種別 | 特徴 |
|---------|------|------|
| **FGSM** | 単一ステップ勾配攻撃 | 高速、ε1パラメータで制御 |
| **PGD** | 反復勾配攻撃 | FGSMの反復版、高い成功率 |
| **C&W** | 最適化ベース攻撃 | 最小摂動を探索、高品質 |
| **DeepFool** | 最小摂動攻撃 | 決定境界への最短距離を計算 |

## ターゲット分類器

- **リポジトリ**: [pNSFWMedia](https://github.com/Kataragi/pNSFWMedia)
- **入力**: 256次元埋め込みベクトル
- **構造**: BatchNorm → Dense(256, tanh) → Dense(1, sigmoid)
- **出力**: NSFW確率 (≥0.5でNSFW判定)

## セットアップ

### 必要環境

- Python 3.10+
- TensorFlow 2.16.1+
- CUDA対応GPU (推奨、CPUでも動作可)

### インストール

```bash
git clone https://github.com/Kataragi/nsfw-adversarial.git
cd nsfw-adversarial
pip install -r requirements.txt
```

### ターゲットモデルの配置

pNSFWMediaの学習済みモデルを配置します:

```bash
# 方法1: ファイルを直接コピー
cp /path/to/pNSFWMedia/models/pnsfwmedia_classifier.keras models/target_classifier/

# 方法2: gitサブモジュール
git submodule add https://github.com/Kataragi/pNSFWMedia.git models/pNSFWMedia
ln -s models/pNSFWMedia/models/pnsfwmedia_classifier.keras models/target_classifier/
```

### 埋め込みデータの準備

NudeNetで抽出した埋め込みベクトル(.npy形式)を配置:

```bash
mkdir -p dataset/nsfw_embeddings
# .npy ファイルを dataset/nsfw_embeddings/ に配置
```

## 使い方

### FGSM攻撃

```bash
python src/noise_generator.py \
    --method fgsm \
    --target-model models/target_classifier/pnsfwmedia_classifier.keras \
    --embeddings-dir dataset/nsfw_embeddings \
    --epsilon 0.05 \
    --output-dir experiments/attack_results/fgsm_eps0.05
```

### PGD攻撃

```bash
python src/noise_generator.py \
    --method pgd \
    --target-model models/target_classifier/pnsfwmedia_classifier.keras \
    --embeddings-dir dataset/nsfw_embeddings \
    --epsilon 0.05 \
    --alpha 0.01 \
    --iterations 20 \
    --output-dir experiments/attack_results/pgd_eps0.05
```

### C&W攻撃

```bash
python src/noise_generator.py \
    --method cw \
    --target-model models/target_classifier/pnsfwmedia_classifier.keras \
    --embeddings-dir dataset/nsfw_embeddings \
    --cw-c 1.0 \
    --cw-kappa 0.0 \
    --iterations 500 \
    --output-dir experiments/attack_results/cw_c1.0
```

### DeepFool攻撃

```bash
python src/noise_generator.py \
    --method deepfool \
    --target-model models/target_classifier/pnsfwmedia_classifier.keras \
    --embeddings-dir dataset/nsfw_embeddings \
    --max-iterations 100 \
    --overshoot 0.02 \
    --output-dir experiments/attack_results/deepfool
```

### 一括ロバスト性評価

複数手法・パラメータで一括評価し、可視化を生成:

```bash
python src/evaluate_robustness.py \
    --target-model models/target_classifier/pnsfwmedia_classifier.keras \
    --embeddings-dir dataset/nsfw_embeddings \
    --methods fgsm pgd cw deepfool \
    --epsilon-range 0.01 0.05 0.1 \
    --output-dir experiments/robustness_analysis
```

### TensorBoardによるモニタリング

攻撃の進行状況をTensorBoardで確認できます:

```bash
# TensorBoardを有効にして攻撃を実行
python src/noise_generator.py \
    --method pgd \
    --target-model models/target_classifier/pnsfwmedia_classifier.keras \
    --embeddings-dir dataset/nsfw_embeddings \
    --tensorboard \
    --tb-log-dir logs

# 別ターミナルでTensorBoardを起動
tensorboard --logdir logs
```

ブラウザで http://localhost:6006 を開くと以下を確認できます:
- 各イテレーションでの平均NSFW確率
- 損失関数の推移
- 攻撃成功率の推移

## 出力形式

各攻撃の結果はJSON形式で保存されます:

```json
{
    "attack_method": "PGD",
    "parameters": {"epsilon": 0.05, "alpha": 0.01, "iterations": 20},
    "results": {
        "total_samples": 1000,
        "nsfw_samples": 800,
        "success_rate_0.5": 0.92,
        "success_rate_0.4": 0.87,
        "success_rate_0.3": 0.75,
        "avg_prob_reduction": 0.45,
        "avg_l2_norm": 0.023,
        "avg_linf_norm": 0.048,
        "avg_iterations": 18.3
    }
}
```

### 評価メトリクス

| メトリクス | 説明 |
|-----------|------|
| `success_rate_0.5` | NSFW確率が0.5未満に低下した割合 |
| `success_rate_0.4` | NSFW確率が0.4未満に低下した割合 |
| `success_rate_0.3` | NSFW確率が0.3未満に低下した割合 |
| `avg_prob_reduction` | 攻撃前後の平均確率低下量 |
| `avg_l2_norm` | 摂動のL2ノルム平均 |
| `avg_linf_norm` | 摂動のL∞ノルム平均 |
| `avg_iterations` | 反復攻撃の平均収束回数 |

### 生成される可視化

`evaluate_robustness.py`は以下の可視化を生成します:

- **epsilon_vs_success_rate.png** - ε値と攻撃成功率の関係
- **probability_distributions.png** - 攻撃前後のNSFW確率分布
- **noise_distributions.png** - 摂動ノルムの分布
- **comparison_table.png** - 全手法の比較サマリー

## プロジェクト構成

```
nsfw-adversarial/
├── README.md                     # このファイル
├── requirements.txt              # 依存パッケージ
├── .gitignore                    # Git除外設定
├── config/
│   └── attack_config.yaml       # 攻撃パラメータ設定
├── src/
│   ├── __init__.py
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── fgsm.py              # FGSM攻撃
│   │   ├── pgd.py               # PGD攻撃
│   │   ├── cw.py                # C&W攻撃
│   │   └── deepfool.py          # DeepFool攻撃
│   ├── noise_generator.py       # CLI メインスクリプト
│   ├── embedding_attacker.py    # 攻撃オーケストレーション
│   ├── evaluate_robustness.py   # ロバスト性評価・可視化
│   └── utils.py                 # ユーティリティ関数
├── models/
│   └── target_classifier/       # ターゲットモデル配置先
├── experiments/
│   ├── attack_results/          # 攻撃結果出力
│   └── visualizations/          # 可視化出力
└── notebooks/
    └── attack_demo.ipynb        # デモノートブック
```

## 設定ファイル

`config/attack_config.yaml`でデフォルトパラメータを管理:

```yaml
attacks:
  fgsm:
    epsilon: [0.01, 0.03, 0.05, 0.07, 0.1]
  pgd:
    epsilon: [0.01, 0.03, 0.05, 0.07, 0.1]
    alpha: [0.001, 0.005, 0.01]
    iterations: [10, 20, 40]
```

## パラメータチューニングガイド

高い攻撃成功率(95%以上)を得るための推奨設定:

| 手法 | 推奨パラメータ |
|------|--------------|
| FGSM | ε = 0.07〜0.1 |
| PGD  | ε = 0.05, α = ε/4, iterations = 20〜40 |
| C&W  | c = 1.0〜10.0, iterations = 500+ |
| DeepFool | overshoot = 0.02〜0.05 |

## ライセンス

Research use only.
