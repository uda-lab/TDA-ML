# 再現性ガイド

この文書は **読者向け** です。実験の再現手順、データと成果物の置き場所、図がコードのどこに対応するかをまとめています。内部手順・監査メモなど（従来 `docs/` に置いていたもの）は **この公開リポジトリの git ツリーには含めません**（ローカルや別管理で保持してください）。

## リポジトリに含まれる範囲（目安）

- **含む:** `tda_ml/`、**`configs/` 直下の正本 YAML**（`base.yaml` と `reproduce` / `dev` / `prod` / `test_fast` の各ファイル）、`tests/`、追跡されている `scripts/`、`experiments/run_backend_multiseed.py`、および `README.md` / `REPRODUCIBILITY.md` / `pyproject.toml` / `uv.lock` / `LICENSE` / `CITATION.cff` などのメタデータ。
- **含めない:** `docs/` 以下、`configs/archive/`（履歴用 YAML を置く場合は **ローカルのみ**）、`outputs/`、`data/` など。`load_config("archive/...")` は、手元に `configs/archive/*.yaml` を置いた場合にのみ使えます。

## 環境

- **Python**: `.python-version` を参照（現状 3.12）。
- **依存関係**: **uv** で管理。リポジトリのルートで:

  ```bash
  git submodule update --init --recursive
  test -f pytorch-topological/pyproject.toml || \
    git clone --depth 1 https://github.com/aidos-lab/pytorch-topological.git pytorch-topological
  uv sync --all-groups   # ローカル検証用に dev（pytest, ruff）を含める
  ```

  `torch_topological` は **`pytorch-topological/` の path 依存**であり、このリポジトリの既定クローンには含まれない。上記 `git clone` は CI と同じ不足時の取得手順である。

- **PyTorch / CUDA**: 数値結果はデバイスや dtype によって変わり得ます。`tda_ml/main.py` の学習ループを使う場合、有効な設定は各実行の `logs/` 配下の `runtime_profile.json` などに記録されます。

## データ（MNIST）

- MNIST は git にコミットしません（`data/` は無視対象）。
- 初回の学習またはデータセットアクセス時に、`torchvision` 経由で **`./data`** 以下にダウンロードされます（`configs/reproduce.yaml` を前提とした設定が典型です）。
- 初回はインターネットに到達できるようにするか、キャッシュ済みの MNIST を自分で `./data` に置いてください。
- 設定 YAML の役割分担は **`configs/README.md`**（正本 5 本）を参照。旧設定はローカルで `configs/archive/` に置けるが、公開クローンには同梱されない。

## チェックポイントと実行出力

- `experiments/run_backend_multiseed.py` が内部で読み込む **`tda_ml/main.py`** の学習処理により、実行ごとのディレクトリ以下に成果物が書き出されます（README の「Expected artifacts」参照）。典型例は `logs/metrics.csv`、`logs/runtime_profile.json`、`best_model.pth`、可視化が有効なら `images/` などです。
- **`outputs/`** は git の対象外です。論文用に実行ツリーを保存する場合は、原稿や付録で **コミットハッシュ・シード・使用した設定名** とあわせてパスを示すと追跡しやすいです。`progress_summary.csv` には絶対パスが入るため、共有時のプライバシーに注意してください。

## 数値再現の公式手順

**公式**のマルチシード・バックエンド比較は次のとおりです。

```bash
uv run python experiments/run_backend_multiseed.py \
  --base-config reproduce \
  --epochs 50 \
  --seeds 42 123 456 789 1024 \
  --backends mahalanobis ellphi \
  --out-base outputs/backend_compare
```

再開の挙動、ロックファイル、CSV の意味は `README.md` に書いてあります。

### バックエンド比較と outlier 確率の重み（非対称）

位相損失用の距離行列は `model.topology_loss.distance_backend` ごとに別定義です。**`mahalanobis`** では、学習で予測した **outlier 確率 `probs`** を距離の重み付けに織り込めます（`tda_ml.topology.compute_anisotropic_distance_matrix`）。**`ellphi`** では楕円の接触距離のみを用い、**確率に基づく重み付けは未実装のため `probs` は使われません**（初回のみ `UserWarning` が出ます。実装は `tda_ml.distance_backend.compute_distance_matrix_batch`）。

したがって、`run_backend_multiseed.py` で同じ YAML を回しても、**位相損失が見ている距離空間はバックエンド間で同一ではありません**。ここでは「同一のデータ・スケジュール・設定表面での再現パイプライン比較」を意図しており、**両バックエンドが数学的に完全に同型の重み付き距離目的関数を共有する**という読み方はしません。`ellphi` 側に Mahalanobis の確率重みに相当する項を無理に足す予定はなく、比較の解釈は本節および `README.md` の英語節（*Backend comparison: outlier-probability weighting*）に従ってください。

## 図・定性出力

`docs/paper/` 以下の LaTeX の **すべての図を一括で出す単一スクリプトはありません**。原稿専用のアセットもあります。下の表は **コードに近い** 図の流れを示すものです。論文の図を増やしたら表も追記してください。

| 種類 | スクリプト / 場所 | 典型出力 |
|------|-------------------|----------|
| 学習中のスナップショット（楕円・点） | `tda_ml.trainer` → `tda_ml.visualization.visualize` | `<run_dir>/images/`、`result_epoch_*.png` を含むファイル名 |
| PD アニメ（任意 extra） | `tda_ml/reproduce_pd_animation.py`（`pyproject` の optional `repro-pd-animation` 参照） | `experiments/repro_pd_animation_final/frames/` 付近（スクリプトが `image_dir` を設定） |
| ノイズ感度プロット | `tda_ml/experiments/run_noise_sensitivity.py` | `outputs/metrics_vs_noise_level.png` |
| クラスタリングベンチマーク図 | `tda_ml/experiments/clustering_benchmark.py`（`--checkpoint` 必須） | `--output` で指定（既定 `outputs/clustering_benchmark.png`） |
| ロバストネススイープの図 | `tda_ml/robustness_sweep.py` | `important_results/robustness_*.png` |
| 学習 PNG の分割・後処理 | `tda_ml/split_results_images.py` | ユーザー指定の `--image_dir` / `--output_dir` |
| 静的な論文用アセット | `docs/paper/` 以下の LaTeX（`\includegraphics` のパスは各 `.tex` を参照） | 論文ビルドが参照する PNG/PDF（ローカル生成のこともあり、常に git に無いとは限らない） |

楕円描画は `tda_ml/visualization.py` と `tda_ml/geometry.py`（パラメータ → 共分散 → 描画）を参照してください。内部用の長い数式メモは公開 git には含めません。

## 自動テスト

ローカルでは:

```bash
uv run pytest
```

PR および **`main` と `feature/**` への push** のたびに、CI では **ruff**、**pytest**、軽量な **再現スモーク**（`experiments/run_backend_multiseed.py` を `--epochs 1`・1 seed・`mahalanobis` で実行）が走ります。**50 epoch × 5 seed × 2 backend の本番コマンドは CI では実行しません。**

## 論文提出時のスナップショット

論文投稿時は、提出結果と一致する **git コミットを固定**し、`CITATION.cff` およびリポジトリ URL を、正本の組織リポジトリ（公式が `uda-lab/TDA-ML` のときはそれ）と揃えて引用してください。
