# 再現性チェックリスト（D-1: seed・設定・依存固定）

## 目的

- D-1 として、再現実行に必要な seed 固定・設定固定・依存固定・実行手順固定の現状を監査し、再実行可能性を阻害する不足/曖昧点を明確化する。
- 根拠はローカル checkout の現存ファイル（`docs/spec/reproduction_targets.md`, `docs/spec/reproduction_commands.md`, `docs/audit/file_inventory.csv`, `docs/audit/repository_baseline.md`, `docs/audit/missing_inputs.md`, 実装/設定ファイル）に限定する。

## 監査対象と判定基準

- 判定基準
  - `OK`: 固定手段と実行根拠が揃い、第三者が同等条件で再実行できる。
  - `PARTIAL`: 一部固定されているが、入口不統一・不足記録・曖昧な運用が残る。
  - `NG`: 固定の仕組みが不足/不整合で、再実行時に結果差分リスクが高い。
- 優先度
  - `high`: 再現失敗や比較不能を直接引き起こす可能性が高い。
  - `medium`: ばらつきや運用ミスを増やすが、回避策がある。
  - `low`: 直ちに破綻しないが、将来的な監査性を下げる。

## seed固定の監査

- 対象: `configs/base.yaml:data.seed`, `src/main.py`, `experiments/run_v73_backend_multiseed.py`, `configs/multi_seed_config.yaml`
  - 現状: 単一 seed (`42`) と複数 seed (`42,123,456,789,1024` など) の両運用が実装されているが、再現ターゲットでどちらを正とするかは文書間で一意化されていない。
  - 根拠:
    - `configs/base.yaml` に `data.seed: 42` がある。
    - `src/main.py` は `data.seed` から `torch.Generator().manual_seed(seed)` と `noise_seed=seed` を用いてデータ生成の乱数を固定。
    - `experiments/run_v73_backend_multiseed.py` は `--seeds` デフォルト `[42,123,456,789,1024]` を上書き適用。
    - `configs/multi_seed_config.yaml` は `seeds: [42, 123, ...]` を持つが、同ファイルコメントが `multi_seed_training.py` を参照しており、現行入口との対応が曖昧。
  - 判定: `PARTIAL`
  - リスク: 再現対象を単一 seed の1回実行とみなすか、複数 seed 集計とみなすかで結論が変わる。
  - 推奨対応:
    - D系仕様に「公式 seed セット（単一/複数）」を固定し、優先順位を明記する。
    - 実行Issue側で `--seeds` を明示し、ログに seed 一覧を保存する。
  - 優先度: `high`

- 対象: Python/NumPy/random/CUDA の包括的 seed 制御
  - 現状: `torch.Generator` とデータノイズ seed は固定されるが、`random.seed`, `np.random.seed`, `torch.cuda.manual_seed_all`, deterministic algorithm 指定は確認できない。
  - 根拠:
    - `src/main.py` に `torch.Generator().manual_seed(seed)` はある。
    - リポジトリ横断検索で `np.random.seed` / `random.seed` / `torch.cuda.manual_seed_all` の統一初期化コードは確認できない。
  - 判定: `NG`
  - リスク: GPU実行や前処理の経路差で run-to-run ばらつきが残る。
  - 推奨対応:
    - seed初期化ユーティリティを1箇所に集約し、`torch`, `numpy`, `random`, CUDA を同時固定する。
    - 必要時は `torch.use_deterministic_algorithms(True)` と関連環境変数を方針化する。
  - 優先度: `high`

## 設定固定の監査

- 対象: 設定ロード規約（`configs/base.yaml` + `configs/<name>.yaml`）
  - 現状: マージ手順は実装上固定されているが、どの config を再現公式として採用するかはタスク単位依存。
  - 根拠:
    - `src/utils.py:load_config` が `configs/base.yaml` を読み、対象YAMLを再帰マージ。
    - `docs/spec/reproduction_commands.md` は「設定ファイル明示」「実行コマンド1本固定」を要求。
    - `docs/experiments/v73_backend_multiseed_reproducibility.md` は `reproduce_1209` を前提とするが、プロジェクト全体の唯一入口までは固定していない。
  - 判定: `PARTIAL`
  - リスク: Issueごとに config 選択が揺れ、比較条件が暗黙に変わる。
  - 推奨対応:
    - 再現ターゲットごとに「公式 config 名」を `docs/spec/reproduction_commands.md` へ固定追記する。
    - CLI引数と設定上書きの優先順位（base/env/script override）を仕様化する。
  - 優先度: `high`

- 対象: 実行環境差分パラメータ（`device`, `num_workers`, `enable_tf32`, `cudnn_benchmark`, `amp_dtype`）
  - 現状: 設定項目は存在するが、再現時に必須固定値としての明示は不足。
  - 根拠:
    - `configs/base.yaml` に `device: auto`, `performance.enable_tf32: true`, `cudnn_benchmark: true`, `use_amp: true`, `amp_dtype: float16`。
    - 複数 config で `device`, `num_workers`, `seed` が異なる値を持つ。
  - 判定: `PARTIAL`
  - リスク: CPU/GPU差、thread数、AMP dtype差でメトリクスが変動する。
  - 推奨対応:
    - 再現実行コマンドごとに `device`, thread, dtype, TF32 使用有無を明示する。
    - `auto` 依存を避け、再現Issueでは固定値設定を推奨する。
  - 優先度: `high`

## 依存固定の監査

- 対象: `pyproject.toml` / `uv.lock` / `requirements.txt` の整合
  - 現状: 3者不整合。`pyproject.toml` と `requirements.txt` は依存記述の粒度・内容が異なり、`uv.lock` は `tda-ml` の requires-dist が `ellphi` と `numpy` のみで `pyproject.toml` の主要依存（`torch`, `torchvision`, `gudhi`, `scikit-learn` 等）と一致しない。
  - 根拠:
    - `pyproject.toml` は `ellphi==0.1.2`, `numpy<2`, `torch>=2.0`, `torchvision>=0.15`, `gudhi>=3.4`, `scikit-learn>=1.3`, `scikit-image>=0.22`, `matplotlib>=3.8`, `tqdm>=4.65`, `pyyaml>=6`, `scipy>=1.10`, `torch_topological` を宣言。
    - `requirements.txt` は unpinned 行が多く、`geomloss`, `ipykernel` を含む一方で `gudhi`, `scikit-image` などが見当たらない。
    - `uv.lock` の `[[package]] name = "tda-ml"` の `requires-dist` が `ellphi`, `numpy` のみ。
  - 判定: `NG`
  - リスク: `uv sync` / `pip install -r requirements.txt` で解決結果が一致せず、同一コードでも実行可否や結果が変わる。
  - 推奨対応:
    - 依存インストール方式を `uv sync` に一本化し、`uv.lock` を `pyproject.toml` と整合する形で再生成・固定する。
    - `requirements.txt` は補助用途に限定するか、自動生成へ変更する。
  - 優先度: `high`

- 対象: Pythonバージョン固定
  - 現状: 最低バージョンは宣言されるが、厳密な単一バージョン固定（例: 3.12.x）ではない。
  - 根拠:
    - `pyproject.toml` と `uv.lock` は `requires-python = ">=3.12"`。
    - `docs/spec/reproduction_commands.md` は `python --version` 実行を要求するが、許容バージョン帯は未定義。
  - 判定: `PARTIAL`
  - リスク: 3.12/3.13 以降の差分で依存解決や数値挙動が変わる。
  - 推奨対応:
    - 再現実行用に `3.12.x` のような固定minorを明記し、Issueテンプレに記録欄を追加する。
  - 優先度: `medium`

- 対象: 外部サブモジュール `ellphi_repo`, `pytorch-topological` の再現上の扱い
  - 現状: `pytorch-topological` は `.gitmodules` 定義ありだが、両ディレクトリとも通常Gitディレクトリとして存在し、親リポジトリ側の gitlink 固定（コミットSHA追従）が監査上確認できない。
  - 根拠:
    - `.gitmodules` には `pytorch-topological` のみ記載。
    - `ellphi_repo/` と `pytorch-topological/` はそれぞれ独立 `.git` を持つディレクトリとして存在。
    - `docs/audit/file_inventory.csv` では `ellphi_repo` が `external:ellphi_repo` として分類されている。
  - 判定: `PARTIAL`
  - リスク: サブディレクトリ側 HEAD が動くと同名実験でも依存実装が変わる。
  - 推奨対応:
    - 外部依存の固定方式（git submodule/git subtree/commit hash記録）を一本化する。
    - 再現Issueに外部リポジトリの commit SHA を必須記録とする。
  - 優先度: `high`

- 対象: 依存インストール手順の再実行可能性
  - 現状: `docs/spec/reproduction_commands.md` は `uv sync` を正として定義済み。ただし上記ロック不整合により実効性は不足。
  - 根拠:
    - `docs/spec/reproduction_commands.md` に `uv sync` が必須コマンドとして記載。
    - `docs/ml_compute_optimization.md` でも `uv sync` 必要性への言及あり。
  - 判定: `PARTIAL`
  - リスク: 手順自体は明示されても、ロック不整合で再現できない環境が出る。
  - 推奨対応:
    - `uv sync` 後の import smoke test（主要依存）を Verification に固定追加する。
  - 優先度: `medium`

## 実行手順固定の監査

- 対象: 再現実行コマンドの入口一貫性（どのコマンドが正か）
  - 現状: 仕様は「実行コマンド1本固定」を要求するが、リポジトリ全体で公式入口が単一化されていない。
  - 根拠:
    - `docs/spec/reproduction_commands.md` は要件レベルの規定。
    - `docs/experiments/v73_backend_multiseed_reproducibility.md` は `uv run python experiments/run_v73_backend_multiseed.py` を提示。
    - 実装入口として `src/main.py` も存在し、用途ごとに複数入口が並立。
  - 判定: `PARTIAL`
  - リスク: 実行者が異なる入口を選び、設定上書きや出力が一致しない。
  - 推奨対応:
    - ターゲット別に「正規入口コマンド」を1つずつ `docs/spec/reproduction_commands.md` に固定する。
  - 優先度: `high`

- 対象: 出力先規約（`outputs/...`）の再現性
  - 現状: `config_id + timestamp` 命名で一意保存されるが、同時に時刻依存のため完全同一パス再現は不可。成果物パス追跡には補助CSV運用が必要。
  - 根拠:
    - `src/main.py` が `run_dir = <base_dir>/<config_id>_<timestamp>` を生成。
    - `docs/experiments/v73_backend_multiseed_reproducibility.md` は `progress_summary.csv` / `backend_stats.csv` を追跡基盤として定義。
  - 判定: `PARTIAL`
  - リスク: run比較時に成果物対応がログ依存になり、手作業誤対応が起きやすい。
  - 推奨対応:
    - 実行Issueに `run_dir` 記録を必須化し、成果物マニフェストを固定出力する。
  - 優先度: `medium`

- 対象: 実行環境差分（CPU/GPU, thread, dtype）の明示有無
  - 現状: コード/設定には項目があるが、再現仕様としての必須明記は不足。
  - 根拠:
    - `src/main.py` は `device:auto` で CUDA/MPS/CPU を自動選択。
    - `src/main.py` は `num_workers`, pin_memory, prefetch_factor を環境依存で解決。
    - `docs/spec/reproduction_commands.md` は環境確認の大枠のみで、CPU/GPUやdtype固定値の規定はない。
  - 判定: `NG`
  - リスク: 同じコマンドでも実行環境により計算経路・速度・数値が変わる。
  - 推奨対応:
    - 再現対象ごとに「許容環境プロファイル（CPU/GPU, thread, dtype, TF32）」を明記する。
    - 可能なら `device` と主要 performance パラメータを固定値で配布する。
  - 優先度: `high`

## リスク一覧（優先度付き）

- `high`: `pyproject.toml` / `uv.lock` / `requirements.txt` の不整合により依存再現が破綻しうる。
- `high`: 公式実行入口と公式seedセットがタスク横断で一意化されていない。
- `high`: CPU/GPU・thread・dtype の固定方針が不足し、環境差分で結果が揺れる。
- `high`: 外部依存 (`ellphi_repo`, `pytorch-topological`) のコミット固定運用が監査上不明確。
- `medium`: timestamp付き出力によりパス再現性が弱く、成果物追跡が運用依存。
- `medium`: Pythonバージョンが `>=3.12` までで、実行minorの固定がない。

## 要owner判断

- `ellphi_repo` の管理方式をどう固定するか（外部参照/サブモジュール化/コミット記録義務化）。
  - 理由: 現状は親側固定点が見えず、再実行時に外部コード版がずれる。
- `pytorch-topological` を親リポジトリでどの粒度で固定するか（gitlink固定か、独立管理のままか）。
  - 理由: `.gitmodules` 記載と実体運用の整合が曖昧。
- 再現公式 seed ポリシーを単一値運用にするか、複数 seed 集計を正にするか。
  - 理由: 設定・スクリプト・文書で両方が併存し、成功判定値に直結する。
- 再現対象ごとの正規実行入口（`src/main.py` 系か実験ラッパースクリプト系か）をどこまで固定するか。
  - 理由: 入口ごとに上書き規則と出力構造が変わる。
- 再現許容環境（CPU/GPU, thread, dtype, TF32/AMP）をどの範囲で許容するか。
  - 理由: 現状は `auto` 選択が多く、比較公正性の境界が未定義。

## README/docs導線整備 更新ログ（E-1, #18）

- 更新日: 2026-04-27
- 対象:
  - `README.md`
  - `docs/spec/reproduction_commands.md`
  - `docs/audit/reproducibility_checklist.md`
- 反映内容:
  - README に再現の最短導線（環境準備 → 正規入口実行 → 出力確認）を追加。
  - README に公式入口コマンドを1本で明示し、`src/main.py` 直接実行を非公式と明記。
  - README に Known Gaps（seed, 環境差分, 依存固定, 外部依存未確定）を追加。
  - `docs/spec/reproduction_commands.md` に正規入口・固定パラメータ（seed/config/device/thread/dtype）・`outputs` 規約・最小 Verification を追加。
  - 判定不能事項は本書 `## 要owner判断` へ集約した。

## 判定変化（E-1反映）

- 改善:
  - 実行手順固定の監査「再現実行コマンドの入口一貫性」: `PARTIAL` のままではあるが、README と仕様書で正規入口を明文化し運用揺れを縮小。
  - 設定固定の監査「実行環境差分パラメータ明示」: `PARTIAL` のままではあるが、device/thread/dtype を固定記録対象として仕様に追加。
  - 実行手順固定の監査「出力先規約」: `PARTIAL` のままではあるが、`outputs/v73_backend_compare` と主要成果物確認を仕様化。
- 据え置き:
  - seed固定の包括制御（`random`/`numpy`/CUDA deterministic）の不足: `NG` 据え置き。
  - 依存固定（`pyproject.toml` / `uv.lock` / `requirements.txt` 不整合）: `NG` 据え置き。
  - 外部依存のコミット固定運用不明確: `PARTIAL` 据え置き（要owner判断）。
- 悪化:
  - なし。

## F-1 公開前ゲート整備 更新ログ（#19）

- 更新日: 2026-04-27
- 対象:
  - `README.md`
  - `docs/audit/reproducibility_checklist.md`
  - `docs/audit/publication_decisions.md`
- 反映内容:
  - README に「公開前ゲート（Release Gate）」を追加し、CI最小要件・秘密情報・巨大ファイルの公開判定条件を断定記述した。
  - 公開可否判定の正本を `docs/audit/publication_decisions.md` に統一し、判定不能事項は `## 要owner判断` へ送る運用を固定した。
  - 既存の公開境界分類（A-1）と公開可否基準（F-1）が混在しうる点を明示し、公開判定は F-1 基準を優先することを追記した。

## ゲート判定（F-1）

- CI最小要件: `OK`（ワークフロー実装済み。マージ時の強制ブロックは Branch protection の必須チェック登録が前提）
  - 根拠: `.github/workflows/ruff.yml` が Pull request および `main` への push で `uv sync --all-groups --frozen` 後に `uv run ruff check .` を実行し、失敗時にジョブが失敗する。
  - 運用上の残差: リポジトリ設定で当該チェックを必須ステータスに指定しない限り、マージブロックは保証されない（設定側の作業）。
- 秘密情報: `PARTIAL`
  - 根拠: 取り扱いルールは文書化したが、全履歴を含む機械スキャン結果は未添付。
  - 必要対応: 公開直前に秘密情報スキャン（履歴含む）を実行し、検証ログを残す。
- 巨大ファイル: `PARTIAL`
  - 根拠: しきい値（10 MiB 禁止、1-10 MiB は要owner判断）を定義したが、現存大容量ファイルの最終処置は未判定。
  - 必要対応: 容量超過候補の棚卸し結果を `publication_decisions.md` へ反映する。

## 判定変化（F-1反映）

- 改善:
  - 公開前判定基準（CI/秘密情報/巨大ファイル）が README と監査文書で機械確認可能な形に明文化された。
  - 判定不能事項の送付先を `## 要owner判断` に一本化し、公開判定フローの曖昧さを削減した。
- 据え置き:
  - 秘密情報・巨大ファイルの実地スキャン結果は未添付のため `PARTIAL` 据え置き。
- 悪化:
  - なし。

## 判定変化（#23 / G-4 F-1 CI 最低ゲート反映）

- 改善:
  - F-1 の「CI最小要件」: GitHub Actions による `uv run ruff check .` の自動実行が実装され、`PARTIAL` から `OK`（上記 Branch protection 前提の注記付き）へ更新した。
- 据え置き:
  - 秘密情報・巨大ファイルゲートは従来どおり `PARTIAL`。
- 悪化:
  - なし。

## G-4 F-1 CI 最低ゲート 実装ログ（#23）

- 更新日: 2026-04-28
- 対象:
  - `.github/workflows/ruff.yml`
  - `README.md`
  - 本書（`docs/audit/reproducibility_checklist.md`）
- 反映内容:
  - `astral-sh/setup-uv` と Python 3.12、`uv sync --all-groups --frozen`、`uv run ruff check .` を PR / `main` push で実行するワークフローを追加した。
  - `pytorch-topological` は `[tool.uv.sources]` の path 依存のため、`actions/checkout` で `submodules: recursive` を指定した。
  - README と本書に CI 実行内容およびローカル再現コマンド（`uv run ruff check .`）を追記し、F-1 の正規コマンドとの整合を維持した。

## 検証ログ

- 監査実施日: 2026-04-27
- 監査対象入力:
  - `docs/spec/reproduction_targets.md`
  - `docs/spec/reproduction_commands.md`
  - `docs/audit/file_inventory.csv`
  - `docs/audit/repository_baseline.md`
  - `docs/audit/missing_inputs.md`
  - `pyproject.toml`, `uv.lock`, `requirements.txt`
  - `src/main.py`, `src/utils.py`, `experiments/run_v73_backend_multiseed.py`, `configs/*.yaml`, `.gitmodules`
- 予定検証コマンド:
  - `python3 - <<'PY' ... required sections check ... PY`
  - `"$HOME/.local/bin/rg" "^##|OK|PARTIAL|NG|high|medium|low|seed|config|dependency|uv.lock|requirements|要owner判断" docs/audit/reproducibility_checklist.md`
  - `"$HOME/.local/bin/rg" "公開前ゲート|CI|秘密情報|巨大ファイル|要owner判断|公開可|条件付き公開|公開不可" README.md docs/audit/reproducibility_checklist.md docs/audit/publication_decisions.md`
