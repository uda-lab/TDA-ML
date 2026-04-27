# TDA-ML

本リポジトリの再現実行は、`docs/spec/reproduction_commands.md` で定義する正規入口に従う。

## 再現の最短導線

再現は以下の順序を正とする。

1. 環境準備: `uv sync`
2. 正規入口コマンド実行: `uv run python experiments/run_v73_backend_multiseed.py --base-config reproduce_1209 --epochs 50 --seeds 42 123 456 789 1024 --backends mahalanobis ellphi --out-base outputs/v73_backend_compare`
3. 出力確認: `outputs/v73_backend_compare/progress_summary.csv` と `outputs/v73_backend_compare/backend_stats.csv` の生成、および各 run ディレクトリ内 `logs/metrics.csv` の存在確認

## 公式入口コマンド

再現実行の公式入口コマンドは次の1本に固定する。

```bash
uv run python experiments/run_v73_backend_multiseed.py \
  --base-config reproduce_1209 \
  --epochs 50 \
  --seeds 42 123 456 789 1024 \
  --backends mahalanobis ellphi \
  --out-base outputs/v73_backend_compare
```

`src/main.py` の直接実行は公式再現入口として扱わない。

## 参照ドキュメント

- 仕様（再現対象）: `docs/spec/reproduction_targets.md`
- 仕様（再現コマンド）: `docs/spec/reproduction_commands.md`
- 監査（再現性チェック）: `docs/audit/reproducibility_checklist.md`
- 監査（公開可否判定）: `docs/audit/publication_decisions.md`
- 監査（現況ベースライン）: `docs/audit/repository_baseline.md`
- 監査（不足入力）: `docs/audit/missing_inputs.md`
- 実験プロトコル詳細: `docs/experiments/v73_backend_multiseed_reproducibility.md`

## 公開前ゲート（Release Gate）

本リポジトリの公開可否は、以下3ゲートをすべて満たした場合にのみ `公開可` と判定する。  
1つでも未達がある場合は公開しない。判定不能な事項は `docs/audit/publication_decisions.md` の `要owner判断` に送る。

### 1) CI最小要件（必須）

- 自動チェック入口コマンドは `uv run ruff check .` を正とする。
- 当該コマンドが失敗した状態（exit code non-zero）では公開不可とする。
- CI導線の監査結果と例外の扱いは `docs/audit/reproducibility_checklist.md` と `docs/audit/publication_decisions.md` に一本化する。

### 2) 秘密情報チェック（必須）

- `.env`、秘密鍵（例: `*.pem`, `*.key`）、トークン、認証情報ファイル（例: `credentials*.json`, `*.p12`）は Git 管理禁止とする。
- 認証情報を含む可能性がある設定値やログは、公開前に除去またはマスク済みであることを必須とする。
- 秘密情報混入が疑われるが機械判定できない場合は公開停止とし、`要owner判断` で最終判定する。

### 3) 巨大ファイルチェック（必須）

- しきい値は 10 MiB とし、10 MiB 超のファイルは Git 管理禁止とする。
- 1 MiB 超 10 MiB 以下のファイルは、再現性に不可欠な根拠がある場合のみ `要owner判断` で例外可否を判定する。
- 生成物（特に `outputs/` 配下）を公開リポジトリへ含める場合は、容量・再生成可能性・公開範囲を `publication_decisions.md` で明示しない限り公開不可とする。

## Known Gaps

以下は現時点で既知の差分要因であり、再現結果の解釈時に必ず参照する。

- seed運用: 単一 seed と複数 seed 集計の両運用が実装上併存するため、公式比較は本 README と `docs/spec/reproduction_commands.md` に記載の5-seed固定を正とする。
- 環境差分: `device` 自動選択や thread 数差分で結果が変動しうるため、再現時は `device` / thread / dtype を実行ログへ記録する。
- 依存固定: `uv sync` を唯一の導入手段とするが、`uv.lock` と他依存定義の整合は継続監査対象である。
- 外部依存の未確定点: `ellphi_repo` と `pytorch-topological` の固定運用（commit 固定方式）は要owner判断である。