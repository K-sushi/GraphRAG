# GraphRAG Implementation with LightRAG & n8n

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LightRAG](https://img.shields.io/badge/LightRAG-Latest-blue.svg)](https://github.com/HKUDS/LightRAG)
[![n8n](https://img.shields.io/badge/n8n-Compatible-green.svg)](https://github.com/n8n-io/n8n)
[![CLAUDEFLOW](https://img.shields.io/badge/CLAUDEFLOW-Optimized-purple.svg)](https://github.com/K-sushi/GraphRAG)

このプロジェクトは、LightRAG（シンプルで高速なRetrieval-Augmented Generationフレームワーク）とn8n（柔軟なAIワークフロー自動化プラットフォーム）を組み合わせた、堅牢で効率的なGraphRAGシステムの実装です。

## 🚀 概要

### 主な機能

- **知識グラフの自動構築**: LightRAGがドキュメントからエンティティと関係を自動抽出し、知識グラフを構築
- **マルチモーダルデータ処理**: PDF、画像、テーブル、数式などの多様な形式の文書を処理（RAG-Anything連携）
- **高度な検索機能**: セマンティック検索、グラフトラバーサル、ハイブリッド検索、リランキングを組み合わせ
- **柔軟なLLM統合**: Gemini 2.5モデル（Pro/Flash/Flash-Lite）を含む多様なLLMモデルをサポート
- **データインジェッションの自動化**: n8nワークフローによる自動文書取得とインジェッション
- **エージェント思考のRAG**: n8nのAIエージェントによる状況に応じたツール呼び出し
- **トークン使用量トラッキング**: コスト最適化のためのトークン消費量監視

### アーキテクチャ概要

```mermaid
graph TD
    subgraph "n8n AI エージェントシステム"
        U[ユーザー] --> V[n8n AI Agent];
        V -- "クエリルーティング (LLM/Gemini Agent)" --> W{ツールの呼び出し};
        W -- "知識グラフ検索" --> X[LightRAG Retrieval (API)];
        W -- "標準RAG検索" --> Y[n8n/Superbase Vector Store];
        W -- "Web検索/その他" --> Z[追加のn8nツール];
        X --> AA[LightRAGコンテキスト (JSON)];
        Y --> BB[関連チャンク];
        Z --> CC[追加情報];
        AA --> DD[n8n内のLLM (例: Gemini 2.5 Pro)];
        BB --> DD;
        CC --> DD;
        DD --> EE[最終回答生成];
        EE --> V;

        subgraph "n8n データインジェッションパイプライン"
            F1[データソース (Google Drive, Web Scraper)] --> F2[n8nドキュメント処理];
            F2 -- "コンテンツ&メタデータ抽出 (LLM/Gemini 2.5 Flash)" --> F3[LightRAG Ingestion (API)];
            F2 --> F4[n8n/Superbase Vector Store Ingestion];
            F3 --> F5[LightRAG知識グラフ & ベクトルストア];
            F4 --> F6[n8n/Superbase レコードマネージャー];
        end
    end
```

## 🛠️ CLAUDEFLOW 実装ガイド

### Phase 1: 初期セットアップ

```bash
# プロジェクト環境の準備
/implement "GraphRAG project initialization" --type project --framework docker

# LightRAGサーバーのセットアップ
/build lightrag/server --type api --framework fastapi --deploy render

# n8nワークフロー環境の構築
/implement "n8n workflow environment" --type service --framework n8n
```

### Phase 2: 設定とデプロイ

```bash
# Docker環境の構築
/build deployment/docker --type container --orchestrate compose

# 環境変数とシークレットの設定
/implement config/environment --type configuration --secure

# LightRAGカスタムモデル統合
/implement lightrag/custom-models --type integration --llm gemini
```

### Phase 3: ワークフロー開発

```bash
# n8nワークフローテンプレートの作成
/design n8n/workflows --type template --pattern ingestion,query

# APIエンドポイントの実装
/implement docs/api --type documentation --format openapi

# テストスイートの構築
/test tests/ --type comprehensive --coverage 80%
```

### Phase 4: 最適化と監視

```bash
# パフォーマンス分析と最適化
/analyze --focus performance --scope system

# セキュリティ強化
/improve --focus security --validate --persona-security

# ドキュメント生成
/document docs/ --type comprehensive --persona-scribe=ja
```

## 📊 LightRAG ワークフロー図

### 文書取り込み（インジェッション）プロセス

```mermaid
graph TD
    A[ドキュメントのアップロード] --> B{フィルタリング & 重複排除};
    B --> C[ドキュメントのチャンク化];
    C --> D[チャンクの埋め込み & ベクトルストアへの保存];
    D --> E[LLM: エンティティ & 関係の抽出];
    E --> F{解析, 変換 & マージ (情報収集)};
    F --> G[エンティティ解決 & 記述の生成 (LLM)];
    G --> H[セマンティック検索用埋め込みの作成];
    H --> I[ベクトルがセマンティック検索DBに保存];
    I --> J[エンティティ & 関係がグラフDBに保存];
```

### 検索・クエリプロセス（Mixモード）

```mermaid
graph TD
    K[ユーザーが質問] --> L[ローカル & グローバルキーワードの抽出];
    L --> M[キーワードの埋め込み];
    M --> N[セマンティック検索 (エンティティ & 関係)];
    N --> O[グラフトラバーサル (1ホップ近隣)];
    O --> P[テキストチャンクの取得];
    P --> Q[リランキング (クロスエンコーダー)];
    Q --> R[コンテキスト (エンティティ, 関係, 上位チャンク) をLLMへ];
    R --> S[回答生成];
    S --> T[ユーザーへ回答返却];
```

## ⚙️ システム設定

### 必要要件

- **Python**: 3.8+
- **Node.js**: 16+
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Git**: 2.30+

### 推奨スペック

- **RAM**: 16GB以上
- **Storage**: 50GB以上の空き容量
- **CPU**: 4コア以上
- **GPU**: CUDA対応（オプション）

## 🔧 クイックスタート

### 1. リポジトリのクローン

```bash
git clone https://github.com/K-sushi/GraphRAG.git
cd GraphRAG-Implementation
```

### 2. 環境変数の設定

```bash
cp config/environment/.env.example .env
# .envファイルを編集してAPIキーやデータベース接続情報を設定
```

### 3. Docker環境の起動

```bash
docker-compose up -d
```

### 4. LightRAGサーバーの初期化

```bash
# LightRAGサーバーの起動確認
curl http://localhost:8000/health

# 初期データの投入
python scripts/setup/init_lightrag.py
```

### 5. n8nワークフローのインポート

```bash
# n8n Web UI（http://localhost:5678）にアクセス
# n8n/workflows/内のテンプレートをインポート
```

## 📋 プロジェクト構造

```
GraphRAG-Implementation/
├── config/                    # 設定ファイル
│   ├── models/               # AIモデル設定
│   ├── lightrag/            # LightRAG設定
│   ├── n8n/                 # n8n設定
│   └── environment/         # 環境変数
├── docs/                     # ドキュメント
│   ├── architecture/        # アーキテクチャ図
│   ├── api/                 # API仕様
│   └── guides/              # 使用ガイド
├── deployment/              # デプロイメント
│   ├── docker/             # Dockerファイル
│   ├── render/             # Render.com設定
│   └── scripts/            # デプロイスクリプト
├── lightrag/               # LightRAG関連
│   ├── server/            # サーバー実装
│   ├── client/            # クライアント実装
│   └── custom-models/     # カスタムモデル
├── n8n/                   # n8n関連
│   ├── workflows/         # ワークフローテンプレート
│   ├── templates/         # ノードテンプレート
│   └── tools/             # カスタムツール
├── examples/              # 実装例
│   ├── basic/            # 基本例
│   └── advanced/         # 高度な例
├── scripts/              # ユーティリティスクリプト
│   ├── setup/           # セットアップスクリプト
│   ├── deployment/      # デプロイスクリプト
│   └── utils/           # ユーティリティ
├── tests/               # テストスイート
│   ├── unit/           # ユニットテスト
│   ├── integration/    # 統合テスト
│   └── e2e/            # エンドツーエンドテスト
└── assets/              # アセット
    ├── diagrams/       # 図表
    └── screenshots/    # スクリーンショット
```

## 🔍 詳細ガイド

### [LightRAG設定ガイド](docs/guides/lightrag-setup.md)
### [n8nワークフロー設定ガイド](docs/guides/n8n-workflow-setup.md)
### [Geminiモデル統合ガイド](docs/guides/gemini-integration.md)
### [デプロイメントガイド](docs/guides/deployment.md)
### [トラブルシューティング](docs/guides/troubleshooting.md)

## 📚 CLAUDEFLOW コマンドリファレンス

### 分析コマンド

```bash
# システム全体の分析
/analyze --scope system --focus architecture --ultrathink

# パフォーマンス分析
/analyze --focus performance --scope project --persona-performance

# セキュリティ分析
/analyze --focus security --validate --persona-security
```

### 実装コマンド

```bash
# 新機能の実装
/implement "RAG query optimization" --type feature --framework lightrag

# APIエンドポイント実装
/implement "GraphRAG API endpoints" --type api --framework fastapi

# ワークフロー実装
/implement "document ingestion workflow" --type workflow --framework n8n
```

### 改善コマンド

```bash
# コード品質改善
/improve --focus quality --scope codebase --loop

# パフォーマンス最適化
/improve --focus performance --validate --persona-performance

# セキュリティ強化
/improve --focus security --safe-mode --persona-security
```

### テストコマンド

```bash
# 包括的テスト実行
/test --type comprehensive --coverage 80%

# 統合テスト
/test tests/integration --type integration --playwright

# E2Eテスト
/test tests/e2e --type e2e --playwright
```

### ドキュメント生成

```bash
# 技術ドキュメント生成
/document docs/api --type technical --persona-scribe=ja

# ユーザーガイド生成
/document docs/guides --type user-guide --persona-mentor
```

## 🤝 コントリビューション

プロジェクトへの貢献を歓迎します！以下の手順でご参加ください：

1. フォークを作成
2. フィーチャーブランチを作成 (`git checkout -b feature/AmazingFeature`)
3. 変更をコミット (`git commit -m 'Add some AmazingFeature'`)
4. ブランチにプッシュ (`git push origin feature/AmazingFeature`)
5. プルリクエストを作成

### CLAUDEFLOW 開発フロー

```bash
# 新機能開発の開始
/task "Implement new GraphRAG feature" --priority high

# 開発プロセス
/implement --loop --validate --test

# レビューとマージ
/analyze --focus quality --persona-reviewer
/git "create pull request" --validate
```

## 📄 ライセンス

このプロジェクトは[MIT License](LICENSE)の下で公開されています。

## 🔗 関連リンク

- [LightRAG GitHub](https://github.com/HKUDS/LightRAG)
- [n8n GitHub](https://github.com/n8n-io/n8n)
- [プロジェクトリポジトリ](https://github.com/K-sushi/GraphRAG)
- [CLAUDEFLOW フレームワーク](https://claude.ai/code)

## 📞 サポート

問題や質問がある場合は、以下の方法でサポートを受けることができます：

- GitHub Issues: バグ報告や機能リクエスト
- Discussions: 一般的な質問や議論
- Wiki: 詳細なドキュメントと FAQ

---

**Built with ❤️ using CLAUDEFLOW and SuperClaude Framework**