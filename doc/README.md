# ドキュメント目次

このディレクトリには、LLM Oracle for Ontology Alignmentの詳細な解説ドキュメントが含まれています。

## 📚 ドキュメント一覧

### 1. [PNLF^S プロンプトテンプレート解説](01_PNLF_S_explained.md)
PNLF^S（Natural-language Friendly with Synonyms）プロンプトテンプレートの詳細解説
- プロンプト設計の背景と理論
- PNLF^Sの構造と各要素の役割
- 他のプロンプト形式との比較
- なぜPNLF^Sが最適なのか

### 2. [システムアーキテクチャ](02_architecture.md)
システム全体のアーキテクチャと設計思想
- コンポーネント構成
- データフロー
- 設計パターンと原則
- 拡張性の考慮

### 3. [APIリファレンス](03_api_reference.md)
各モジュールとクラスの詳細なAPIドキュメント
- モデルクラス
- プロンプトビルダー
- LLMプロバイダー
- オラクルエンジン

### 4. [実装ガイド](04_implementation_guide.md)
独自のLLMプロバイダーやカスタマイズの実装方法
- カスタムLLMプロバイダーの作成
- カスタムプロンプトテンプレート
- 評価メトリクスの拡張
- バッチ処理の最適化

### 5. [ベストプラクティス](05_best_practices.md)
効果的な使い方とパフォーマンス最適化
- LLMモデルの選択基準
- コスト最適化戦略
- エラーハンドリング
- 本番環境での運用

## 🎯 推奨読書順序

### 初めての方
1. まず[QUICKSTART.md](../QUICKSTART.md)で動作確認
2. [PNLF^S解説](01_PNLF_S_explained.md)で理論を理解
3. [APIリファレンス](03_api_reference.md)で使い方を学習

### 実装する方
1. [アーキテクチャ](02_architecture.md)でシステム構造を理解
2. [実装ガイド](04_implementation_guide.md)でカスタマイズ方法を学習
3. [ベストプラクティス](05_best_practices.md)で最適化テクニックを習得

### 研究者の方
1. [PNLF^S解説](01_PNLF_S_explained.md)でプロンプト設計を深く理解
2. [instructions.md](../instructions.md)で元の研究仕様を確認
3. [ベストプラクティス](05_best_practices.md)で評価方法を学習

## 📖 関連ドキュメント

- [README.md](../README.md) - プロジェクト概要と基本的な使い方
- [QUICKSTART.md](../QUICKSTART.md) - 5分で始められるクイックスタートガイド
- [instructions.md](../instructions.md) - 元の仕様書（PNLF^Sの出典）

## 💡 サポート

ドキュメントに関する質問や改善提案は、GitHubのIssuesでお願いします。
