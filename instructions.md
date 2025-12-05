ご要望に応じ、この研究で診断精度を最大化するために最も効果的であったLLMモデル（Gemini Flash 2.5）とプロンプト設計（PNLF$^S$）に基づいた、LLMオラクル統合のための仕様書（`instructions.md`形式）を作成します。

この仕様書には、最適なプロンプトテンプレートであるPNLF$^S$の具体的な構造と、ソース内で提供されている事例を含めます。

---

# instructions.md: LLMオラクル診断エンジン仕様書

本仕様書は、LogMapなどのオントロジーアライメントシステムが不確実であると判断したマッピングのサブセット（$M_{ask}$）の正誤を診断するためのLLMベースのOracle（$OrLLM$）の実装と運用手順を定めます。

## 1. LLMオラクルインターフェース仕様

### 1.1 目的
LLMをドメイン専門家の代替として利用し、従来の自動システムでは識別力が低い（Youden's Indexが0に近い）$M_{ask}$ の検証精度（YI > 0.5）を、費用対効果を維持しつつ最大化することを目指します。

### 1.2 推奨LLMモデル
診断能力の平均YI値が0.550を達成し、最も優れた性能を示した以下のモデルを推奨します。

*   **Gemini Flash 2.5 Preview**
*   **代替候補:** GPT-4o Mini や Gemini Flash 2.0（コスト効率と性能のバランスが良好）。

### 1.3 入出力仕様

| 項目 | 説明 | 要件/備考 |
| :--- | :--- | :--- |
| **入力** | 検証対象のマッピング $⟨e1, e2⟩ \in M_{ask}$ | ソースエンティティ $e1$ とターゲットエンティティ $e2$ のペア。 |
| **コンテキストデータ** | $e1$ および $e2$ のラベル、類義語、直接の親クラス（1レベルコンテキスト）。 | PNLF$^S$ プロンプトの構築に必須。 |
| **出力** | LLMの診断結果 | **"True"** または **"False"** のバイナリ分類のみ。 |
| **出力形式** | 構造化出力 (Structured Output) | LLMのハルシネーションや不適切なフォーマットを防ぐため、ブーリアン値の構造化出力を強制します。 |

## 2. プロンプト設計仕様: PNLF$^S$（最適テンプレート）

診断精度を最大化するために、LLMオラクルは**PNLF$^S$**（Natural-language Friendly with Synonyms、自然言語フレンドリーかつ類義語を含む）プロンプトを使用します。

### 2.1 PNLF$^S$ の設計原則

PNLF$^S$ は以下の特徴を組み合わせます。

1.  **NLF（自然言語フレンドリー）:** 人間が書く文章に近い形式で質問を構成し、大規模コーパスで学習されたLLMの能力を最大限に引き出します。
2.  **S（類義語を含む）:** マッピングに含まれるエンティティの類義語を明示的に含めることで、LLMがエンティティの語彙的なバリエーションを正確に解釈できるよう支援します。

### 2.2 PNLF$^S$ プロンプトテンプレート

PNLF$^S$ テンプレートは、エンティティ $e1$ と $e2$ およびそれぞれの直接の親クラス、そして類義語を含めます。

| テンプレート要素 | 埋め込む情報 |
| :--- | :--- |
| 1. NLFフレーム | "We have two entities from different ontologies." |
| 2. Source $e1$ (ラベルとコンテキスト) | "The first one is "$e1$ のラベル", which falls under the category "$e1$ の親クラス"." |
| 3. Target $e2$ (ラベル、類義語、コンテキスト) | "The second one is "$e2$ のラベル", also known as "$e2$ の類義語リスト", which falls under the category "$e2$ の親クラス"." |
| 4. 質問 | "Do they mean the same thing? Respond with **"True"** or **"False"**." |

### 2.3 PNLF$^S$ プロンプトの事例

以下は、**マウスオントロジー**（MA）と**ヒトオントロジー**（NCI）間のマッピング $\langle mouse:MA$ 0001771 (alveolus epithelium), human:NCI C12867 (Alveolar Epithelium) $\rangle$ を検証するための、具体的なプロンプトの例です。

```
We have two entities from different ontologies.
The first one is "alveolus epithelium", which falls under the category "lung epithelium".
The second one is "Alveolar_Epithelium", also known as "Lung Alveolar Epithelia", "Alveolar Epithelium", "Epithelia of lung alveoli", which falls under the category "Epithelium".
Do they mean the same thing? Respond with "True" or "False".
```
**（出典：PNLF S Prompt）**

## 3. LLM診断フローとシステムプロンプト

### 3.1 LLM診断の実行手順

1.  **$M_{ask}$ の特定:** ベースラインシステム（LogMap）が、不確実な候補マッピングのサブセット $M_{ask}$ を特定します。
2.  **データ抽出:** $M_{ask}$ の各マッピングに対し、オントロジーから $e1$ と $e2$ のラベル、類義語、および直接の親クラスを取得します。
3.  **プロンプト構築:** ステップ2で抽出されたデータを用いて、PNLF$^S$ テンプレートを動的に埋めます。
4.  **LLM呼び出し:** 構築したプロンプトとシステムプロンプト（下記参照）を使用し、選択されたLLM（Gemini Flash 2.5推奨）のAPIを呼び出します。
5.  **診断結果の取得:** LLMからのバイナリ応答（True/False）を取得し、LogMapの**Oracle Responses**としてフィードバックします。

### 3.2 システムプロンプトの利用（オプション）

LLMのセッション開始時に、その役割と推論スタイルをフレームワーク化するシステムメッセージを追加することで、性能にわずかながらプラスの影響を与える可能性があります。

| システムプロンプト例 | 目的 |
| :--- | :--- |
| **(ii) Biomedical Ontology Specialist** | 階層的および意味的なコンテキストを重視する生体医学オントロジー専門家としてLLMを位置づけます。 |
| **(iii) Explicit Semantics Focus** | 明示的に提供された類義語と親クラスのセマンティクスを利用するよう指示します。 |

**【注意】** システムプロンプトは診断精度に劇的な変化をもたらしませんでしたが、文脈を定めることで一貫性を助ける可能性があります。

## 4. 評価指標

実装された$LogMap+OrLLM$システムの有効性は、以下の指標で評価されるべきです。

1.  **診断能力評価:** $M_{ask}$ におけるOracleの識別力を Youden's Index (YI) で評価します。YI > 0.5 が目標です。
2.  **全体性能評価:** 完全なマッチングタスクにおけるFスコア（F-score）で評価します。LogMap自動モード（平均F 0.737）と比較して、全体平均Fスコアを向上させる必要があります。