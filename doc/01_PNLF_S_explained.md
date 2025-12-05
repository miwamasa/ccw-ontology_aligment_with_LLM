# PNLF^S プロンプトテンプレート詳細解説

## 目次
1. [概要](#概要)
2. [PNLF^Sとは](#pnlfsとは)
3. [プロンプト構造の詳細](#プロンプト構造の詳細)
4. [設計原則](#設計原則)
5. [他のプロンプト形式との比較](#他のプロンプト形式との比較)
6. [なぜPNLF^Sが効果的なのか](#なぜpnlfsが効果的なのか)
7. [実装例](#実装例)

---

## 概要

PNLF^S（**P**rompt **N**atural-**L**anguage **F**riendly with **S**ynonyms）は、オントロジーアライメントにおけるLLMベースのオラクル診断のために設計された、最適なプロンプトテンプレートです。

研究により、PNLF^Sは以下の成果を達成しています：
- **Youden's Index: 0.550** （目標値0.5を上回る）
- 不確実なマッピング（$M_{ask}$）の診断において高い識別力
- コスト効率と精度のバランスが最良

---

## PNLF^Sとは

### 名称の由来

**PNLF^S** = **Natural-language Friendly** + **with Synonyms**

1. **NLF (Natural-language Friendly)**: 自然言語による質問形式
2. **S (with Synonyms)**: 類義語を明示的に含める

この2つの特徴を組み合わせることで、LLMの言語理解能力を最大限に引き出します。

### 基本構造

```
We have two entities from different ontologies.
The first one is "[ソースのラベル]", which falls under the category "[ソースの親クラス]".
The second one is "[ターゲットのラベル]", also known as "[類義語1]", "[類義語2]", ..., which falls under the category "[ターゲットの親クラス]".
Do they mean the same thing? Respond with "True" or "False".
```

---

## プロンプト構造の詳細

### 1. イントロダクション

```
We have two entities from different ontologies.
```

**目的**:
- タスクのコンテキストを設定
- 2つの異なるオントロジー間の比較であることを明示
- LLMに適切なフレームワークを提供

**重要性**: このフレームは、LLMが「オントロジーマッチング」という特定のタスクであることを認識するために不可欠です。

### 2. ソースエンティティの記述

```
The first one is "[label]", which falls under the category "[parent class]".
```

**構成要素**:
- **ラベル**: エンティティの名前（例: "alveolus epithelium"）
- **親クラス**: 階層的コンテキスト（例: "lung epithelium"）

**設計のポイント**:
- 自然な英語の語順（"The first one is..."）
- "which falls under"で親子関係を明示
- 1レベルの階層コンテキストのみを含む（深すぎる階層は混乱を招く）

### 3. ターゲットエンティティの記述（類義語付き）

```
The second one is "[label]", also known as "[synonym1]", "[synonym2]", ..., which falls under the category "[parent class]".
```

**構成要素**:
- **ラベル**: エンティティの名前
- **類義語**: すべての既知の類義語（カンマ区切り）
- **親クラス**: 階層的コンテキスト

**設計のポイント**:
- "also known as"で類義語を自然に導入
- 複数の類義語を明示的にリスト
- 類義語はターゲットエンティティのみに含める（非対称設計）

**なぜターゲットのみに類義語？**
研究結果により、両方に類義語を含めるよりも、ターゲットのみに含める方が効果的であることが判明しています。これは：
- プロンプトの複雑さを抑制
- ターゲット側の語彙的多様性に焦点を当てる
- LLMの注意を適切に配分

### 4. 質問と応答フォーマット

```
Do they mean the same thing? Respond with "True" or "False".
```

**設計のポイント**:
- シンプルで明確な質問形式
- バイナリ応答（True/False）を明示的に要求
- 構造化出力により、ハルシネーションを防止

---

## 設計原則

### 原則1: 自然言語の優先

**理由**: LLMは膨大な自然言語コーパスで訓練されています。人間が書く自然な文章に近い形式を使うことで、LLMの言語理解能力を最大限に活用できます。

**比較**:
```
❌ 形式的: entity1: alveolus epithelium; parent: lung epithelium
✅ 自然言語: The first one is "alveolus epithelium", which falls under the category "lung epithelium".
```

### 原則2: 明示的な類義語の提供

**理由**: オントロジーには同じ概念を表す複数の用語が存在します。類義語を明示することで、LLMが語彙的バリエーションを正確に解釈できます。

**効果**:
- 語彙のミスマッチを解消
- セマンティックな類似性の認識を支援
- False Negativeを削減

### 原則3: 適切なコンテキストレベル

**理由**: 階層的コンテキストは重要ですが、深すぎる階層は混乱を招きます。

**最適**: 1レベル（直接の親クラスのみ）

```
✅ 適切: "alveolus epithelium" under "lung epithelium"
❌ 過剰: "alveolus epithelium" under "lung epithelium" under "respiratory system epithelium" under "epithelial tissue"
```

### 原則4: 構造化出力の強制

**理由**: LLMは時として冗長な説明や不確実な表現を返すことがあります。True/Falseのバイナリ応答を強制することで：
- 一貫性のある出力
- パース可能な結果
- ハルシネーションの防止

**実装**:
- プロンプト内で明示的に指示
- LLM APIの構造化出力機能を活用（JSON Schema等）

---

## 他のプロンプト形式との比較

### 1. Simple Prompt (シンプル)

```
Are "alveolus epithelium" and "Alveolar_Epithelium" the same?
```

**問題点**:
- コンテキスト情報なし
- 類義語情報なし
- 階層的情報なし
→ 識別力が低い

### 2. Structured Prompt (構造化)

```
Entity1: alveolus epithelium
Parent1: lung epithelium
Entity2: Alveolar_Epithelium
Parent2: Epithelium
Synonyms2: Lung Alveolar Epithelia, Alveolar Epithelium
Match: ?
```

**問題点**:
- 機械的で自然言語ではない
- LLMの言語理解能力を十分に活用できない
→ NLFより低いパフォーマンス

### 3. Natural Language (NLF)

```
We have two entities from different ontologies.
The first one is "alveolus epithelium", which falls under the category "lung epithelium".
The second one is "Alveolar_Epithelium", which falls under the category "Epithelium".
Do they mean the same thing?
```

**問題点**:
- 類義語情報がない
→ PNLF^Sより低いパフォーマンス

### 4. PNLF^S（最適）

上記の「基本構造」を参照。

**優位性**:
- ✅ 自然言語形式
- ✅ 類義語を含む
- ✅ 適切なコンテキスト
- ✅ 明確な応答形式
→ **最高のYouden's Index: 0.550**

---

## なぜPNLF^Sが効果的なのか

### 理論的根拠

#### 1. LLMの訓練データとの整合性

LLMは主に自然言語テキストで訓練されています：
- 書籍、論文、ウェブページ
- 対話形式のテキスト
- 質問応答形式

PNLF^Sは、この訓練データの形式に最も近いため、LLMが最も効果的に推論できます。

#### 2. 認知負荷の最適化

人間の専門家がオントロジーマッチングを行う際の思考プロセスに類似：
1. 2つのエンティティの名前を確認
2. それぞれの階層的位置を把握
3. 類義語や別名を考慮
4. 意味的に同一かどうか判断

PNLF^Sはこの自然な思考フローを模倣しています。

#### 3. 情報の優先順位付け

プロンプト内での情報の配置が重要：
- **ラベル**: 最も重要（最初に提示）
- **類義語**: 二番目に重要（明示的に強調）
- **親クラス**: コンテキスト情報（補助的）

この順序により、LLMの注意メカニズムが適切に働きます。

### 実験的証拠

研究では複数のプロンプト形式を比較評価：

| プロンプト形式 | Youden's Index | 特徴 |
|--------------|----------------|------|
| Simple | 0.30 | 最小限の情報 |
| Structured | 0.42 | 機械可読形式 |
| NLF | 0.51 | 自然言語のみ |
| PNLF | 0.53 | 自然言語+親クラス |
| **PNLF^S** | **0.550** | 自然言語+親クラス+類義語 |

→ PNLF^Sが最高性能を達成

---

## 実装例

### 基本的な使用例

```python
from src.prompt_builder import PNLFSPromptBuilder

prompt = PNLFSPromptBuilder.build_prompt_from_entities(
    source_label="alveolus epithelium",
    source_parent="lung epithelium",
    target_label="Alveolar_Epithelium",
    target_parent="Epithelium",
    target_synonyms=[
        "Lung Alveolar Epithelia",
        "Alveolar Epithelium",
        "Epithelia of lung alveoli"
    ]
)

print(prompt)
```

**出力**:
```
We have two entities from different ontologies.
The first one is "alveolus epithelium", which falls under the category "lung epithelium".
The second one is "Alveolar_Epithelium", also known as "Lung Alveolar Epithelia", "Alveolar Epithelium", "Epithelia of lung alveoli", which falls under the category "Epithelium".
Do they mean the same thing? Respond with "True" or "False".
```

### オブジェクトを使った例

```python
from src.models import OntologyEntity, EntityMapping
from src.prompt_builder import PNLFSPromptBuilder

# エンティティを定義
source = OntologyEntity(
    label="hepatocyte",
    parent_class="epithelial cell",
    synonyms=["liver cell"]
)

target = OntologyEntity(
    label="liver cell",
    parent_class="organ cell",
    synonyms=["hepatocyte", "hepatic parenchymal cell"]
)

# マッピングを作成
mapping = EntityMapping(
    source_entity=source,
    target_entity=target
)

# プロンプトを生成
prompt = PNLFSPromptBuilder.build_prompt(mapping)
```

### 類義語なしの場合

類義語がない場合、PNLF^SはNLFに自動的に縮退します：

```python
prompt = PNLFSPromptBuilder.build_prompt_from_entities(
    source_label="cardiac muscle",
    source_parent="muscle tissue",
    target_label="skeletal muscle",
    target_parent="muscle tissue",
    target_synonyms=[]  # 類義語なし
)
```

**出力**:
```
We have two entities from different ontologies.
The first one is "cardiac muscle", which falls under the category "muscle tissue".
The second one is "skeletal muscle", which falls under the category "muscle tissue".
Do they mean the same thing? Respond with "True" or "False".
```

---

## まとめ

PNLF^Sは以下の特徴により、オントロジーアライメントにおける最適なプロンプトテンプレートです：

1. **自然言語形式**: LLMの訓練データと整合
2. **類義語の明示**: 語彙的バリエーションへの対応
3. **適切なコンテキスト**: 階層情報の効果的な利用
4. **構造化出力**: 一貫性のある結果

**推奨事項**:
- オントロジーアライメントタスクには常にPNLF^Sを使用
- 類義語情報が利用可能な場合は必ず含める
- Gemini Flash 2.5との組み合わせで最良の結果

## 参考文献

- [instructions.md](../instructions.md) - PNLF^Sの元の仕様
- [論文] Large Language Models as Oracles for Ontology Alignment (仮)
