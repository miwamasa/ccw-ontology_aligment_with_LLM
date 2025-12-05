# システムアーキテクチャ解説

## 目次
1. [概要](#概要)
2. [アーキテクチャ図](#アーキテクチャ図)
3. [コンポーネント構成](#コンポーネント構成)
4. [データフロー](#データフロー)
5. [設計パターン](#設計パターン)
6. [拡張性と保守性](#拡張性と保守性)

---

## 概要

本システムは、オントロジーアライメントのためのLLMベースのオラクル診断エンジンです。モジュラー設計により、以下を実現しています：

- **複数LLMプロバイダー対応**: OpenAI、Anthropic、Google Geminiなど
- **プラグイン可能な設計**: 新しいプロバイダーやプロンプト形式を容易に追加
- **疎結合アーキテクチャ**: 各コンポーネントが独立して動作
- **テスタビリティ**: Mockプロバイダーによるテスト容易性

---

## アーキテクチャ図

### 全体構成図

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                    │
├─────────────────────────┬───────────────────────────────────┤
│   interactive.py        │         demo.py                   │
│   (Interactive CLI)     │         (Demo Script)             │
└────────────┬────────────┴───────────────┬───────────────────┘
             │                            │
             └────────────┬───────────────┘
                          │
            ┌─────────────▼─────────────────┐
            │      Application Layer        │
            │   (User-facing Programs)      │
            └─────────────┬─────────────────┘
                          │
        ┌─────────────────▼──────────────────────┐
        │         Core Library (src/)            │
        ├────────────────────────────────────────┤
        │                                        │
        │  ┌──────────────────────────────────┐ │
        │  │   OntologyAlignmentOracle        │ │
        │  │   (Orchestration Layer)          │ │
        │  └──────────┬───────────────────────┘ │
        │             │                          │
        │  ┌──────────▼──────────┐              │
        │  │  PNLFSPromptBuilder │              │
        │  │  (Prompt Generation)│              │
        │  └──────────┬──────────┘              │
        │             │                          │
        │  ┌──────────▼──────────────────────┐  │
        │  │    LLMProvider (Interface)      │  │
        │  │  ┌────────────────────────────┐ │  │
        │  │  │  OpenAI | Anthropic |      │ │  │
        │  │  │  Gemini | Mock             │ │  │
        │  │  └────────────────────────────┘ │  │
        │  └──────────┬──────────────────────┘  │
        │             │                          │
        │  ┌──────────▼──────────┐              │
        │  │   Data Models        │              │
        │  │   - OntologyEntity   │              │
        │  │   - EntityMapping    │              │
        │  │   - DiagnosticResult │              │
        │  └──────────────────────┘              │
        └────────────────────────────────────────┘
                          │
        ┌─────────────────▼─────────────────┐
        │   External Services Layer         │
        ├───────────────────────────────────┤
        │  OpenAI API │ Anthropic API │     │
        │  Google Gemini API                │
        └───────────────────────────────────┘
```

### レイヤー構造

```
┌────────────────────────────────────┐
│  Presentation Layer                │  ← interactive.py, demo.py
│  (CLI, User Interaction)           │
├────────────────────────────────────┤
│  Business Logic Layer              │  ← OntologyAlignmentOracle
│  (Orchestration, Evaluation)       │
├────────────────────────────────────┤
│  Service Layer                     │  ← LLMProvider (Abstract)
│  (LLM Integration)                 │
├────────────────────────────────────┤
│  Data Access Layer                 │  ← Prompt Builder, Models
│  (Prompt Generation, Data Models)  │
└────────────────────────────────────┘
```

---

## コンポーネント構成

### 1. データモデル層 (`src/models.py`)

オントロジーエンティティとマッピングを表現するデータ構造。

#### OntologyEntity
```python
@dataclass
class OntologyEntity:
    label: str
    parent_class: str
    synonyms: List[str]
    ontology_id: Optional[str]
```

**責務**:
- エンティティの基本情報を保持
- ラベル、親クラス、類義語の管理
- オントロジーID（オプション）

**設計のポイント**:
- Immutableなデータクラスとして設計
- シンプルで理解しやすいフィールド構成
- オプショナルフィールドはデフォルト値を提供

#### EntityMapping
```python
@dataclass
class EntityMapping:
    source_entity: OntologyEntity
    target_entity: OntologyEntity
    ground_truth: Optional[bool]
```

**責務**:
- 2つのエンティティ間のマッピングを表現
- 評価用のground truth（正解ラベル）を保持

#### DiagnosticResult
```python
@dataclass
class DiagnosticResult:
    mapping: EntityMapping
    prediction: bool
    confidence: Optional[float]
    raw_response: Optional[str]
```

**責務**:
- 診断結果を保持
- 予測値とground truthの比較機能を提供

### 2. プロンプトビルダー層 (`src/prompt_builder.py`)

PNLF^Sプロンプトを生成する。

#### PNLFSPromptBuilder

**責務**:
- PNLF^Sテンプレートに基づくプロンプト生成
- 類義語のフォーマット処理
- 2つのインターフェース提供：
  - `build_prompt(mapping)`: Mappingオブジェクトから生成
  - `build_prompt_from_entities(...)`: パラメータから直接生成

**設計のポイント**:
- ステートレスなクラスメソッド設計
- テンプレート文字列を定数として管理
- 柔軟な入力方式（オブジェクト vs パラメータ）

**主要メソッド**:

```python
@classmethod
def build_prompt(cls, mapping: EntityMapping) -> str:
    """EntityMappingからPNLF^Sプロンプトを生成"""

@classmethod
def build_prompt_from_entities(
    cls,
    source_label: str,
    source_parent: str,
    target_label: str,
    target_parent: str,
    target_synonyms: List[str] = None
) -> str:
    """パラメータから直接PNLF^Sプロンプトを生成"""
```

### 3. LLMプロバイダー層 (`src/oracle.py`)

LLMとの通信を抽象化。

#### LLMProvider (抽象基底クラス)

```python
class LLMProvider(ABC):
    @abstractmethod
    def diagnose(self, prompt: str, system_prompt: Optional[str] = None) -> bool:
        """プロンプトをLLMに送信しTrue/False応答を取得"""
        pass
```

**設計のポイント**:
- Strategy パターンの実装
- 異なるLLMプロバイダーを統一インターフェースで扱う
- 新しいプロバイダーの追加が容易

#### 具体的プロバイダー

```
LLMProvider (ABC)
    ├── OpenAIProvider
    ├── AnthropicProvider
    ├── GeminiProvider
    └── MockProvider
```

各プロバイダーの責務:
- 対応するLLM APIとの通信
- 構造化出力の処理
- エラーハンドリング
- レスポンスのパース（True/False抽出）

**実装例（GeminiProvider）**:

```python
class GeminiProvider(LLMProvider):
    def __init__(self, model: str = "gemini-2.0-flash-exp", api_key: Optional[str] = None):
        # 初期化処理

    def diagnose(self, prompt: str, system_prompt: Optional[str] = None) -> bool:
        # Gemini APIコール
        # 構造化出力（JSON）の処理
        # booleanを返す
```

### 4. オラクルエンジン層 (`src/oracle.py`)

全体のオーケストレーションを担当。

#### OntologyAlignmentOracle

**責務**:
- プロンプトビルダーとLLMプロバイダーの連携
- 単一および複数マッピングの診断
- パフォーマンス評価メトリクスの計算

**主要メソッド**:

```python
class OntologyAlignmentOracle:
    def __init__(
        self,
        provider: LLMProvider,
        system_prompt_type: Optional[str] = None,
        custom_system_prompt: Optional[str] = None
    ):
        """初期化"""

    def diagnose_mapping(self, mapping: EntityMapping) -> DiagnosticResult:
        """単一マッピングを診断"""

    def diagnose_batch(self, mappings: List[EntityMapping]) -> List[DiagnosticResult]:
        """複数マッピングをバッチ診断"""

    def evaluate_performance(self, results: List[DiagnosticResult]) -> Dict[str, float]:
        """パフォーマンスメトリクスを計算"""
```

**評価メトリクス**:
- Accuracy（精度）
- Precision（適合率）
- Recall（再現率）
- F1 Score
- **Youden's Index**（重要）

---

## データフロー

### 単一診断のフロー

```
1. User Input
   └─> EntityMapping (source, target)

2. OntologyAlignmentOracle.diagnose_mapping()
   ├─> PNLFSPromptBuilder.build_prompt()
   │   └─> PNLF^S formatted prompt
   │
   ├─> LLMProvider.diagnose()
   │   ├─> Format API request
   │   ├─> Call LLM API
   │   ├─> Parse response
   │   └─> Return bool
   │
   └─> Create DiagnosticResult
       └─> Return to user

3. Output
   └─> DiagnosticResult (prediction, is_correct, etc.)
```

### バッチ評価のフロー

```
1. User Input
   └─> List[EntityMapping] (with ground_truth)

2. OntologyAlignmentOracle.diagnose_batch()
   ├─> For each mapping:
   │   └─> diagnose_mapping()
   │
   └─> List[DiagnosticResult]

3. OntologyAlignmentOracle.evaluate_performance()
   ├─> Calculate confusion matrix (TP, TN, FP, FN)
   ├─> Calculate metrics
   │   ├─> Accuracy
   │   ├─> Precision
   │   ├─> Recall
   │   ├─> F1 Score
   │   └─> Youden's Index
   │
   └─> Return Dict[str, float]

4. Output
   └─> Performance metrics
```

### インタラクティブCLIのフロー

```
interactive.py
    │
    ├─> Load sample_mappings.json
    │   └─> Parse into List[EntityMapping]
    │
    ├─> Main Menu Loop
    │   ├─> 1. Preview Prompt
    │   │   └─> PNLFSPromptBuilder.build_prompt()
    │   │
    │   ├─> 2. Diagnose
    │   │   ├─> Select mapping
    │   │   ├─> Build prompt
    │   │   └─> oracle.diagnose_mapping()
    │   │
    │   ├─> 3. Batch Evaluation
    │   │   ├─> oracle.diagnose_batch()
    │   │   └─> oracle.evaluate_performance()
    │   │
    │   ├─> 4. Manual Input
    │   │   ├─> User inputs entity data
    │   │   ├─> Create EntityMapping
    │   │   └─> oracle.diagnose_mapping()
    │   │
    │   └─> 5. Configure Provider
    │       └─> Create new LLMProvider instance
    │
    └─> Exit
```

---

## 設計パターン

### 1. Strategy Pattern（戦略パターン）

**適用箇所**: LLMProvider

```python
# Interface
class LLMProvider(ABC):
    def diagnose(self, prompt: str, system_prompt: Optional[str] = None) -> bool:
        pass

# Concrete Strategies
class OpenAIProvider(LLMProvider): ...
class GeminiProvider(LLMProvider): ...
class MockProvider(LLMProvider): ...

# Context
class OntologyAlignmentOracle:
    def __init__(self, provider: LLMProvider):
        self.provider = provider  # Strategy injection
```

**利点**:
- 実行時にプロバイダーを切り替え可能
- 新しいLLMプロバイダーの追加が容易
- テスト時にMockProviderを使用可能

### 2. Builder Pattern（ビルダーパターン）

**適用箇所**: PNLFSPromptBuilder

```python
class PNLFSPromptBuilder:
    @classmethod
    def build_prompt(cls, mapping: EntityMapping) -> str:
        source_desc = cls.SOURCE_TEMPLATE.format(...)
        target_desc = cls.TARGET_TEMPLATE.format(...)
        synonyms_clause = cls._format_synonyms_clause(...)
        return f"{cls.FRAME_INTRO}\n{source_desc}\n{target_desc}\n{cls.QUESTION}"
```

**利点**:
- 複雑なプロンプト構築プロセスを段階的に実行
- テンプレート部品の再利用
- 一貫性のあるプロンプト生成

### 3. Template Method Pattern（テンプレートメソッドパターン）

**適用箇所**: LLMProvider.diagnose()

各プロバイダーは共通のフローを持ちながら、詳細が異なる：

```python
def diagnose(self, prompt: str, system_prompt: Optional[str] = None) -> bool:
    # 1. プロンプトの準備（各プロバイダーで異なる）
    # 2. APIコール（各プロバイダーで異なる）
    # 3. レスポンスのパース（各プロバイダーで異なる）
    # 4. booleanを返す（共通）
```

### 4. Facade Pattern（ファサードパターン）

**適用箇所**: OntologyAlignmentOracle

```python
# 複雑なサブシステムを単純なインターフェースで提供
oracle = OntologyAlignmentOracle(provider)
result = oracle.diagnose_mapping(mapping)  # シンプルなAPI
```

ユーザーは以下を意識する必要がない：
- プロンプトの構築方法
- LLM APIの詳細
- レスポンスのパース

---

## 拡張性と保守性

### 拡張ポイント1: 新しいLLMプロバイダーの追加

```python
# 新しいプロバイダーの追加は簡単
class NewLLMProvider(LLMProvider):
    def __init__(self, model: str, api_key: str):
        # 初期化

    def diagnose(self, prompt: str, system_prompt: Optional[str] = None) -> bool:
        # 実装
        return True  # or False
```

**必要な手順**:
1. `LLMProvider`を継承
2. `diagnose()`メソッドを実装
3. OracleでインスタンスをDI

### 拡張ポイント2: カスタムプロンプトテンプレート

```python
class CustomPromptBuilder:
    @classmethod
    def build_prompt(cls, mapping: EntityMapping) -> str:
        # カスタムテンプレート実装
        return custom_prompt
```

### 拡張ポイント3: 評価メトリクスの追加

```python
def evaluate_performance(self, results: List[DiagnosticResult]) -> Dict[str, float]:
    # 既存のメトリクス計算
    ...

    # 新しいメトリクスを追加
    matthews_correlation = self._calculate_mcc(tp, tn, fp, fn)

    return {
        ...
        "mcc": matthews_correlation
    }
```

### 保守性の特徴

#### 1. 疎結合
- 各コンポーネントが独立
- インターフェースを通じた通信
- 依存性注入（DI）の活用

#### 2. 単一責任原則
- 各クラスが明確な責務を持つ
- コードの理解が容易
- テストが簡単

#### 3. オープン・クローズド原則
- 拡張に対して開いている（新しいプロバイダー追加など）
- 修正に対して閉じている（既存コードの変更不要）

#### 4. テスタビリティ
```python
# Mockプロバイダーでテスト
def test_oracle():
    mock_provider = MockProvider(default_response=True)
    oracle = OntologyAlignmentOracle(mock_provider)
    result = oracle.diagnose_mapping(mapping)
    assert result.prediction == True
```

---

## まとめ

本システムのアーキテクチャは以下の特徴を持ちます：

1. **モジュラー設計**: 各コンポーネントが独立して動作
2. **拡張性**: 新しいLLMプロバイダーやメトリクスの追加が容易
3. **保守性**: クリーンアーキテクチャの原則に従った設計
4. **テスタビリティ**: Mockプロバイダーによるテストが容易
5. **再利用性**: 各コンポーネントが独立して再利用可能

この設計により、研究用途から本番環境まで、幅広いユースケースに対応できます。
