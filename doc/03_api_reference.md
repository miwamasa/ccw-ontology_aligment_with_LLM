# APIリファレンス

## 目次
1. [データモデル](#データモデル)
2. [プロンプトビルダー](#プロンプトビルダー)
3. [LLMプロバイダー](#llmプロバイダー)
4. [オラクルエンジン](#オラクルエンジン)

---

## データモデル

### OntologyEntity

オントロジーエンティティを表現するデータクラス。

```python
from src.models import OntologyEntity
```

#### クラス定義

```python
@dataclass
class OntologyEntity:
    label: str
    parent_class: str
    synonyms: List[str] = field(default_factory=list)
    ontology_id: Optional[str] = None
```

#### フィールド

| フィールド名 | 型 | 必須 | 説明 |
|------------|-----|------|------|
| `label` | `str` | ✅ | エンティティのラベル（例: "alveolus epithelium"） |
| `parent_class` | `str` | ✅ | 親クラス（例: "lung epithelium"） |
| `synonyms` | `List[str]` | ❌ | 類義語のリスト（デフォルト: 空リスト） |
| `ontology_id` | `Optional[str]` | ❌ | オントロジーID（例: "MA_0001771"） |

#### メソッド

##### `__str__()`

エンティティの文字列表現を返します。

```python
entity = OntologyEntity("neuron", "nerve cell")
print(entity)  # "neuron (parent: nerve cell)"
```

#### 使用例

```python
# 基本的な使用
source = OntologyEntity(
    label="hepatocyte",
    parent_class="epithelial cell"
)

# 類義語付き
target = OntologyEntity(
    label="liver cell",
    parent_class="organ cell",
    synonyms=["hepatocyte", "hepatic parenchymal cell"]
)

# オントロジーID付き
entity = OntologyEntity(
    label="alveolus epithelium",
    parent_class="lung epithelium",
    synonyms=["alveolar epithelium"],
    ontology_id="MA_0001771"
)
```

---

### EntityMapping

2つのエンティティ間のマッピングを表現。

```python
from src.models import EntityMapping
```

#### クラス定義

```python
@dataclass
class EntityMapping:
    source_entity: OntologyEntity
    target_entity: OntologyEntity
    ground_truth: Optional[bool] = None
```

#### フィールド

| フィールド名 | 型 | 必須 | 説明 |
|------------|-----|------|------|
| `source_entity` | `OntologyEntity` | ✅ | ソースエンティティ |
| `target_entity` | `OntologyEntity` | ✅ | ターゲットエンティティ |
| `ground_truth` | `Optional[bool]` | ❌ | 正解ラベル（評価用） |

#### メソッド

##### `__str__()`

マッピングの文字列表現を返します。

```python
mapping = EntityMapping(source, target)
print(mapping)  # "<hepatocyte, liver cell>"
```

#### 使用例

```python
# 基本的なマッピング
mapping = EntityMapping(
    source_entity=OntologyEntity("neuron", "nerve cell"),
    target_entity=OntologyEntity("nerve cell", "nervous system cell")
)

# Ground truth付き（評価用）
mapping_with_truth = EntityMapping(
    source_entity=source,
    target_entity=target,
    ground_truth=True  # このマッピングは正しい
)
```

---

### DiagnosticResult

診断結果を保持するデータクラス。

```python
from src.models import DiagnosticResult
```

#### クラス定義

```python
@dataclass
class DiagnosticResult:
    mapping: EntityMapping
    prediction: bool
    confidence: Optional[float] = None
    raw_response: Optional[str] = None
```

#### フィールド

| フィールド名 | 型 | 必須 | 説明 |
|------------|-----|------|------|
| `mapping` | `EntityMapping` | ✅ | 診断対象のマッピング |
| `prediction` | `bool` | ✅ | LLMの予測（True/False） |
| `confidence` | `Optional[float]` | ❌ | 信頼度スコア（将来の拡張用） |
| `raw_response` | `Optional[str]` | ❌ | LLMの生レスポンス |

#### プロパティ

##### `is_correct`

予測がground truthと一致するかを返します。

**戻り値**: `Optional[bool]`
- `True`: 予測が正解
- `False`: 予測が不正解
- `None`: ground truthが未設定

```python
result = DiagnosticResult(
    mapping=mapping_with_truth,
    prediction=True
)
print(result.is_correct)  # True (予測=True, ground_truth=True)
```

---

## プロンプトビルダー

### PNLFSPromptBuilder

PNLF^Sプロンプトを生成するビルダークラス。

```python
from src.prompt_builder import PNLFSPromptBuilder
```

#### クラス定数

| 定数名 | 値 | 説明 |
|--------|-----|------|
| `FRAME_INTRO` | `"We have two entities..."` | イントロダクション文 |
| `SOURCE_TEMPLATE` | `'The first one is...'` | ソースエンティティテンプレート |
| `TARGET_TEMPLATE` | `'The second one is...'` | ターゲットエンティティテンプレート |
| `QUESTION` | `'Do they mean the same thing?...'` | 質問文 |

#### メソッド

##### `build_prompt(mapping: EntityMapping) -> str`

EntityMappingからPNLF^Sプロンプトを生成します。

**パラメータ**:
- `mapping` (`EntityMapping`): マッピングオブジェクト

**戻り値**: `str` - 生成されたプロンプト

**使用例**:

```python
mapping = EntityMapping(
    source_entity=OntologyEntity("neuron", "nerve cell"),
    target_entity=OntologyEntity(
        "nerve cell",
        "nervous system cell",
        synonyms=["neuron", "neurocyte"]
    )
)

prompt = PNLFSPromptBuilder.build_prompt(mapping)
print(prompt)
```

**出力**:
```
We have two entities from different ontologies.
The first one is "neuron", which falls under the category "nerve cell".
The second one is "nerve cell", also known as "neuron", "neurocyte", which falls under the category "nervous system cell".
Do they mean the same thing? Respond with "True" or "False".
```

##### `build_prompt_from_entities(...) -> str`

パラメータから直接PNLF^Sプロンプトを生成します。

**パラメータ**:
- `source_label` (`str`): ソースエンティティのラベル
- `source_parent` (`str`): ソースの親クラス
- `target_label` (`str`): ターゲットエンティティのラベル
- `target_parent` (`str`): ターゲットの親クラス
- `target_synonyms` (`List[str]`, optional): ターゲットの類義語

**戻り値**: `str` - 生成されたプロンプト

**使用例**:

```python
prompt = PNLFSPromptBuilder.build_prompt_from_entities(
    source_label="cardiac muscle",
    source_parent="muscle tissue",
    target_label="heart muscle",
    target_parent="muscle tissue",
    target_synonyms=["myocardium", "cardiac muscle"]
)
```

---

### システムプロンプト

事前定義されたシステムプロンプトのコレクション。

```python
from src.prompt_builder import SYSTEM_PROMPTS
```

#### 利用可能なシステムプロンプト

| キー | 説明 |
|------|------|
| `"biomedical_specialist"` | 生体医学オントロジー専門家としてLLMを位置づける |
| `"explicit_semantics"` | 明示的なセマンティクスに焦点を当てる |
| `"combined"` | 上記2つを組み合わせたもの |

**使用例**:

```python
system_prompt = SYSTEM_PROMPTS["biomedical_specialist"]
oracle = OntologyAlignmentOracle(
    provider=provider,
    custom_system_prompt=system_prompt
)
```

---

## LLMプロバイダー

### LLMProvider (抽象基底クラス)

すべてのLLMプロバイダーの基底クラス。

```python
from src.oracle import LLMProvider
```

#### 抽象メソッド

##### `diagnose(prompt: str, system_prompt: Optional[str] = None) -> bool`

プロンプトをLLMに送信し、True/False応答を取得します。

**パラメータ**:
- `prompt` (`str`): PNLF^Sプロンプト
- `system_prompt` (`Optional[str]`): システムプロンプト

**戻り値**: `bool` - LLMの診断結果

---

### OpenAIProvider

OpenAI GPTモデル用のプロバイダー。

```python
from src.oracle import OpenAIProvider
```

#### コンストラクタ

```python
OpenAIProvider(model: str = "gpt-4o-mini", api_key: Optional[str] = None)
```

**パラメータ**:
- `model` (`str`): モデル名（デフォルト: `"gpt-4o-mini"`）
  - 利用可能: `"gpt-4o"`, `"gpt-4o-mini"`, etc.
- `api_key` (`Optional[str]`): OpenAI APIキー
  - 未指定の場合、環境変数 `OPENAI_API_KEY` を使用

**使用例**:

```python
# 環境変数からAPIキーを読み込み
provider = OpenAIProvider(model="gpt-4o-mini")

# APIキーを直接指定
provider = OpenAIProvider(
    model="gpt-4o",
    api_key="sk-..."
)
```

**構造化出力**: JSON Schemaを使用してTrue/Falseを強制

---

### AnthropicProvider

Anthropic Claudeモデル用のプロバイダー。

```python
from src.oracle import AnthropicProvider
```

#### コンストラクタ

```python
AnthropicProvider(model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None)
```

**パラメータ**:
- `model` (`str`): モデル名（デフォルト: `"claude-3-5-sonnet-20241022"`）
- `api_key` (`Optional[str]`): Anthropic APIキー
  - 未指定の場合、環境変数 `ANTHROPIC_API_KEY` を使用

**使用例**:

```python
provider = AnthropicProvider()

# カスタムモデル
provider = AnthropicProvider(model="claude-3-opus-20240229")
```

---

### GeminiProvider (推奨)

Google Geminiモデル用のプロバイダー。**PNLF^Sで最高のパフォーマンス**を達成。

```python
from src.oracle import GeminiProvider
```

#### コンストラクタ

```python
GeminiProvider(model: str = "gemini-2.0-flash-exp", api_key: Optional[str] = None)
```

**パラメータ**:
- `model` (`str`): モデル名（デフォルト: `"gemini-2.0-flash-exp"`）
  - 推奨: `"gemini-2.0-flash-exp"` (Flash 2.5相当)
  - 代替: `"gemini-1.5-flash"`, `"gemini-1.5-pro"`
- `api_key` (`Optional[str]`): Google APIキー
  - 未指定の場合、環境変数 `GOOGLE_API_KEY` を使用

**使用例**:

```python
# 推奨設定
provider = GeminiProvider(model="gemini-2.0-flash-exp")

oracle = OntologyAlignmentOracle(
    provider=provider,
    system_prompt_type="biomedical_specialist"
)
```

**構造化出力**: Response Schema機能を使用

---

### MockProvider

テスト用のモックプロバイダー（API呼び出しなし）。

```python
from src.oracle import MockProvider
```

#### コンストラクタ

```python
MockProvider(default_response: bool = True)
```

**パラメータ**:
- `default_response` (`bool`): 常に返す応答（デフォルト: `True`）

**使用例**:

```python
# 常にTrueを返すモック
mock = MockProvider(default_response=True)

# テストで使用
oracle = OntologyAlignmentOracle(mock)
result = oracle.diagnose_mapping(mapping)
# result.prediction は常に True
```

---

## オラクルエンジン

### OntologyAlignmentOracle

LLMオラクルのメインエンジン。

```python
from src.oracle import OntologyAlignmentOracle
```

#### コンストラクタ

```python
OntologyAlignmentOracle(
    provider: LLMProvider,
    system_prompt_type: Optional[str] = None,
    custom_system_prompt: Optional[str] = None
)
```

**パラメータ**:
- `provider` (`LLMProvider`): 使用するLLMプロバイダー
- `system_prompt_type` (`Optional[str]`): 事前定義されたシステムプロンプトのタイプ
  - `"biomedical_specialist"`
  - `"explicit_semantics"`
  - `"combined"`
  - `None` (システムプロンプトなし)
- `custom_system_prompt` (`Optional[str]`): カスタムシステムプロンプト
  - 指定すると `system_prompt_type` より優先

**使用例**:

```python
# 基本
oracle = OntologyAlignmentOracle(GeminiProvider())

# システムプロンプト付き
oracle = OntologyAlignmentOracle(
    provider=GeminiProvider(),
    system_prompt_type="biomedical_specialist"
)

# カスタムシステムプロンプト
oracle = OntologyAlignmentOracle(
    provider=GeminiProvider(),
    custom_system_prompt="You are an expert in biomedical ontologies..."
)
```

#### メソッド

##### `diagnose_mapping(mapping: EntityMapping) -> DiagnosticResult`

単一のマッピングを診断します。

**パラメータ**:
- `mapping` (`EntityMapping`): 診断するマッピング

**戻り値**: `DiagnosticResult` - 診断結果

**使用例**:

```python
mapping = EntityMapping(
    source_entity=OntologyEntity("neuron", "nerve cell"),
    target_entity=OntologyEntity("nerve cell", "nervous system cell", ["neuron"])
)

result = oracle.diagnose_mapping(mapping)
print(f"Prediction: {result.prediction}")  # True or False
```

##### `diagnose_batch(mappings: List[EntityMapping]) -> List[DiagnosticResult]`

複数のマッピングをバッチ診断します。

**パラメータ**:
- `mappings` (`List[EntityMapping]`): 診断するマッピングのリスト

**戻り値**: `List[DiagnosticResult]` - 診断結果のリスト

**使用例**:

```python
mappings = [mapping1, mapping2, mapping3]
results = oracle.diagnose_batch(mappings)

for result in results:
    print(f"{result.mapping} -> {result.prediction}")
```

##### `evaluate_performance(results: List[DiagnosticResult]) -> Dict[str, float]`

診断結果からパフォーマンスメトリクスを計算します。

**パラメータ**:
- `results` (`List[DiagnosticResult]`): ground truth付きの診断結果

**戻り値**: `Dict[str, float]` - メトリクスの辞書

**メトリクス**:
- `accuracy`: 精度
- `precision`: 適合率
- `recall`: 再現率
- `f1_score`: F1スコア
- `youden_index`: Youden's Index（重要）
- `true_positives`: 真陽性の数
- `true_negatives`: 真陰性の数
- `false_positives`: 偽陽性の数
- `false_negatives`: 偽陰性の数
- `total`: 総数

**使用例**:

```python
# ground truth付きのマッピングを準備
mappings = [
    EntityMapping(source1, target1, ground_truth=True),
    EntityMapping(source2, target2, ground_truth=False),
    # ...
]

results = oracle.diagnose_batch(mappings)
metrics = oracle.evaluate_performance(results)

print(f"Youden's Index: {metrics['youden_index']:.3f}")
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")
```

---

## 完全な使用例

```python
from src.models import OntologyEntity, EntityMapping
from src.oracle import OntologyAlignmentOracle, GeminiProvider
from src.prompt_builder import PNLFSPromptBuilder

# 1. エンティティを作成
source = OntologyEntity(
    label="alveolus epithelium",
    parent_class="lung epithelium"
)

target = OntologyEntity(
    label="Alveolar_Epithelium",
    parent_class="Epithelium",
    synonyms=["Lung Alveolar Epithelia", "Alveolar Epithelium"]
)

# 2. マッピングを作成
mapping = EntityMapping(
    source_entity=source,
    target_entity=target,
    ground_truth=True  # 評価用
)

# 3. プロンプトをプレビュー（オプション）
prompt = PNLFSPromptBuilder.build_prompt(mapping)
print("Generated prompt:")
print(prompt)

# 4. オラクルを初期化
provider = GeminiProvider(model="gemini-2.0-flash-exp")
oracle = OntologyAlignmentOracle(
    provider=provider,
    system_prompt_type="biomedical_specialist"
)

# 5. 診断を実行
result = oracle.diagnose_mapping(mapping)

# 6. 結果を確認
print(f"Prediction: {result.prediction}")
print(f"Ground Truth: {mapping.ground_truth}")
print(f"Correct: {result.is_correct}")

# 7. バッチ評価
mappings = [mapping, ...]  # 複数のマッピング
results = oracle.diagnose_batch(mappings)
metrics = oracle.evaluate_performance(results)

print(f"Youden's Index: {metrics['youden_index']:.3f}")
```

---

## エラーハンドリング

### よくあるエラー

#### `ValueError: API key not provided`

**原因**: APIキーが設定されていない

**解決策**:
```bash
export GOOGLE_API_KEY="your-api-key"
```

または

```python
provider = GeminiProvider(api_key="your-api-key")
```

#### `ImportError: No module named 'openai'`

**原因**: 必要なパッケージがインストールされていない

**解決策**:
```bash
pip install openai  # OpenAI用
pip install anthropic  # Anthropic用
pip install google-generativeai  # Gemini用
```

#### `ValueError: Unexpected response from LLM`

**原因**: LLMがTrue/False以外の応答を返した

**解決策**:
- 構造化出力をサポートするモデルを使用
- システムプロンプトで応答形式を明確化
- Mockプロバイダーでテスト

---

## まとめ

このAPIリファレンスは、LLM Oracle for Ontology Alignmentの全モジュールとクラスを網羅しています。

**次のステップ**:
- [実装ガイド](04_implementation_guide.md)でカスタマイズ方法を学習
- [ベストプラクティス](05_best_practices.md)で効果的な使い方を習得
