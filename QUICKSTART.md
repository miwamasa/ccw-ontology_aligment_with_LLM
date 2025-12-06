# Quick Start Guide

PNLF^S実装を5分で試す方法

## 1. デモを実行（API不要）

```bash
python demo.py
```

このデモでは：
- PNLF^Sプロンプトの生成方法を表示
- Mockプロバイダーを使用してAPI呼び出しなしで動作確認
- バッチ評価とメトリクス計算のデモ

## 2. インタラクティブ環境を起動

```bash
python interactive.py
```

### 推奨の操作手順：

1. **メニューオプション 7**: PNLF^Sについて学ぶ
2. **メニューオプション 6**: サンプルマッピングを閲覧
3. **メニューオプション 1**: プロンプトをプレビュー
4. **メニューオプション 5**: LLMプロバイダーを設定（Mock、OpenAI、Anthropic、Gemini）
5. **メニューオプション 2**: 診断を実行
6. **メニューオプション 3**: バッチ評価を実行

## 3. プログラムで使用

### 最小限の例（Mockプロバイダー）

```python
from src.models import OntologyEntity, EntityMapping
from src.oracle import OntologyAlignmentOracle, MockProvider

# エンティティを作成
source = OntologyEntity("neuron", "nerve cell")
target = OntologyEntity("nerve cell", "nervous system cell", ["neuron"])

# マッピングを作成
mapping = EntityMapping(source, target)

# オラクルを初期化
oracle = OntologyAlignmentOracle(MockProvider())

# 診断
result = oracle.diagnose_mapping(mapping)
print(f"Prediction: {result.prediction}")
```

### 実際のLLMを使用（Gemini推奨）

```python
from src.oracle import GeminiProvider

# 環境変数 GOOGLE_API_KEY を設定してから実行
provider = GeminiProvider(model="gemini-2.0-flash-exp")
oracle = OntologyAlignmentOracle(provider, system_prompt_type="biomedical_specialist")

result = oracle.diagnose_mapping(mapping)
```

## 4. サンプルマッピングを使用

### 生体医学オントロジー

```python
import json

# 生体医学サンプルマッピングをロード
with open('examples/sample_mappings.json', 'r') as f:
    samples = json.load(f)

# 最初のサンプルを使用
sample = samples[0]
print(f"Testing: {sample['name']}")
print(f"Description: {sample['description']}")
```

### 製造業・IoTオントロジー

```python
# 製造業・IoTサンプルマッピングをロード
with open('examples/sample_mappings_manufacturing_iot.json', 'r') as f:
    samples = json.load(f)

# 例：温度センサーのマッピング
sample = samples[0]  # Temperature Sensor vs Thermometer
print(f"Testing: {sample['name']}")
```

### インタラクティブCLIでデータセット切替

```bash
python interactive.py
# メニューから 7 (Switch Dataset) を選択
# 1. Biomedical
# 2. Manufacturing & IoT
```

## よくある使用ケース

### ケース1: プロンプトだけを生成したい

```python
from src.prompt_builder import PNLFSPromptBuilder

prompt = PNLFSPromptBuilder.build_prompt_from_entities(
    source_label="kidney",
    source_parent="excretory organ",
    target_label="renal organ",
    target_parent="urinary system organ",
    target_synonyms=["kidney", "nephros"]
)

print(prompt)
```

### ケース2: カスタムマッピングを診断したい

```bash
python interactive.py
# メニューから 4 (Manual Input Mode) を選択
```

### ケース3: パフォーマンスを評価したい

```python
from src.oracle import OntologyAlignmentOracle, MockProvider

# ground_truth付きのマッピングを準備
mappings = [...]  # ground_truthを含むEntityMappingのリスト

oracle = OntologyAlignmentOracle(MockProvider())
results = oracle.diagnose_batch(mappings)

# メトリクスを計算
metrics = oracle.evaluate_performance(results)
print(f"Youden's Index: {metrics['youden_index']:.3f}")
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")
```

## API キーの設定

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
```

### Anthropic Claude

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Google Gemini（推奨）

```bash
export GOOGLE_API_KEY="AIza..."
```

## トラブルシューティング

### ImportError: No module named 'openai'

```bash
pip install openai
```

### API key not provided

環境変数が正しく設定されているか確認：

```bash
echo $GOOGLE_API_KEY
```

### サンプルマッピングが見つからない

カレントディレクトリがプロジェクトルートであることを確認：

```bash
pwd
ls examples/sample_mappings.json
```

## 次のステップ

- `instructions.md` を読んで仕様の詳細を理解
- `README.md` で完全なドキュメントを確認
- 自分のオントロジーデータでテスト
- バッチ評価でパフォーマンスを測定

## サポート

質問や問題があれば、GitHubのIssuesで報告してください。
