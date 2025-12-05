# 実装ガイド

## 目次
1. [カスタムLLMプロバイダーの実装](#カスタムllmプロバイダーの実装)
2. [カスタムプロンプトテンプレートの作成](#カスタムプロンプトテンプレートの作成)
3. [評価メトリクスの拡張](#評価メトリクスの拡張)
4. [バッチ処理の最適化](#バッチ処理の最適化)
5. [実装例集](#実装例集)

---

## カスタムLLMプロバイダーの実装

### 基本実装

新しいLLMプロバイダーを追加するには、`LLMProvider`抽象基底クラスを継承します。

```python
from src.oracle import LLMProvider
from typing import Optional

class CustomLLMProvider(LLMProvider):
    """カスタムLLMプロバイダーの例"""

    def __init__(self, model: str, api_key: Optional[str] = None):
        """
        初期化

        Args:
            model: モデル名
            api_key: APIキー（オプション）
        """
        self.model = model
        self.api_key = api_key or os.getenv("CUSTOM_API_KEY")

        if not self.api_key:
            raise ValueError("API key not provided")

        # クライアントの初期化
        self.client = CustomAPIClient(api_key=self.api_key)

    def diagnose(self, prompt: str, system_prompt: Optional[str] = None) -> bool:
        """
        プロンプトを送信してTrue/False応答を取得

        Args:
            prompt: PNLF^Sプロンプト
            system_prompt: システムプロンプト（オプション）

        Returns:
            bool: 診断結果
        """
        # 1. リクエストを構築
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 2. APIコール
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0  # 決定論的な応答
        )

        # 3. レスポンスをパース
        result_text = response.choices[0].message.content.strip().lower()

        # 4. True/Falseに変換
        if "true" in result_text:
            return True
        elif "false" in result_text:
            return False
        else:
            raise ValueError(f"Unexpected response: {result_text}")
```

### 構造化出力の実装

より確実にTrue/Falseを取得するには、構造化出力を使用します。

#### OpenAI形式（JSON Schema）

```python
def diagnose(self, prompt: str, system_prompt: Optional[str] = None) -> bool:
    response = self.client.chat.completions.create(
        model=self.model,
        messages=[...],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "diagnostic_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "result": {
                            "type": "boolean",
                            "description": "True if entities mean the same, False otherwise"
                        }
                    },
                    "required": ["result"],
                    "additionalProperties": False
                }
            }
        }
    )

    result_json = json.loads(response.choices[0].message.content)
    return result_json["result"]
```

#### Gemini形式（Response Schema）

```python
def diagnose(self, prompt: str, system_prompt: Optional[str] = None) -> bool:
    model = genai.GenerativeModel(
        model_name=self.model,
        generation_config={
            "temperature": 0.0,
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "object",
                "properties": {
                    "result": {"type": "boolean"}
                },
                "required": ["result"]
            }
        }
    )

    response = model.generate_content(prompt)
    result_json = json.loads(response.text)
    return result_json["result"]
```

### エラーハンドリング

ロバストなプロバイダーにはエラーハンドリングが必要です。

```python
import time
from typing import Optional

class RobustLLMProvider(LLMProvider):
    def __init__(self, model: str, max_retries: int = 3):
        self.model = model
        self.max_retries = max_retries

    def diagnose(self, prompt: str, system_prompt: Optional[str] = None) -> bool:
        """リトライロジック付き診断"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # APIコール
                response = self._call_api(prompt, system_prompt)
                return self._parse_response(response)

            except RateLimitError:
                # レート制限：指数バックオフで待機
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                last_error = "Rate limit exceeded"

            except APIError as e:
                # API エラー
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    time.sleep(1)

            except Exception as e:
                # その他のエラー
                raise

        raise Exception(f"Failed after {self.max_retries} attempts: {last_error}")
```

---

## カスタムプロンプトテンプレートの作成

### 基本テンプレート

PNLF^S以外のプロンプト形式を試すこともできます。

```python
class CustomPromptBuilder:
    """カスタムプロンプトビルダー"""

    @classmethod
    def build_prompt(cls, mapping: EntityMapping) -> str:
        """カスタム形式のプロンプトを生成"""
        source = mapping.source_entity
        target = mapping.target_entity

        # カスタムテンプレート
        template = """
Compare these two ontology entities:

Entity A:
- Name: {source_label}
- Category: {source_parent}

Entity B:
- Name: {target_label}
- Category: {target_parent}
- Synonyms: {target_synonyms}

Question: Are these entities semantically equivalent?
Answer with True or False only.
        """

        synonyms_str = ", ".join(target.synonyms) if target.synonyms else "None"

        return template.format(
            source_label=source.label,
            source_parent=source.parent_class,
            target_label=target.label,
            target_parent=target.parent_class,
            target_synonyms=synonyms_str
        ).strip()
```

### プロンプトビルダーをOracleで使用

```python
class CustomOracle(OntologyAlignmentOracle):
    """カスタムプロンプトビルダーを使用するOracle"""

    def __init__(self, provider: LLMProvider, prompt_builder=None):
        super().__init__(provider)
        self.prompt_builder = prompt_builder or CustomPromptBuilder()

    def diagnose_mapping(self, mapping: EntityMapping) -> DiagnosticResult:
        # カスタムプロンプトビルダーを使用
        prompt = self.prompt_builder.build_prompt(mapping)

        prediction = self.provider.diagnose(prompt, self.system_prompt)

        return DiagnosticResult(
            mapping=mapping,
            prediction=prediction,
            raw_response=str(prediction)
        )
```

---

## 評価メトリクスの拡張

### カスタムメトリクスの追加

既存のメトリクスに加えて、独自のメトリクスを計算できます。

```python
class ExtendedOracle(OntologyAlignmentOracle):
    """拡張メトリクス付きOracle"""

    def evaluate_performance(self, results: List[DiagnosticResult]) -> Dict[str, float]:
        # 基本メトリクスを取得
        metrics = super().evaluate_performance(results)

        if not metrics:
            return {}

        # 混同行列の値を取得
        tp = metrics['true_positives']
        tn = metrics['true_negatives']
        fp = metrics['false_positives']
        fn = metrics['false_negatives']

        # Matthews相関係数（MCC）を追加
        mcc = self._calculate_mcc(tp, tn, fp, fn)
        metrics['matthews_correlation'] = mcc

        # Cohen's Kappaを追加
        kappa = self._calculate_kappa(tp, tn, fp, fn)
        metrics['cohen_kappa'] = kappa

        return metrics

    def _calculate_mcc(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """Matthews相関係数を計算"""
        numerator = (tp * tn) - (fp * fn)
        denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _calculate_kappa(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """Cohen's Kappaを計算"""
        total = tp + tn + fp + fn

        if total == 0:
            return 0.0

        # 観測一致率
        po = (tp + tn) / total

        # 期待一致率
        pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (total ** 2)

        if pe == 1:
            return 0.0

        return (po - pe) / (1 - pe)
```

### Per-Class メトリクス

クラス別のメトリクスを計算する場合：

```python
def evaluate_by_category(
    self,
    results: List[DiagnosticResult]
) -> Dict[str, Dict[str, float]]:
    """カテゴリ別にメトリクスを計算"""

    # カテゴリごとにグループ化
    by_category = {}
    for result in results:
        category = result.mapping.source_entity.parent_class
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(result)

    # カテゴリごとにメトリクスを計算
    category_metrics = {}
    for category, cat_results in by_category.items():
        category_metrics[category] = self.evaluate_performance(cat_results)

    return category_metrics
```

---

## バッチ処理の最適化

### 並列処理

大量のマッピングを効率的に処理するには並列処理が有効です。

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

class ParallelOracle(OntologyAlignmentOracle):
    """並列処理対応Oracle"""

    def __init__(self, provider: LLMProvider, max_workers: int = 5):
        super().__init__(provider)
        self.max_workers = max_workers

    def diagnose_batch(
        self,
        mappings: List[EntityMapping]
    ) -> List[DiagnosticResult]:
        """並列バッチ診断"""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 全タスクを投入
            future_to_mapping = {
                executor.submit(self.diagnose_mapping, mapping): mapping
                for mapping in mappings
            }

            # 完了順に結果を取得
            for future in as_completed(future_to_mapping):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    mapping = future_to_mapping[future]
                    print(f"Error diagnosing {mapping}: {e}")

        return results
```

### プログレスバー付き処理

大量処理の進捗を可視化：

```python
from tqdm import tqdm

class ProgressOracle(OntologyAlignmentOracle):
    """プログレスバー付きOracle"""

    def diagnose_batch(
        self,
        mappings: List[EntityMapping],
        show_progress: bool = True
    ) -> List[DiagnosticResult]:
        """プログレスバー付きバッチ診断"""
        results = []

        iterator = tqdm(mappings, desc="Diagnosing") if show_progress else mappings

        for mapping in iterator:
            try:
                result = self.diagnose_mapping(mapping)
                results.append(result)
            except Exception as e:
                print(f"Error: {e}")
                # エラー時はスキップまたはデフォルト値を使用

        return results
```

### キャッシング

同じマッピングの再診断を避ける：

```python
from functools import lru_cache
import hashlib

class CachedOracle(OntologyAlignmentOracle):
    """キャッシング対応Oracle"""

    def __init__(self, provider: LLMProvider):
        super().__init__(provider)
        self._cache = {}

    def _get_mapping_hash(self, mapping: EntityMapping) -> str:
        """マッピングのハッシュを生成"""
        key = f"{mapping.source_entity.label}|{mapping.source_entity.parent_class}|"
        key += f"{mapping.target_entity.label}|{mapping.target_entity.parent_class}|"
        key += "|".join(mapping.target_entity.synonyms)
        return hashlib.md5(key.encode()).hexdigest()

    def diagnose_mapping(self, mapping: EntityMapping) -> DiagnosticResult:
        """キャッシュを使用した診断"""
        # キャッシュを確認
        mapping_hash = self._get_mapping_hash(mapping)

        if mapping_hash in self._cache:
            print(f"Cache hit for {mapping}")
            return self._cache[mapping_hash]

        # キャッシュミス：診断を実行
        result = super().diagnose_mapping(mapping)

        # キャッシュに保存
        self._cache[mapping_hash] = result

        return result

    def clear_cache(self):
        """キャッシュをクリア"""
        self._cache.clear()
```

---

## 実装例集

### 例1: コスト追跡Oracle

API呼び出しのコストを追跡：

```python
class CostTrackingOracle(OntologyAlignmentOracle):
    """コスト追跡Oracle"""

    # モデルごとのトークン単価（例）
    COSTS_PER_1K_TOKENS = {
        "gpt-4o": {"input": 0.0025, "output": 0.010},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},  # 無料
    }

    def __init__(self, provider: LLMProvider, model_name: str):
        super().__init__(provider)
        self.model_name = model_name
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def diagnose_mapping(self, mapping: EntityMapping) -> DiagnosticResult:
        # プロンプトのトークン数を推定
        prompt = self.prompt_builder.build_prompt(mapping)
        estimated_input_tokens = len(prompt.split()) * 1.3  # 概算

        result = super().diagnose_mapping(mapping)

        # トークン数を記録
        self.total_input_tokens += estimated_input_tokens
        self.total_output_tokens += 5  # True/False は約5トークン

        return result

    def get_estimated_cost(self) -> float:
        """推定コストを計算"""
        if self.model_name not in self.COSTS_PER_1K_TOKENS:
            return 0.0

        costs = self.COSTS_PER_1K_TOKENS[self.model_name]
        input_cost = (self.total_input_tokens / 1000) * costs["input"]
        output_cost = (self.total_output_tokens / 1000) * costs["output"]

        return input_cost + output_cost

    def print_cost_summary(self):
        """コストサマリーを表示"""
        print(f"\nCost Summary ({self.model_name}):")
        print(f"  Input tokens: {self.total_input_tokens:.0f}")
        print(f"  Output tokens: {self.total_output_tokens:.0f}")
        print(f"  Estimated cost: ${self.get_estimated_cost():.4f}")
```

### 例2: ロギングOracle

詳細なログを記録：

```python
import logging
from datetime import datetime

class LoggingOracle(OntologyAlignmentOracle):
    """ロギング対応Oracle"""

    def __init__(self, provider: LLMProvider, log_file: str = "oracle.log"):
        super().__init__(provider)

        # ロガーを設定
        self.logger = logging.getLogger("OracleLogger")
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def diagnose_mapping(self, mapping: EntityMapping) -> DiagnosticResult:
        start_time = datetime.now()

        self.logger.info(f"Diagnosing mapping: {mapping}")

        try:
            result = super().diagnose_mapping(mapping)

            duration = (datetime.now() - start_time).total_seconds()

            self.logger.info(
                f"Result: {result.prediction}, "
                f"Duration: {duration:.2f}s"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error diagnosing {mapping}: {e}")
            raise
```

### 例3: アンサンブルOracle

複数のLLMの投票で決定：

```python
from collections import Counter

class EnsembleOracle:
    """複数LLMのアンサンブルOracle"""

    def __init__(self, providers: List[LLMProvider]):
        self.oracles = [
            OntologyAlignmentOracle(provider)
            for provider in providers
        ]

    def diagnose_mapping(
        self,
        mapping: EntityMapping,
        voting: str = "majority"
    ) -> DiagnosticResult:
        """アンサンブル診断"""

        # 各Oracleで診断
        predictions = []
        for oracle in self.oracles:
            result = oracle.diagnose_mapping(mapping)
            predictions.append(result.prediction)

        # 投票
        if voting == "majority":
            # 多数決
            final_prediction = Counter(predictions).most_common(1)[0][0]
        elif voting == "unanimous":
            # 全員一致
            final_prediction = all(predictions)
        else:
            raise ValueError(f"Unknown voting method: {voting}")

        return DiagnosticResult(
            mapping=mapping,
            prediction=final_prediction,
            raw_response=f"Votes: {predictions}"
        )
```

---

## まとめ

この実装ガイドでは以下をカバーしました：

1. **カスタムLLMプロバイダー**: 新しいLLMサービスの統合方法
2. **カスタムプロンプト**: PNLF^S以外のテンプレート作成
3. **メトリクス拡張**: 独自の評価指標の追加
4. **最適化**: 並列処理、キャッシング、プログレスバー
5. **実装例**: コスト追跡、ロギング、アンサンブル

**次のステップ**:
- [ベストプラクティス](05_best_practices.md)で効果的な使い方を学習
- 実際のプロジェクトで実装を試す
