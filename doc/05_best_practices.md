# ベストプラクティスガイド

## 目次
1. [LLMモデルの選択](#llmモデルの選択)
2. [コスト最適化](#コスト最適化)
3. [パフォーマンス最適化](#パフォーマンス最適化)
4. [エラーハンドリング](#エラーハンドリング)
5. [本番環境での運用](#本番環境での運用)
6. [評価とチューニング](#評価とチューニング)

---

## LLMモデルの選択

### 推奨モデル

instructions.mdの研究結果に基づく推奨順位：

#### 1位: Google Gemini Flash 2.5 ⭐推奨

```python
from src.oracle import GeminiProvider, OntologyAlignmentOracle

provider = GeminiProvider(model="gemini-2.0-flash-exp")
oracle = OntologyAlignmentOracle(provider, system_prompt_type="biomedical_specialist")
```

**理由**:
- **Youden's Index: 0.550**（最高値）
- **コスト効率が最良**（Flash 2.5は無料/低価格）
- PNLF^Sとの相性が最適

**適用シーン**:
- 本番環境での大規模バッチ処理
- コスト重視のプロジェクト
- 研究用途

#### 2位: GPT-4o Mini

```python
from src.oracle import OpenAIProvider

provider = OpenAIProvider(model="gpt-4o-mini")
oracle = OntologyAlignmentOracle(provider)
```

**理由**:
- 良好なYouden's Index（約0.52）
- OpenAIエコシステムとの統合が容易
- 安定した性能

**適用シーン**:
- 既存のOpenAI環境がある場合
- APIの安定性を重視
- エンタープライズ環境

#### 3位: Claude 3.5 Sonnet

```python
from src.oracle import AnthropicProvider

provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
oracle = OntologyAlignmentOracle(provider)
```

**理由**:
- 高い言語理解力
- 長いコンテキスト対応
- 安全性の高い応答

**適用シーン**:
- 複雑なオントロジー構造
- 安全性重視のアプリケーション

### モデル選択のマトリックス

| 優先順位 | 最重要 | モデル |
|---------|--------|--------|
| **コスト** | ✅ | Gemini Flash 2.5 → GPT-4o Mini |
| **精度** | ✅ | Gemini Flash 2.5 → GPT-4o → Claude |
| **速度** | ✅ | Gemini Flash 2.5 → GPT-4o Mini |
| **安定性** | ✅ | GPT-4o → Claude → Gemini |

---

## コスト最適化

### 戦略1: 適切なモデル選択

```python
# ❌ 高コスト
provider = OpenAIProvider(model="gpt-4o")  # $0.0025/1K input tokens

# ✅ コスト効率良
provider = GeminiProvider(model="gemini-2.0-flash-exp")  # 無料または低価格
```

### 戦略2: プロンプト長の最適化

```python
# 不要な情報は省く
entity = OntologyEntity(
    label="neuron",
    parent_class="nerve cell",
    synonyms=["nerve cell"],  # 重複する類義語は除外
    # ontology_id は診断に不要なら省略
)
```

### 戦略3: バッチ処理の効率化

```python
# キャッシングを使用して重複診断を避ける
from doc.implementation_guide import CachedOracle

oracle = CachedOracle(provider)

# 同じマッピングの再診断は無料
result1 = oracle.diagnose_mapping(mapping)
result2 = oracle.diagnose_mapping(mapping)  # キャッシュヒット
```

### 戦略4: コスト追跡

```python
from doc.implementation_guide import CostTrackingOracle

oracle = CostTrackingOracle(provider, model_name="gpt-4o-mini")

# バッチ処理
results = oracle.diagnose_batch(mappings)

# コストを確認
oracle.print_cost_summary()
# Output:
#   Input tokens: 15000
#   Output tokens: 500
#   Estimated cost: $0.0052
```

### 戦略5: 段階的フィルタリング

```python
# 1. 安価なモデルで簡単なケースをフィルタ
cheap_oracle = OntologyAlignmentOracle(GeminiProvider())
uncertain_mappings = []

for mapping in all_mappings:
    result = cheap_oracle.diagnose_mapping(mapping)
    # 確信度が低い場合のみ次のステップへ
    if needs_verification(result):
        uncertain_mappings.append(mapping)

# 2. 高性能モデルで不確実なケースのみを診断
expensive_oracle = OntologyAlignmentOracle(OpenAIProvider("gpt-4o"))
final_results = expensive_oracle.diagnose_batch(uncertain_mappings)
```

---

## パフォーマンス最適化

### 並列処理

```python
from concurrent.futures import ThreadPoolExecutor

class OptimizedOracle(OntologyAlignmentOracle):
    def diagnose_batch(self, mappings, max_workers=10):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.diagnose_mapping, mappings))
        return results

# 使用
oracle = OptimizedOracle(GeminiProvider())
results = oracle.diagnose_batch(large_mapping_list, max_workers=10)
```

**推奨並列数**:
- Gemini: 10-20 (レート制限に注意)
- OpenAI: 5-10
- Anthropic: 5

### レート制限の管理

```python
import time
from ratelimit import limits, sleep_and_retry

class RateLimitedOracle(OntologyAlignmentOracle):
    @sleep_and_retry
    @limits(calls=60, period=60)  # 60 calls per minute
    def diagnose_mapping(self, mapping):
        return super().diagnose_mapping(mapping)
```

### キャッシング戦略

```python
import pickle
from pathlib import Path

class PersistentCacheOracle(OntologyAlignmentOracle):
    """永続的なキャッシュ"""

    def __init__(self, provider, cache_file="oracle_cache.pkl"):
        super().__init__(provider)
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()

    def _load_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)

    def diagnose_mapping(self, mapping):
        key = self._get_cache_key(mapping)

        if key in self.cache:
            return self.cache[key]

        result = super().diagnose_mapping(mapping)
        self.cache[key] = result
        self._save_cache()

        return result
```

---

## エラーハンドリング

### リトライロジック

```python
import time
from functools import wraps

def retry_on_error(max_attempts=3, delay=1, backoff=2):
    """リトライデコレータ"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay

            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise

                    print(f"Attempt {attempt} failed: {e}")
                    print(f"Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff

        return wrapper
    return decorator

class RobustOracle(OntologyAlignmentOracle):
    @retry_on_error(max_attempts=3, delay=1, backoff=2)
    def diagnose_mapping(self, mapping):
        return super().diagnose_mapping(mapping)
```

### グレースフルデグラデーション

```python
class FallbackOracle(OntologyAlignmentOracle):
    """フォールバック付きOracle"""

    def __init__(self, primary_provider, fallback_provider):
        self.primary_oracle = OntologyAlignmentOracle(primary_provider)
        self.fallback_oracle = OntologyAlignmentOracle(fallback_provider)

    def diagnose_mapping(self, mapping):
        try:
            return self.primary_oracle.diagnose_mapping(mapping)
        except Exception as e:
            print(f"Primary failed: {e}. Using fallback...")
            return self.fallback_oracle.diagnose_mapping(mapping)
```

### エラーログ

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('oracle_errors.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class LoggedOracle(OntologyAlignmentOracle):
    def diagnose_mapping(self, mapping):
        try:
            logger.info(f"Diagnosing: {mapping}")
            result = super().diagnose_mapping(mapping)
            logger.info(f"Result: {result.prediction}")
            return result
        except Exception as e:
            logger.error(f"Error diagnosing {mapping}: {e}", exc_info=True)
            raise
```

---

## 本番環境での運用

### 環境変数の管理

```python
# .env ファイル
GOOGLE_API_KEY=your_api_key_here
OPENAI_API_KEY=your_api_key_here
ANTHROPIC_API_KEY=your_api_key_here
ENVIRONMENT=production
```

```python
# Python
from dotenv import load_dotenv
import os

load_dotenv()

# 環境に応じた設定
if os.getenv("ENVIRONMENT") == "production":
    provider = GeminiProvider(model="gemini-2.0-flash-exp")
    max_workers = 10
else:
    provider = MockProvider()  # 開発環境ではMock
    max_workers = 1
```

### モニタリング

```python
import time
from datetime import datetime

class MonitoredOracle(OntologyAlignmentOracle):
    """メトリクスモニタリング付きOracle"""

    def __init__(self, provider):
        super().__init__(provider)
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_time': 0.0
        }

    def diagnose_mapping(self, mapping):
        start = time.time()
        self.metrics['total_requests'] += 1

        try:
            result = super().diagnose_mapping(mapping)
            self.metrics['successful_requests'] += 1
            return result
        except Exception as e:
            self.metrics['failed_requests'] += 1
            raise
        finally:
            self.metrics['total_time'] += time.time() - start

    def get_metrics(self):
        """メトリクスを取得"""
        return {
            **self.metrics,
            'success_rate': self.metrics['successful_requests'] / max(self.metrics['total_requests'], 1),
            'avg_time': self.metrics['total_time'] / max(self.metrics['total_requests'], 1)
        }
```

### バッチ処理のベストプラクティス

```python
def process_large_dataset(mappings, batch_size=100):
    """大規模データセットの処理"""
    oracle = OntologyAlignmentOracle(GeminiProvider())

    all_results = []

    # バッチに分割
    for i in range(0, len(mappings), batch_size):
        batch = mappings[i:i+batch_size]

        print(f"Processing batch {i//batch_size + 1}/{(len(mappings)-1)//batch_size + 1}")

        try:
            # バッチ処理
            results = oracle.diagnose_batch(batch)
            all_results.extend(results)

            # 中間結果を保存（障害時のリカバリ用）
            save_checkpoint(all_results, f"checkpoint_{i}.pkl")

            # レート制限対策の待機
            time.sleep(1)

        except Exception as e:
            print(f"Batch {i} failed: {e}")
            # エラー時は次のバッチへ
            continue

    return all_results
```

---

## 評価とチューニング

### クロスバリデーション

```python
from sklearn.model_selection import KFold

def cross_validate_oracle(oracle, mappings_with_truth, n_splits=5):
    """K-Fold クロスバリデーション"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(mappings_with_truth)):
        print(f"Fold {fold + 1}/{n_splits}")

        test_mappings = [mappings_with_truth[i] for i in test_idx]

        # テストセットで評価
        results = oracle.diagnose_batch(test_mappings)
        metrics = oracle.evaluate_performance(results)

        fold_metrics.append(metrics)

    # 平均メトリクスを計算
    avg_metrics = {}
    for key in fold_metrics[0].keys():
        if isinstance(fold_metrics[0][key], (int, float)):
            avg_metrics[f"avg_{key}"] = sum(m[key] for m in fold_metrics) / n_splits

    return avg_metrics
```

### A/Bテスト

```python
def compare_oracles(oracle_a, oracle_b, test_mappings):
    """2つのOracleを比較"""

    print("Testing Oracle A...")
    results_a = oracle_a.diagnose_batch(test_mappings)
    metrics_a = oracle_a.evaluate_performance(results_a)

    print("Testing Oracle B...")
    results_b = oracle_b.diagnose_batch(test_mappings)
    metrics_b = oracle_b.evaluate_performance(results_b)

    # 比較表示
    print("\nComparison:")
    print(f"{'Metric':<20} {'Oracle A':>10} {'Oracle B':>10} {'Diff':>10}")
    print("-" * 52)

    for metric in ['youden_index', 'accuracy', 'f1_score']:
        a_val = metrics_a[metric]
        b_val = metrics_b[metric]
        diff = b_val - a_val
        sign = "+" if diff > 0 else ""

        print(f"{metric:<20} {a_val:>10.3f} {b_val:>10.3f} {sign}{diff:>9.3f}")
```

### システムプロンプトのチューニング

```python
def tune_system_prompt(provider, test_mappings, prompts_to_test):
    """最適なシステムプロンプトを見つける"""

    best_yi = 0
    best_prompt = None

    for prompt_name, prompt_text in prompts_to_test.items():
        oracle = OntologyAlignmentOracle(
            provider=provider,
            custom_system_prompt=prompt_text
        )

        results = oracle.diagnose_batch(test_mappings)
        metrics = oracle.evaluate_performance(results)

        yi = metrics['youden_index']
        print(f"{prompt_name}: YI = {yi:.3f}")

        if yi > best_yi:
            best_yi = yi
            best_prompt = prompt_name

    print(f"\nBest: {best_prompt} (YI = {best_yi:.3f})")
    return best_prompt, best_yi
```

---

## チェックリスト

### 開発フェーズ

- [ ] Mockプロバイダーでロジックをテスト
- [ ] 小規模データセットで動作確認
- [ ] エラーハンドリングの実装
- [ ] ログ機能の実装

### 評価フェーズ

- [ ] Ground truth付きデータで評価
- [ ] Youden's Index > 0.5 を確認
- [ ] 複数モデルで比較
- [ ] コスト試算

### 本番フェーズ

- [ ] 環境変数の設定
- [ ] レート制限の設定
- [ ] モニタリングの実装
- [ ] バックアップとリカバリ戦略
- [ ] コスト監視

---

## まとめ

### 推奨構成（本番環境）

```python
from src.oracle import OntologyAlignmentOracle, GeminiProvider
from doc.implementation_guide import CachedOracle, LoggingOracle

# 1. 最適なプロバイダー
provider = GeminiProvider(model="gemini-2.0-flash-exp")

# 2. 機能を組み合わせる
oracle = LoggingOracle(
    CachedOracle(
        OntologyAlignmentOracle(
            provider=provider,
            system_prompt_type="biomedical_specialist"
        )
    ),
    log_file="production_oracle.log"
)

# 3. バッチ処理
results = oracle.diagnose_batch(mappings)

# 4. 評価
metrics = oracle.evaluate_performance(results)
print(f"Youden's Index: {metrics['youden_index']:.3f}")
```

### 重要ポイント

1. **モデル選択**: Gemini Flash 2.5を第一選択に
2. **コスト管理**: キャッシングと適切なモデル選択
3. **エラー処理**: リトライとフォールバック
4. **モニタリング**: メトリクスとログの記録
5. **評価**: Ground truthでの継続的な評価

これらのベストプラクティスに従うことで、高品質で効率的なオントロジーアライメントシステムを構築できます。
