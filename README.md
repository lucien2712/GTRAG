# TimeRAG

一個具備時間感知能力的知識圖譜問答(RAG)框架，特別適用於需要分析和理解跨時間資訊的場景，例如財報分析、市場研究等。

---

## 核心功能

- **時間感知圖譜**: 自動建立實體在不同時間點的狀態，並連接成演化路徑。
- **知識圖譜驅動**: 將非結構化文本轉換為結構化的知識圖譜，實現更精確的資訊檢索。
- **模型設定集中化**: 所有 LLM 與 Embedding 模型在系統初始化時一次性設定，簡化後續呼叫。
- **客製化**: 系統的各個環節（實體類型、提示詞、模型參數）都易於配置。
- **支援自訂模型**: 可輕鬆接入任何自訂的 LLM 和嵌入模型函式。

## 系統架構

TimeRAG 的運作流程如下：

```
文件 -> [1. 文件分塊] -> [2. LLM 資訊擷取] -> [3. 建構知識圖譜] -> [4. 時間連結]

使用者問題 -> [5. 查詢理解] -> [6. 圖譜檢索] -> [7. 上下文生成] -> [8. LLM 生成答案]
```

1.  **文件分塊**: 將長文件切割成適合處理的小片段。
2.  **資訊擷取**: 使用 LLM 從每個片段中擷取實體和關係。
3.  **建構知識圖譜**: 將擷取到的資訊存入 `networkx` 圖結構中。
4.  **時間連結**: 為同一個實體在不同時間的節點建立「演化」關係邊。
5.  **查詢理解**: 使用 LLM 分析使用者問題，拆解出關鍵詞和意圖。
6.  **圖譜檢索**: 在知識圖譜中搜尋與問題最相關的節點和路徑。
7.  **上下文生成**: 組合檢索到的資訊，並根據 Token 限制進行截斷。
8.  **生成答案**: 將組合好的上下文和原始問題交給 LLM，生成最終答案。


## 快速上手

以下範例展示了如何初始化系統、索引文件並提出查詢。您可以直接執行 `examples/demo.py` 來查看效果。

```python
# 節錄自 examples/demo.py

import os
from dotenv import load_dotenv
from timerag_system import GraphRAGSystem
from config import QueryParams

# 載入 .env 檔案中的環境變數
load_dotenv()

# --- 自訂模型函式 (可選) ---
def gpt_4o_mini_llm(system_prompt: str, user_prompt: str) -> str:
    # ... 此處省略呼叫 OpenAI API 的實作細節 ...
    pass

def openai_embedding_func(text: str) -> list:
    # ... 此處省略呼叫 OpenAI Embedding API 的實作細節 ...
    pass

# 1. 初始化系統
# 所有模型與金鑰設定皆在此完成。
# 如果不傳入 llm_func 或 embedding_func，系統會使用預設的 OpenAI 模型。
rag = GraphRAGSystem(
    llm_func=gpt_4o_mini_llm,
    embedding_func=openai_embedding_func
)

# 2. 插入文件 (Insert)
# `quarter` (季度) 資訊透過 metadata 傳入，是時間感知的關鍵
documents = [
    {"text": "蘋果公司在2023年Q4的iPhone銷量達到8000萬部。", "doc_id": "apple_q4_2023", "metadata": {"quarter": "2023Q4"}},
    {"text": "到了2024年Q1，蘋果的iPhone銷量增長至9000萬部。", "doc_id": "apple_q1_2024", "metadata": {"quarter": "2024Q1"}},
]

for doc in documents:
    rag.insert(doc["text"], doc["doc_id"], doc["metadata"])

# 3. 建立時間連結
# 索引完所有文件後，執行此步驟以建立跨時間的關聯
rag.build_temporal_links()

# 4. 提出問題 (Query)
question = "蘋果iPhone銷量的趨勢如何？"
result = rag.query(question)

# 5. 查看結果
print(result.get("answer"))
```

## 客製化設定

您可以輕易地修改位於 `configs/` 資料夾中的設定檔來客製化系統。

-   **實體類型**: 修改 `configs/entity_types.json` 來新增、刪除或修改您想擷取的實體類型及其描述。
    
-   **提示詞 (Prompts)**: 修改 `configs/prompts.py` 來改變系統與 LLM 互動的方式。例如，您可以修改 `RESULT_SUMMARIZATION` 中的模板，來改變最終答案的語氣或格式。

## 參數設定

### 索引參數 (Chunking)

您可以在初始化 `GraphRAGSystem` 時，透過傳入 `ChunkingConfig` 物件來設定文件分塊的行為。

-   `MAX_TOKENS_PER_CHUNK` (int): 每個文字區塊的最大 Token 數量。
-   `OVERLAP_TOKENS` (int): 區塊之間重疊的 Token 數量，用以保留上下文。

**設定範例:**

```python
from config import ChunkingConfig
from timerag_system import GraphRAGSystem

# 建立自訂的分塊設定
# 適合處理較長、需要更多上下文的段落
chunk_config = ChunkingConfig(
    MAX_TOKENS_PER_CHUNK=2000,
    OVERLAP_TOKENS=300
)

# 初始化系統時傳入
rag = GraphRAGSystem(chunking_config=chunk_config)

# 後續所有 .insert() 都會使用此分塊設定
rag.insert(...)
```

### 查詢參數 (Querying)

您可以透過 `QueryParams` 物件來精細地控制查詢時的行為。

-   `top_k` (int): 控制從圖譜中檢索回的實體/關係數量上限。
-   `similarity_threshold` (float): 語意相似度的門檻值，只有高於此分數的結果會被考慮。
-   `max_hops` (int): 在圖中進行多跳擴展時，探索的最大步數。
-   `final_max_tokens` (int): 最終組合給 LLM 的總 Token 上限，用以控制成本與避免超出模型限制。

有兩種方式可以設定這些參數：

**1. 初始化時設定 (全域預設)**

在建立 `GraphRAGSystem` 物件時傳入 `query_params`，這將會成為所有查詢的預設值。

```python
from config import QueryParams
from timerag_system import GraphRAGSystem

# 建立一個自訂的參數設定
default_params = QueryParams(
    top_k=15,
    similarity_threshold=0.3,
    final_max_tokens=10000
)

# 初始化系統時傳入
rag = GraphRAGSystem(query_params=default_params)

# 後續所有查詢都會使用這套預設參數
result = rag.query("...") 
```

**2. 查詢時設定 (單次有效)**

在呼叫 `.query()` 方法時傳入 `query_params`，這次的設定將只對當次查詢有效，不會影響全域預設值。

```python
# 假設 rag 已經被初始化
rag = GraphRAGSystem() 

# 針對一個複雜問題，使用更寬鬆的檢索參數
specific_params = QueryParams(
    top_k=20,
    max_hops=4 
)

result = rag.query(
    "一個需要深度分析的複雜問題...",
    query_params=specific_params
)
```

## 其他範例

在 `examples/` 資料夾中，您可以找到更完整的程式碼：

-   `demo.py`: 包含一個完整、可執行的範例，展示了初始化、索引、查詢的完整流程。
