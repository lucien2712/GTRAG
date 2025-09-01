"""
TimeRAG 系統的 Token 管理與截斷模組。

此模組的核心是 `TokenManager` 類別，其主要職責是確保最終組合給
大型語言模型（LLM）的上下文（Context）不會超過其 Token 長度限制。

它會根據設定的 Token 上限，智慧地截斷實體、關係和來源文本，
並以合理的順序組合它們，以在有限的空間內保留最重要的資訊。
"""

import logging
from typing import List, Dict, Any, Tuple, Callable

# 嘗試導入 tiktoken，如果失敗則使用替代方案
try:
    import tiktoken
except ImportError:
    tiktoken = None

from config import QueryParams # 從主設定檔導入查詢參數

logger = logging.getLogger(__name__)


class TokenManager:
    """管理 Token 計數與內容截斷，以符合 LLM 的輸入限制。"""
    
    def __init__(self, query_params: QueryParams = None, encoding_name: str = "cl100k_base"):
        if not tiktoken:
            raise ImportError("tiktoken 套件未安裝，無法進行 Token 管理。請執行 `pip install tiktoken`。\n")
        
        self.limits = query_params or QueryParams()
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(f"無法載入 tiktoken 編碼器 '{encoding_name}': {e}。將使用預設的 'cl100k_base'。\n")
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """計算給定文字的 Token 數量。"""
        if not text:
            return 0
        return len(self.encoding.encode(str(text)))
    
    def prepare_final_context(
        self, 
        retrieved_entities: List[Dict[str, Any]], 
        retrieved_relations: List[Dict[str, Any]], 
        source_chunks: List[str],
        query: str,
        prompt_template: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        準備最終給 LLM 的上下文，並確保不超過總 Token 限制。

        此方法會依序組合 Prompt、查詢、實體、關係和來源文本，並在過程中進行截斷。
        """
        # 1. 計算固定內容的 Token (模板和查詢本身)
        prompt_base_tokens = self.count_tokens(prompt_template.format(query="", retrieval_results=""))
        query_tokens = self.count_tokens(query)
        
        # 2. 計算可用於所有檢索結果的總預算
        total_results_budget = self.limits.final_max_tokens - prompt_base_tokens - query_tokens

        # 3. 根據預算截斷各部分內容
        # 優先保留實體和關係，剩下的空間給原始文本
        entities_text, entity_tokens = self._truncate_and_format(retrieved_entities, self.limits.entity_max_tokens, self._format_entities)
        relations_text, relation_tokens = self._truncate_and_format(retrieved_relations, self.limits.relation_max_tokens, self._format_relations)
        
        # 計算來源文本的可用預算
        source_chunk_budget = max(0, total_results_budget - entity_tokens - relation_tokens)
        chunks_text, chunk_tokens = self._truncate_and_format(source_chunks, source_chunk_budget, lambda chunks: "\n---\n".join(chunks))

        # 4. 組合最終的檢索結果字串
        retrieval_results = f"""### 相關實體 (Relevant Entities)\n{entities_text}\n\n### 相關關係 (Relevant Relations)\n{relations_text}\n\n### 原始上下文 (Source Context)\n{chunks_text}\n"""

        # 5. 組合最終的完整 Prompt
        final_prompt = prompt_template.format(query=query, retrieval_results=retrieval_results)

        # 6. 產生統計數據
        stats = {
            "total_tokens": self.count_tokens(final_prompt),
            "prompt_template_tokens": prompt_base_tokens,
            "query_tokens": query_tokens,
            "entity_tokens": entity_tokens,
            "relation_tokens": relation_tokens,
            "source_chunk_tokens": chunk_tokens,
        }
        
        logger.info(f"最終上下文已準備完成，總 Token 數: {stats['total_tokens']}")
        return final_prompt, stats

    def _truncate_and_format(self, items: List[Any], budget: int, format_func: Callable) -> Tuple[str, int]:
        """通用截斷與格式化函式。"""
        if not items:
            return "無相關資訊。", 0

        truncated_items = []
        current_tokens = 0

        for item in items:
            # 格式化單一項目以計算其 Token
            item_text = format_func([item])
            item_tokens = self.count_tokens(item_text)

            if current_tokens + item_tokens > budget:
                break
            
            truncated_items.append(item)
            current_tokens += item_tokens
        
        final_text = format_func(truncated_items)
        final_tokens = self.count_tokens(final_text)
        return final_text, final_tokens

    def _format_entities(self, entities: List[Dict[str, Any]]) -> str:
        """將實體列表格式化為字串。"""
        if not entities:
            return "無"
        return "\n".join([
            f"- {e.get('name', 'N/A')} ({e.get('type', 'N/A')}): {e.get('description', 'N/A')}"
            for e in entities
        ])

    def _format_relations(self, relations: List[Dict[str, Any]]) -> str:
        """將關係列表格式化為字串。"""
        if not relations:
            return "無"
        return "\n".join([
            f"- {r.get('source_entity_name', 'N/A')} -> {r.get('target_entity_name', 'N/A')} ({r.get('relation_name', 'N/A')})"
            for r in relations
        ])