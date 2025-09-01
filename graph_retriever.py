"""
TimeRAG 系統的圖譜檢索模組。

此模組的核心是 `GraphRetriever` 類別，它負責從已建構的知識圖譜中，
根據使用者的查詢，找出最相關的資訊。 

其主要檢索流程包含：
1. **查詢理解**：使用 LLM 將自然語言問題轉換為結構化的檢索關鍵詞。
2. **分層檢索**：使用不同層次的關鍵詞，分別從節點和邊兩個維度進行搜尋。
3. **多跳擴展**：從初步檢索到的節點出發，向外探索圖，發掘更多隱含的相關資訊。
"""

import logging
import json
import re
from typing import List, Dict, Any, Tuple
import numpy as np

# 嘗試導入所需函式庫
try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None

from graph_builder import GraphBuilder
from llm_extractor import LLMExtractor
from config import ModelConfig, PromptConfig

logger = logging.getLogger(__name__)


class GraphRetriever:
    """圖譜檢索器，實現了多種從圖中搜尋資訊的策略。"""
    
    def __init__(self, graph_builder: GraphBuilder, extractor: LLMExtractor):
        """
        初始化 GraphRetriever。

        Args:
            graph_builder (GraphBuilder): 已包含知識圖譜的圖譜建構器實例。
            extractor (LLMExtractor): 用於呼叫 LLM 的資訊擷取器實例。
        """
        if not cosine_similarity:
            raise ImportError("需要 scikit-learn 套件來計算相似度。請執行 `pip install scikit-learn`。")
        
        self.graph_builder = graph_builder
        self.graph = graph_builder.graph
        self.extractor = extractor
    
    def understand_query(self, query: str) -> Dict[str, Any]:
        """
        使用 LLM 分析使用者查詢，生成結構化的意圖和關鍵詞。
        """
        prompt_config = PromptConfig.get_instance()
        query_prompt_config = prompt_config.get_query_understanding_prompt()
        
        system_prompt = query_prompt_config.get("system_prompt", "")
        template = query_prompt_config.get("template", "")
        formatted_prompt = template.format(query=query)
        
        try:
            response_text = self.extractor.llm_call(system_prompt, formatted_prompt)
            intent = json.loads(response_text)
            logger.info(f"LLM 查詢理解成功: {intent}")
            return intent
        except Exception as e:
            print(f"LLM 查詢理解失敗: {e}")

    def search(self, intent: Dict[str, Any], top_k: int = 10, similarity_threshold: float = 0.5) -> Tuple[List[Dict], List[Dict]]:
        """
        根據查詢意圖，從圖譜中檢索實體和關係。

        Args:
            intent (Dict): 來自 understand_query 的結構化查詢意圖。
            top_k (int): 返回的結果數量上限。
            similarity_threshold (float): 語意相似度的門檻值。

        Returns:
            Tuple[List[Dict], List[Dict]]: 一個包含實體列表和關係列表的元組。
        """
        high_level_keys = intent.get("high_level_keys", [])
        low_level_keys = intent.get("low_level_keys", [])
        time_range = intent.get("time_range")

        # 1. 使用低層次關鍵詞（實體、術語）搜尋節點
        nodes = self._find_nodes(low_level_keys, time_range, similarity_threshold)
        
        # 2. 使用高層次關鍵詞（概念、主題）搜尋邊
        edges = self._find_edges(high_level_keys, time_range, similarity_threshold)

        # 3. 結合節點與邊的中心性分數進行排序
        ranked_nodes = sorted(nodes, key=lambda x: x['score'], reverse=True)[:top_k]
        ranked_edges = sorted(edges, key=lambda x: x['score'], reverse=True)[:top_k]

        # 4. 將節點和邊的資料轉換為實體和關係的格式
        retrieved_entities = [node['data'] for node in ranked_nodes]
        retrieved_relations = [edge['data'] for edge in ranked_edges]

        return retrieved_entities, retrieved_relations

    def _find_nodes(self, keys: List[str], time_range: List[str], threshold: float) -> List[Dict]:
        """透過語意相似度，從圖中尋找與關鍵詞匹配的節點。"""
        if not keys:
            return []
        
        query_embedding = self.graph_builder.encode(" ".join(keys))
        candidates = []

        for node_id, data in self.graph.nodes(data=True):
            if time_range and data.get("quarter") not in time_range:
                continue
            
            if data.get('embedding') is not None:
                similarity = cosine_similarity([query_embedding], [data['embedding']])[0][0]
                if similarity >= threshold:
                    candidates.append({
                        "id": node_id,
                        "score": similarity,
                        "data": data
                    })
        return candidates

    def _find_edges(self, keys: List[str], time_range: List[str], threshold: float) -> List[Dict]:
        """透過語意相似度，從圖中尋找與關鍵詞匹配的邊。"""
        if not keys:
            return []

        query_embedding = self.graph_builder.encode(" ".join(keys))
        candidates = []

        for u, v, data in self.graph.edges(data=True):
            if time_range and data.get("quarter") not in time_range:
                continue

            edge_text = f"{data.get('relation_name', '')} {data.get('description', '')}"
            edge_embedding = self.graph_builder.encode(edge_text)
            similarity = cosine_similarity([query_embedding], [edge_embedding])[0][0]

            if similarity >= threshold:
                candidates.append({
                    "id": f"{u}->{v}",
                    "score": similarity,
                    "data": data
                })
        return candidates
