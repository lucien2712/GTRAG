"""
TimeRAG 系統的知識圖譜建構模組。

此模組的核心是 `GraphBuilder` 類別，它使用 `networkx` 套件來建立、
管理和儲存知識圖譜。圖譜由代表「實體」的節點和代表「關係」的邊構成。

主要功能包括：
- 將擷取出的實體和關係新增至圖中。
- 為實體和關係描述生成語意嵌入向量（Embedding）。
- 建立跨越不同時間區段的「時間邊」，捕捉資訊的時序性。
- 提供圖的儲存與載入功能。
"""

import json
import logging
from typing import List, Dict, Any, Optional, Callable
import networkx as nx
import numpy as np

# 嘗試導入 sentence_transformers，如果失敗則設為 None
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from llm_extractor import Entity, Relation
from config import ModelConfig

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    負責建立和管理知識圖譜。
    
    這個類別封裝了所有與圖操作相關的邏輯，包括新增節點、新增邊、
    建立時間關聯以及序列化（儲存/載入）。
    """
    
    def __init__(self, embedding_model_name: str = None, embedding_func: Callable[[str], np.ndarray] = None):
        """
        初始化 GraphBuilder。

        支援兩種嵌入向量生成模式：
        1. `embedding_model_name` 模式：使用 `sentence-transformers` 函式庫載入預訓練模型。
        2. `embedding_func` 模式：使用一個使用者自訂的函數來生成嵌入向量。

        Args:
            embedding_model_name (str, optional): SentenceTransformer 模型的名稱。
            embedding_func (Callable, optional): 一個自訂的嵌入向量生成函數。
        """
        self.graph = nx.MultiDiGraph() # 使用多重有向圖，允許節點間存在多種類型的關係
        
        if embedding_func:
            self.encode = embedding_func
        elif SentenceTransformer and (embedding_model_name or ModelConfig.EMBEDDING_MODEL):
            model = embedding_model_name or ModelConfig.EMBEDDING_MODEL
            self.encoder = SentenceTransformer(model)
            self.encode = self._default_encode
        else:
            raise ImportError("必須提供 `embedding_func` 或安裝 `sentence-transformers` 函式庫並指定模型名稱。")

    def _default_encode(self, text: str) -> np.ndarray:
        """使用預設的 SentenceTransformer 進行編碼。"""
        return self.encoder.encode(text)
        
    def add_entities(self, entities: List[Entity]):
        """將一批實體作為節點新增到圖中，如果節點已存在，則合併其屬性。"""
        for entity in entities:
            node_id = f"{entity.entity_name}__{entity.metadata.get('quarter', 'Q_UNKNOWN')}"
            
            if self.graph.has_node(node_id):
                # 節點已存在，進行合併
                existing_data = self.graph.nodes[node_id]
                
                # 1. 合併描述
                new_description = (existing_data.get('description', '') + 
                                 "\n---\n" + 
                                 entity.entity_description)
                
                # 2. 合併來源文件 (確保是列表)
                source_docs = existing_data.get('source_doc_id', [])
                if not isinstance(source_docs, list):
                    source_docs = [source_docs]
                if entity.source_doc_id not in source_docs:
                    source_docs.append(entity.source_doc_id)

                # 3. 重新計算合併後的嵌入向量
                new_embedding = self.encode(new_description)

                # 4. 更新節點屬性
                self.graph.nodes[node_id]['description'] = new_description
                self.graph.nodes[node_id]['embedding'] = new_embedding
                self.graph.nodes[node_id]['source_doc_id'] = source_docs
                logger.info(f"合併實體節點: {node_id}")

            else:
                # 節點不存在，正常新增
                embedding = self.encode(entity.entity_description)
                self.graph.add_node(
                    node_id,
                    node_type="entity",
                    name=entity.entity_name,
                    type=entity.entity_type,
                    description=entity.entity_description,
                    embedding=embedding,
                    source_doc_id=[entity.source_doc_id], # 初始化為列表
                    **entity.metadata
                )
    
    def add_relations(self, relations: List[Relation]):
        """將一批關係作為邊新增到圖中，如果關係已存在，則合併其屬性。"""
        for relation in relations:
            source_id = f"{relation.source_entity_name}__{relation.metadata.get('quarter', 'Q_UNKNOWN')}"
            target_id = f"{relation.target_entity_name}__{relation.metadata.get('quarter', 'Q_UNKNOWN')}"
            key = relation.relation_name

            if self.graph.has_node(source_id) and self.graph.has_node(target_id):
                if self.graph.has_edge(source_id, target_id, key=key):
                    # 邊已存在，進行合併
                    existing_data = self.graph.get_edge_data(source_id, target_id, key=key)
                    
                    # 1. 合併描述
                    new_description = (existing_data.get('description', '') + 
                                     "\n---\n" + 
                                     relation.relation_description)
                    
                    # 2. 合併證據
                    new_evidence = (existing_data.get('evidence', '') + 
                                  "\n---" + 
                                  relation.evidence)

                    # 3. 更新邊的屬性
                    self.graph[source_id][target_id][key]['description'] = new_description
                    self.graph[source_id][target_id][key]['evidence'] = new_evidence
                    logger.info(f"合併關係邊: {source_id} -> {target_id} ({key})")

                else:
                    # 邊不存在，正常新增
                    self.graph.add_edge(
                        source_id,
                        target_id,
                        key=key,
                        relation_name=relation.relation_name,
                        description=relation.relation_description,
                        evidence=relation.evidence,
                        **relation.metadata
                    )
    
    def build_temporal_connections(self):
        """建立跨時間的連接邊，這是 TimeRAG 的核心功能之一。"""
        entity_nodes = {}
        # 依實體名稱將所有節點分組
        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") == "entity":
                entity_name = data.get("name")
                if entity_name not in entity_nodes:
                    entity_nodes[entity_name] = []
                # 記錄節點 ID 和其季度資訊
                entity_nodes[entity_name].append((data.get("quarter", ""), node_id))
        
        # 為同一個實體，在不同季度的節點之間建立時間演化邊
        for entity_name, nodes in entity_nodes.items():
            # 按季度排序，確保時間順序正確
            sorted_nodes = sorted(nodes, key=lambda x: x[0])
            
            for i in range(len(sorted_nodes) - 1):
                source_q, source_id = sorted_nodes[i]
                target_q, target_id = sorted_nodes[i+1]
                
                self.graph.add_edge(
                    source_id,
                    target_id,
                    key="temporal_evolution",
                    relation_name="temporal_evolution",
                    description=f"{entity_name} from {source_q} to {target_q}",
                    edge_type="temporal"
                )
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """獲取圖的統計資訊。"""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": list(set(nx.get_node_attributes(self.graph, 'node_type').values())),
            "edge_types": list(set(nx.get_edge_attributes(self.graph, 'relation_name').values()))
        }
    
    def save(self, filepath: str):
        """將圖的資料序列化並儲存為 JSON 檔案。"""
        # networkx 的 node_link_data 格式便於 JSON 序列化
        # 但需要手動處理 numpy 陣列，將其轉換為 list
        graph_data = nx.node_link_data(self.graph)
        for node in graph_data["nodes"]:
            if 'embedding' in node and isinstance(node['embedding'], np.ndarray):
                node['embedding'] = node['embedding'].tolist()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        """從 JSON 檔案載入圖資料並重建圖。"""
        with open(filepath, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        # 將 embedding 從 list 轉換回 numpy 陣列
        for node in graph_data["nodes"]:
            if 'embedding' in node and isinstance(node['embedding'], list):
                node['embedding'] = np.array(node['embedding'])
        
        self.graph = nx.node_link_graph(graph_data)
