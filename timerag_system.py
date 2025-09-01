"""
TimeRAG 系統主理類別。

此檔案包含核心類別 `GraphRAGSystem`，它封裝了整個 RAG 流程，
提供一個簡潔的上層 API 供使用者互動。
"""

import logging
import os
import json
from typing import List, Dict, Any, Callable
from pathlib import Path
from dotenv import load_dotenv

# 導入專案模組
from config import QueryParams, ChunkingConfig
from llm_extractor import LLMExtractor
from graph_builder import GraphBuilder
from graph_retriever import GraphRetriever
from document_chunker import DocumentChunker
from token_manager import TokenManager

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphRAGSystem:
    """整合 TimeRAG 所有功能的單一介面類別。"""
    
    def __init__(self, 
                 openai_api_key: str = None,
                 llm_func: Callable[[str, str], str] = None,
                 embedding_func: Callable[[str], List[float]] = None,
                 query_params: QueryParams = None,
                 chunking_config: ChunkingConfig = None):
        """
        初始化 GraphRAG 系統。
        """
        load_dotenv()
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        # 初始化核心組件
        self.extractor = LLMExtractor(api_key=api_key, llm_func=llm_func)
        self.graph_builder = GraphBuilder(embedding_func=embedding_func)
        self.retriever = GraphRetriever(self.graph_builder, self.extractor)
        self.chunker = DocumentChunker(config=chunking_config or ChunkingConfig())
        self.token_manager = TokenManager(query_params=query_params or QueryParams())
        
        # 用於儲存原始文本區塊的內容
        self.chunk_store: Dict[str, str] = {}
        
        self.processing_stats = {"indexed_documents": 0, "indexed_chunks": 0}
        logger.info("GraphRAGSystem 初始化完成。")

    def insert(self, text: str, doc_id: str, metadata: Dict[str, Any] = None):
        """
        索引一份文件，包含分塊、儲存區塊、擷取資訊、建構圖譜。
        """
        logger.info(f"開始插入文件: {doc_id}")
        metadata = metadata or {}

        # 1. 文件分塊
        chunks = self.chunker.chunk(doc_id, text, metadata)
        logger.info(f"文件 {doc_id} 被分割成 {len(chunks)} 個區塊。")
        self.processing_stats["indexed_chunks"] += len(chunks)

        # 2. 儲存原始區塊內容並擷取資訊
        all_entities = []
        all_relations = []
        for chunk in chunks:
            # 將區塊原文存入 chunk_store
            self.chunk_store[chunk.chunk_id] = chunk.content
            
            entities, relations = self.extractor.extract(chunk.content, doc_id, chunk.metadata)
            all_entities.extend(entities)
            all_relations.extend(relations)
            logger.info(f"區塊 {chunk.chunk_id} 擷取出 {len(entities)} 個實體, {len(relations)} 個關係。")

        # 3. 將擷取到的實體與關係加入圖譜
        self.graph_builder.add_entities(all_entities)
        self.graph_builder.add_relations(all_relations)
        self.processing_stats["indexed_documents"] += 1

    def build_temporal_links(self):
        """
        在所有文件插入完畢後，建立跨時間的連結。
        """
        logger.info("正在建立時間連結...")
        self.graph_builder.build_temporal_connections()
        logger.info("時間連結建立完成。")

    def query(self, question: str, query_params: QueryParams = None) -> Dict[str, Any]:
        """
        對已建立的知識圖譜提出問題。
        """
        logger.info(f"收到查詢: {question}")
        params = query_params or self.token_manager.limits

        # 1. 查詢理解
        query_intent = self.retriever.understand_query(question)
        logger.info(f"查詢意圖分析完成: {query_intent.get('query_intent')}")

        # 2. 圖譜檢索
        retrieved_entities, retrieved_relations = self.retriever.search(
            intent=query_intent,
            top_k=params.top_k,
            similarity_threshold=params.similarity_threshold
        )
        logger.info(f"檢索到 {len(retrieved_entities)} 個實體, {len(retrieved_relations)} 個關係。")

        # 3. 根據檢索結果，找出對應的原始文本區塊
        source_chunk_ids = set()
        for item in retrieved_entities + retrieved_relations:
            if 'chunk_id' in item.get('metadata', {}):
                source_chunk_ids.add(item['metadata']['chunk_id'])
        
        source_chunks = [self.chunk_store[cid] for cid in source_chunk_ids if cid in self.chunk_store]
        logger.info(f"找到 {len(source_chunks)} 個相關的原始文本區塊。")

        # 4. 準備最終上下文 (包含實體、關係、原始文本)
        prompt_template = self.extractor.prompt_templates.prompt_config.get_result_summarization_prompt()['template']
        final_prompt, token_stats = self.token_manager.prepare_final_context(
            retrieved_entities=retrieved_entities,
            retrieved_relations=retrieved_relations,
            source_chunks=source_chunks,
            query=question,
            prompt_template=prompt_template
        )
        logger.info(f"最終上下文已準備，總 Token 數: {token_stats['total_tokens']}")

        # 5. 生成答案
        system_prompt = self.extractor.prompt_templates.prompt_config.get_result_summarization_prompt()['system_prompt']
        answer = self.extractor.llm_call(system_prompt, final_prompt)
        logger.info("答案生成完成。")

        return {
            "answer": answer,
            "retrieved_entities": retrieved_entities,
            "retrieved_relations": retrieved_relations,
            "retrieved_source_chunks": source_chunks,
            "token_stats": token_stats,
            "query_intent": query_intent
        }

    def save_graph(self, filepath: str):
        """將目前的知識圖譜與文本區塊儲存到檔案。"""
        # 儲存圖譜
        self.graph_builder.save(filepath)
        logger.info(f"知識圖譜已儲存至: {filepath}")

        # 儲存 chunk_store
        chunk_store_path = str(Path(filepath).with_suffix('.chunks.json'))
        with open(chunk_store_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunk_store, f, ensure_ascii=False, indent=2)
        logger.info(f"文本區塊已儲存至: {chunk_store_path}")

    def load_graph(self, filepath: str):
        """從檔案載入知識圖譜與文本區塊。"""
        # 載入圖譜
        self.graph_builder.load(filepath)
        self.retriever = GraphRetriever(self.graph_builder, self.extractor)
        logger.info(f"知識圖譜已從 {filepath} 載入。")

        # 載入 chunk_store
        chunk_store_path = str(Path(filepath).with_suffix('.chunks.json'))
        if Path(chunk_store_path).exists():
            with open(chunk_store_path, 'r', encoding='utf-8') as f:
                self.chunk_store = json.load(f)
            logger.info(f"文本區塊已從 {chunk_store_path} 載入。")
        else:
            logger.warning(f"找不到對應的文本區塊檔案: {chunk_store_path}，部分功能可能受限。")
            self.chunk_store = {}