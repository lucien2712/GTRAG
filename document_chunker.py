"""
TimeRAG 系統的文件分塊模組。

此模組負責將長篇文件，依據指定的 Token 數量，切割成更小、
可管理的文字片段（Chunks），並保留上下文重疊（Overlap）的部分，
以確保在後續資訊擷取時，不會因為切割而遺失跨區塊的關聯資訊。
"""

import logging
import re
from typing import List, Dict, Any
from dataclasses import dataclass, field

# 嘗試導入 tiktoken，如果失敗則使用替代方案
try:
    import tiktoken
except ImportError:
    tiktoken = None

from config import ChunkingConfig

logger = logging.getLogger(__name__)


# --- 資料結構定義 ---

@dataclass
class DocumentChunk:
    """定義一個文件區塊的資料結構。"""
    content: str
    chunk_id: str
    source_doc_id: str
    chunk_index: int
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# --- 文件分塊核心類別 ---

class DocumentChunker:
    """
    實現文件分塊功能的核心類別。
    """
    
    def __init__(self, config: ChunkingConfig = None):
        """
        初始化分塊器。
        """
        self.config = config or ChunkingConfig()
        self.tokenizer = self._get_tokenizer()

    def _get_tokenizer(self):
        """獲取 tiktoken 編碼器。"""
        if not tiktoken:
            raise ImportError("tiktoken 套件未安裝，無法進行分塊。請執行 `pip install tiktoken`。")
        try:
            return tiktoken.get_encoding(self.config.ENCODING_NAME)
        except Exception as e:
            logger.warning(f"無法載入 tiktoken 編碼器 '{self.config.ENCODING_NAME}': {e}。")
            raise e

    def count_tokens(self, text: str) -> int:
        """計算文字的 Token 數量。"""
        return len(self.tokenizer.encode(text))

    def chunk(self, doc_id: str, text: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """
        將單一文件分塊。
        """
        # 如果文件本身不大於最大 Token 限制，則直接作為單一區塊返回
        if self.count_tokens(text) <= self.config.MAX_TOKENS_PER_CHUNK:
            chunk_id = f"{doc_id}_chunk_0"
            chunk_metadata = (metadata or {}).copy()
            chunk_metadata['chunk_id'] = chunk_id
            return [DocumentChunk(
                content=text,
                chunk_id=chunk_id,
                source_doc_id=doc_id,
                chunk_index=0,
                token_count=self.count_tokens(text),
                metadata=chunk_metadata
            )]

        # 使用句子作為基本切割單位
        sentences = self._split_by_sentences(text)
        chunks = self._create_chunks_from_sentences(doc_id, sentences, metadata)
        # 未來可在此處加入重疊(overlap)邏輯
        return chunks

    def _split_by_sentences(self, text: str) -> List[str]:
        """將文本分割成句子列表。"""
        sentences = re.split(r'([。！？；.!?])', text)
        sentences = ["".join(sentences[i:i+2]) for i in range(0, len(sentences), 2)]
        return [s.strip() for s in sentences if s.strip()]

    def _create_chunks_from_sentences(self, doc_id: str, sentences: List[str], metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """從句子列表創建初始的文件區塊。"""
        chunks: List[DocumentChunk] = []
        current_chunk_content = []
        current_token_count = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_token_count = self.count_tokens(sentence)
            
            if current_token_count + sentence_token_count > self.config.MAX_TOKENS_PER_CHUNK and current_chunk_content:
                self._finalize_chunk(chunks, doc_id, chunk_index, " ".join(current_chunk_content), metadata)
                chunk_index += 1
                current_chunk_content = []
                current_token_count = 0

            current_chunk_content.append(sentence)
            current_token_count += sentence_token_count

        if current_chunk_content:
            self._finalize_chunk(chunks, doc_id, chunk_index, " ".join(current_chunk_content), metadata)

        return chunks

    def _finalize_chunk(self, chunks_list: List, doc_id: str, index: int, content: str, metadata: Dict):
        """建立並儲存一個新的 chunk。"""
        chunk_id = f"{doc_id}_chunk_{index}"
        chunk_metadata = (metadata or {}).copy()
        chunk_metadata['chunk_id'] = chunk_id
        
        chunks_list.append(DocumentChunk(
            content=content,
            chunk_id=chunk_id,
            source_doc_id=doc_id,
            chunk_index=index,
            token_count=self.count_tokens(content),
            metadata=chunk_metadata
        ))