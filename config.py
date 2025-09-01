"""
TimeRAG 系統的核心設定檔。

這個檔案使用 dataclass 來定義各模組的設定參數，方便管理與調用。
使用者可以透過修改這個檔案來客製化系統的行為。
"""
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path

# --- 實體設定 ---
@dataclass
class EntityTypes:
    """
    從 JSON 設定檔載入與管理實體類型定義。

    這個類別會讀取 'configs/entity_types.json' 檔案，
    並動態地將實體類型設定為類別屬性，方便在程式中直接調用。
    採用單例模式（Singleton）確保全局只有一個實體。
    """
    _instance = None
    _entity_config: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self._load_entity_types()

    def _load_entity_types(self):
        """從 JSON 設定檔載入實體類型。"""
        config_path = Path(__file__).parent / "configs" / "entity_types.json"
        if not config_path.exists():
            raise FileNotFoundError(f"實體類型設定檔 'entity_types.json' 不存在於 'configs' 資料夾中: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self._entity_config = config
        
        # 動態設定實體類型為類別屬性，例如 self.COMPANY = "COMPANY"
        if "entity_types" in config:
            for entity_type in config["entity_types"]:
                setattr(self, entity_type, entity_type)

    @classmethod
    def get_instance(cls):
        """獲取 EntityTypes 的單例。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_all_types(self) -> List[str]:
        """獲取所有實體類型的名稱列表。"""
        return list(self._entity_config.get("entity_types", {}).keys())

    def get_descriptions(self) -> Dict[str, str]:
        """獲取所有實體類型的中文描述。"""
        return {
            entity_type: details.get("description_zh", "")
            for entity_type, details in self._entity_config.get("entity_types", {}).items()
        }

    def get_descriptions_en(self) -> Dict[str, str]:
        """獲取所有實體類型的英文描述。"""
        return {
            entity_type: details.get("description_en", "")
            for entity_type, details in self._entity_config.get("entity_types", {}).items()
        }

# --- 模型與演算法設定 ---
@dataclass
class ModelConfig:
    """
    模型與演算法相關的設定。

    定義了語言模型、嵌入模型、檢索演算法等的核心參數。
    """
    # OpenAI 大型語言模型設定
    DEFAULT_MODEL: str = "gpt-4"
    TEMPERATURE: float = 0.1

    # 語意嵌入模型設定
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # 檢索演算法設定
    DEFAULT_TOP_K: int = 10  # 預設檢索回傳的節點數量
    SIMILARITY_THRESHOLD: float = 0.3  # 語意相似度門檻值
    MAX_NEIGHBORS_PER_HOP: int = 5  # 在圖中每跳允許探索的最大鄰居數量
    MAX_HOPS: int = 3  # 在圖中探索的最大跳數

    # 重要性分數門檻值
    HIGH_IMPORTANCE_THRESHOLD: float = 0.7
    MEDIUM_IMPORTANCE_THRESHOLD: float = 0.5

    # 圖譜設定
    MAX_CHAIN_LENGTH: int = 4  # 推理鏈的最大長度
    MAX_REASONING_CHAINS: int = 10  # 最大推理鏈數量

# --- 索引參數設定 ---
@dataclass
class IndexingParams:
    """
    文件索引階段的參數設定。

    這些參數影響文件如何被處理並加入到知識圖譜中。
    """
    enable_entity_linking: bool = True  # 是否啟用實體連結（將相同實體關聯起來）
    enable_temporal_connections: bool = True  # 是否啟用時間關聯分析
    entity_similarity_threshold: float = 0.8  # 實體連結時的相似度門檻值

    def to_dict(self) -> Dict[str, Any]:
        """將參數轉換為字典格式。"""
        return {
            "enable_entity_linking": self.enable_entity_linking,
            "enable_temporal_connections": self.enable_temporal_connections,
            "entity_similarity_threshold": self.entity_similarity_threshold
        }

# --- 查詢參數設定 ---
@dataclass
class QueryParams:
    """
    查詢階段的參數設定。

    使用者在提出問題時，可透過這些參數客製化查詢行為。
    """
    # 檢索參數
    top_k: int = 10
    max_hops: int = 3
    max_neighbors_per_hop: int = 5
    similarity_threshold: float = 0.3
    enable_multi_hop: bool = True  # 是否啟用多跳查詢

    # 混合檢索權重
    centrality_weight: float = 0.3  # 圖中心性權重
    similarity_weight: float = 0.7  # 語意相似度權重

    # Token 管理參數
    entity_max_tokens: int = 30000  # 實體資訊的最大 Token 數
    relation_max_tokens: int = 30000  # 關係資訊的最大 Token 數
    final_max_tokens: int = 120000  # 最終送入模型的總 Token 數上限

    def to_dict(self) -> Dict[str, Any]:
        """將參數轉換為字典格式。"""
        return {
            "top_k": self.top_k,
            "max_hops": self.max_hops,
            "max_neighbors_per_hop": self.max_neighbors_per_hop,
            "similarity_threshold": self.similarity_threshold,
            "enable_multi_hop": self.enable_multi_hop,
            "centrality_weight": self.centrality_weight,
            "similarity_weight": self.similarity_weight,
            "entity_max_tokens": self.entity_max_tokens,
            "relation_max_tokens": self.relation_max_tokens,
            "final_max_tokens": self.final_max_tokens
        }

# --- 文件分塊設定 ---
@dataclass
class ChunkingConfig:
    """
    文件分塊（Chunking）的設定。

    定義如何將長文件切割成適合模型處理的小片段。
    """
    # Token 限制
    MAX_TOKENS_PER_CHUNK: int = 3000  # 每個區塊的最大 Token 數
    OVERLAP_TOKENS: int = 200  # 區塊之間的重疊 Token 數
    MIN_CHUNK_TOKENS: int = 500  # 每個區塊的最小 Token 數

    # 切割偏好
    SENTENCE_BOUNDARY: bool = True  # 是否依據句子邊界切割
    PARAGRAPH_BOUNDARY: bool = True  # 是否依據段落邊界切割
    PRESERVE_SECTIONS: bool = True  # 是否保留章節結構

    # 處理設定
    MERGE_DUPLICATE_ENTITIES: bool = True  # 是否合併重複的實體
    MERGE_DUPLICATE_RELATIONS: bool = True  # 是否合併重複的關係
    VALIDATE_CROSS_CHUNK_CONSISTENCY: bool = True  # 是否驗證跨區塊的一致性

    # 編碼器名稱 (用於 tiktoken)
    ENCODING_NAME: str = "o200k_base"

# --- 提示詞設定 ---
@dataclass
class PromptConfig:
    """
    從 Python 模組載入與管理提示詞（Prompt）模板。

    這個類別會動態載入 'configs/prompts.py' 檔案，
    讓使用者可以集中管理所有與 LLM 互動的提示詞。
    採用單例模式（Singleton）。
    """
    _instance = None
    _prompt_config: Dict[str, Any] = field(default_factory=dict, init=False)
    _prompts_module: Any = field(default=None, init=False)

    def __post_init__(self):
        self._load_prompts()

    def _load_prompts(self):
        """從 Python 模組動態載入提示詞模板。"""
        import sys
        import importlib.util
        
        config_path = Path(__file__).parent / "configs" / "prompts.py"
        if not config_path.exists():
            raise FileNotFoundError(f"提示詞設定檔 'prompts.py' 不存在於 'configs' 資料夾中: {config_path}")

        spec = importlib.util.spec_from_file_location("prompts_config", config_path)
        prompts_module = importlib.util.module_from_spec(spec)
        if spec.loader:
            spec.loader.exec_module(prompts_module)
        
        self._prompts_module = prompts_module
        if hasattr(prompts_module, 'get_all_configs'):
            self._prompt_config = prompts_module.get_all_configs()

    @classmethod
    def get_instance(cls):
        """獲取 PromptConfig 的單例。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_entity_extraction_prompt(self) -> Dict[str, Any]:
        """獲取實體擷取的提示詞模板。"""
        return self._prompt_config.get("entity_extraction", {})

    def get_relation_extraction_prompt(self) -> Dict[str, Any]:
        """獲取關係擷取的提示詞模板。"""
        return self._prompt_config.get("relation_extraction", {})

    def get_query_understanding_prompt(self) -> Dict[str, Any]:
        """獲取查詢理解的提示詞模板。"""
        return self._prompt_config.get("query_understanding", {})

    def get_result_summarization_prompt(self) -> Dict[str, Any]:
        """獲取結果總結的提示詞模板。"""
        return self._prompt_config.get("result_summarization", {})
    
    def reload_config(self):
        """重新載入提示詞設定（方便開發時動態更新）。"""
        import importlib
        if self._prompts_module:
            importlib.reload(self._prompts_module)
            if hasattr(self._prompts_module, 'get_all_configs'):
                self._prompt_config = self._prompts_module.get_all_configs()

# --- 檢索權重設定 ---
@dataclass
class RetrievalWeights:
    """
    混合檢索的權重設定。

    用於平衡不同檢索策略的重要性。
    """
    TEMPORAL: float = 0.6  # 時間相關性權重
    SEMANTIC: float = 0.4  # 語意相似度權重

    # 節點重要性分數權重
    DEGREE_WEIGHT: float = 0.3  # 節點度數（連接數）權重
    EDGE_WEIGHT: float = 0.4  # 邊的權重
    SEMANTIC_WEIGHT: float = 0.3  # 語意權重
