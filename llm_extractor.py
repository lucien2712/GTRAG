"""
TimeRAG 系統的資訊擷取模組。

此模組的核心是 `LLMExtractor` 類別，它負責調用大型語言模型（LLM）
來從文本中擷取結構化的資訊，包括「實體（Entities）」和「關係（Relations）」

它透過 `PromptTemplates` 來生成對應的提示詞，並支援使用標準的 OpenAI API
或傳入一個自訂的 LLM 函數，提供了很好的擴充性。
"""

import json
import logging
from typing import List, Dict, Any, Callable
from dataclasses import dataclass, field

# 嘗試導入 openai，如果失敗則設為 None
try:
    import openai
except ImportError:
    openai = None

from config import ModelConfig, PromptConfig

logger = logging.getLogger(__name__)

# --- 資料結構定義 ---

@dataclass
class Entity:
    """定義一個「實體」的資料結構。"""
    entity_name: str
    entity_type: str
    entity_description: str
    source_doc_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.entity_name, self.entity_type))

@dataclass
class Relation:
    """定義一個「關係」的資料結構。"""
    source_entity_name: str
    target_entity_name: str
    relation_name: str
    relation_description: str
    evidence: str  # 來自原文的直接證據
    source_doc_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# --- 提示詞模板生成器 ---

class PromptTemplates:
    """
    根據設定檔，動態生成用於擷取的提示詞。
    """
    def __init__(self):
        # 從單例獲取提示詞設定
        self.prompt_config = PromptConfig.get_instance()

    def get_entity_extraction_prompt(self, text: str) -> Dict[str, str]:
        """獲取實體擷取的系統提示詞和使用者提示詞。"""
        config = self.prompt_config.get_entity_extraction_prompt()
        system_prompt = config.get("system_prompt", "")
        template = config.get("template", "")
        user_prompt = template.format(text=text)
        return {"system": system_prompt, "user": user_prompt}

    def get_relation_extraction_prompt(self, text: str, entity_names: List[str]) -> Dict[str, str]:
        """獲取關係擷取的系統提示詞和使用者提示詞。"""
        config = self.prompt_config.get_relation_extraction_prompt()
        system_prompt = config.get("system_prompt", "")
        template = config.get("template", "")
        user_prompt = template.format(text=text, entity_names=entity_names)
        return {"system": system_prompt, "user": user_prompt}


# --- 核心擷取器 ---

class LLMExtractor:
    """
    使用大型語言模型（LLM）從文本中擷取實體和關係。
    
    這個類別封裝了與 LLM 互動的所有邏輯，包括：
    1. 初始化客戶端（OpenAI 或自訂函數）。
    2. 使用 PromptTemplates 生成提示詞。
    3. 發送請求並解析回傳的結構化資料。
    4. 錯誤處理。
    """
    
    def __init__(self, api_key: str = None, model: str = None, llm_func: Callable = None):
        """
        初始化 LLMExtractor。

        支援兩種模式：
        1. `api_key` 模式：使用標準的 OpenAI API。
        2. `llm_func` 模式：使用一個使用者自訂的函數來調用任何 LLM。

        Args:
            api_key (str, optional): OpenAI 的 API 金鑰。
            model (str, optional): 要使用的模型名稱，預設從 ModelConfig 獲取。
            llm_func (Callable, optional): 一個自訂的 LLM 調用函數，其簽名應為 func(system_prompt: str, user_prompt: str) -> str。
        """
        if llm_func:
            self.llm_call = self._custom_llm_call
            self.llm_func = llm_func
        elif api_key and openai:
            self.client = openai.OpenAI(api_key=api_key)
            self.llm_call = self._openai_call
        else:
            raise ValueError("必須提供 OpenAI `api_key` 或一個自訂的 `llm_func`。如果使用 api_key，請確保 `openai` 套件已安裝。")
            
        self.model = model or ModelConfig.DEFAULT_MODEL
        self.temperature = ModelConfig.TEMPERATURE
        self.prompt_templates = PromptTemplates()

    def _custom_llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """使用自訂的 LLM 函數進行調用。"""
        return self.llm_func(system_prompt, user_prompt)

    def _openai_call(self, system_prompt: str, user_prompt: str) -> str:
        """使用 OpenAI API 進行調用。"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"} # 要求 JSON 輸出
        )
        return response.choices[0].message.content

    def extract(self, text: str, doc_id: str, metadata: Dict[str, Any] = None) -> (List[Entity], List[Relation]):
        """
        從單一段文本中，一次性擷取實體和關係。

        Args:
            text (str): 要從中擷取資訊的文字片段。
            doc_id (str): 來源文件的 ID。
            metadata (Dict, optional): 要附加到實體和關係的元資料。

        Returns:
            Tuple[List[Entity], List[Relation]]: 一個包含實體列表和關係列表的元組。
        """
        prompts = self.prompt_templates.get_entity_extraction_prompt(text)
        
        try:
            response_text = self.llm_call(prompts["system"], prompts["user"])
            # 解析 LLM 回傳的字串，它應該是 "('entity'<|>...)" 的格式
            entities, relations = self._parse_extraction_output(response_text, text, doc_id, metadata or {})
            return entities, relations
        except Exception as e:
            logger.error(f"從文件 ID '{doc_id}' 擷取資訊時發生錯誤: {e}")
            return [], []

    def _parse_extraction_output(self, response_text: str, text: str, doc_id: str, metadata: Dict[str, Any]) -> (List[Entity], List[Relation]):
        """解析來自 LLM 的特殊格式的輸出字串。"""
        entities = []
        relations = []
        items = response_text.strip().split('##')

        for item in items:
            if not item.strip():
                continue
            
            # 去除頭尾的括號
            item_content = item.strip()[1:-1]
            parts = item_content.split('<|>')
            item_type = parts[0].strip('"')

            if item_type == 'entity' and len(parts) == 4:
                entities.append(Entity(
                    entity_name=parts[1].strip('"'),
                    entity_type=parts[2].strip('"'),
                    entity_description=parts[3].strip('"'),
                    source_doc_id=doc_id,
                    metadata=metadata
                ))
            elif item_type == 'relationship' and len(parts) == 5:
                relations.append(Relation(
                    source_entity_name=parts[1].strip('"'),
                    target_entity_name=parts[2].strip('"'),
                    relation_name=parts[3].strip('"'),
                    relation_description=parts[4].strip('"'),
                    evidence=text, # 將整個文本作為證據
                    source_doc_id=doc_id,
                    metadata=metadata
                ))
        return entities, relations
