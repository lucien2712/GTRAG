# TimeRAG 配置文件說明

本目錄包含 TimeRAG 系統的所有配置文件，用戶可以根據需要進行自定義修改。

## 配置文件概覽

### 1. entity_types.json
實體類型定義文件，包含：
- **預定義實體類型**：8種標準實體類型及其描述
- **自定義實體類型**：用戶可在 `custom_types` 區塊添加新的實體類型
- **萃取提示**：每個實體類型的識別提示詞

#### 自定義實體類型範例：
```json
{
  "custom_types": {
    "REGULATION": {
      "name": "REGULATION",
      "description_zh": "法規、政策、合規要求",
      "description_en": "Regulations, policies, compliance requirements",
      "examples": ["GDPR", "SOX", "Basel III"],
      "extraction_hints": [
        "法律法規",
        "政策文件",
        "合規要求"
      ]
    }
  }
}
```

### 2. prompts.py
**企業級英文提示詞模板配置文件**（Python格式），包含：
- **實體萃取提示**：`ENTITY_EXTRACTION` - **自動從 entity_types.json 載入實體類型**
- **關係萃取提示**：`RELATION_EXTRACTION` 
- **查詢理解提示**：`QUERY_UNDERSTANDING`
- **結果總結提示**：`RESULT_SUMMARIZATION`
- **自定義提示**：用戶可在 `CUSTOM_PROMPTS` 字典或檔案底部添加新模板

**🌟 重要特性**：
- **專業英文提示詞**：所有模板使用企業級英文，適合國際化業務場景
- **詳細角色設定**：每個提示詞包含專業角色背景和豐富經驗描述
- **完整執行流程**：Step-by-step 執行過程指導，確保高質量輸出
- **實際範例展示**：包含輸入輸出範例，便於理解和調試
- **嚴格回應限制**：詳細的格式、質量和業務相關性約束
- **自動同步**：prompts.py 會自動從 entity_types.json 讀取實體類型定義

#### 企業級提示詞範例：

**實體萃取提示詞結構**：
```python
# 專業角色設定
"system_prompt": "You are a Senior Financial Document Analyst with expertise in extracting structured business entities..."

# 詳細模板結構
"template": """
# ROLE & CONTEXT
You are a **Senior Financial Document Analyst** with 10+ years of experience...

## TASK OVERVIEW
Extract key business entities from financial text...

## EXECUTION PROCESS
1. **Text Analysis**: Carefully read and analyze the entire text
2. **Entity Identification**: Identify all relevant business entities
3. **Classification**: Classify each entity into predefined categories
...

## EXAMPLES
### Input: "Apple Inc. reported iPhone sales revenue..."
### Expected Output:
```json
{
  "entities": [
    {
      "entity_name": "Apple Inc.",
      "entity_type": "COMPANY",
      "entity_description": "Major technology company...",
    }
  ]
}
```

## RESPONSE CONSTRAINTS
- **Format**: MUST return valid JSON only
- **Entity Names**: Use official, standardized names
- **Quality Score**: Range 0.0-1.0, consider context clarity
...
"""
```

**自定義提示詞範例**：
```python
# 添加專業風險分析提示詞
MY_RISK_ANALYSIS = {
    "system_prompt": "You are a Senior Risk Assessment Analyst with 12+ years of experience...",
    "template": """# ROLE & CONTEXT
    You are a **Senior Risk Assessment Analyst**...
    
    ## RISK CATEGORIES TO ANALYZE
    1. **Market Risk**: Competition, demand fluctuations
    2. **Operational Risk**: Process failures, supply chain issues
    ...
    """,
    "customizable_instructions": [
        "Add new risk categories for specific industries",
        "Customize risk assessment frameworks"
    ]
}

CUSTOM_PROMPTS["risk_analysis"] = MY_RISK_ANALYSIS
```

## 使用方法

### 基本配置載入
```python
from config import EntityTypes, PromptConfig

# 載入實體類型配置
entity_types = EntityTypes.get_instance()
all_types = entity_types.get_all_types()
descriptions = entity_types.get_descriptions()

# 載入提示詞配置  
prompt_config = PromptConfig.get_instance()
entity_prompt = prompt_config.get_entity_extraction_prompt()
```

### 添加自定義實體類型
1. 編輯 `entity_types.json`
2. 在 `custom_types` 區塊添加新類型定義
3. 重啟系統以載入新配置

### 修改提示詞模板
1. 編輯 `prompts.py` 
2. 修改對應的常數（如 `ENTITY_EXTRACTION["template"]`）
3. 可使用 `{variable}` 進行變數替換
4. 支援 Python 多行字串和註解

## 配置驗證

系統會自動驗證配置文件格式：
- **實體名稱長度**：2-100字符
- **描述長度**：10-500字符  
- **置信度範圍**：0.0-1.0
- **JSON格式**：必須是有效的JSON格式

## 故障排除

### 常見錯誤
1. **FileNotFoundError**: 配置文件不存在
   - 確認 `configs/` 目錄存在
   - 檢查文件名是否正確

2. **Python語法錯誤**: prompts.py 文件語法不正確
   - 檢查Python語法（缺少逗號、引號、括號等）
   - 使用IDE或 `python -m py_compile prompts.py` 檢查語法

3. **編碼錯誤**: 中文字符顯示異常
   - 確保文件以UTF-8編碼保存

### 重置配置
如需重置為預設配置：
- `entity_types.json`: 刪除檔案，系統會使用內建預設值
- `prompts.py`: 從 git 恢復原始檔案或重新下載

## 進階配置

### 環境相關配置
```python
# 根據環境載入不同配置
import os
config_env = os.getenv("TIMERAG_ENV", "production") 
# 可創建 prompts_dev.py, prompts_prod.py 等不同環境配置
```

### 動態配置更新
```python
# 運行時重新載入配置（開發時很有用）
prompt_config = PromptConfig.get_instance()
prompt_config.reload_config()  # 重新載入 prompts.py

# 如果修改了 entity_types.json，需要重新載入實體類型
from configs import prompts
prompts.reload_entity_types()  # 重新從 entity_types.json 載入實體類型
```

### Python配置的優勢
- **語法高亮**: IDE提供完整的Python語法支援
- **註解支援**: 可以添加詳細的中文註解說明
- **多行字串**: 使用 `"""` 輕鬆編寫長提示詞
- **變數複用**: 可以定義變數避免重複
- **動態生成**: 支援 Python 邏輯生成配置
- **自動同步**: 實體萃取提示詞自動從 entity_types.json 同步實體類型定義
- **熱重載**: 支援開發時動態重新載入配置

---

**注意**：修改配置文件後，需要重新啟動 TimeRAG 系統以使新配置生效。