#!/usr/bin/env python3
"""
TimeRAG API 使用演示

此範例展示了如何初始化系統、索引文件以及提出查詢。
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 添加專案根目錄到 Python 路徑，確保可以找到 timeRAG 模組
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from timerag_system import GraphRAGSystem
from config import QueryParams

# 載入 .env 檔案中的環境變數 (例如 OPENAI_API_KEY)
load_dotenv()

# --- 模型定義 ---
# 在實際應用中，您可以將這些函式放在獨立的檔案中

def gpt_4o_mini_llm(system_prompt: str, user_prompt: str) -> str:
    """使用 OpenAI GPT-4o-mini 模型的自訂 LLM 函式。"""
    import openai
    client = openai.OpenAI()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"調用 GPT-4o-mini 時發生錯誤: {e}")
        return "{}" # 發生錯誤時回傳一個空的 JSON 字串

def openai_embedding_func(text: str) -> list:
    """使用 OpenAI text-embedding-3-small 模型的自訂嵌入函式。"""
    import openai
    client = openai.OpenAI()
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"生成嵌入向量時發生錯誤: {e}")
        return [] # 發生錯誤時回傳空列表


def main():
    """主執行函式"""
    print("--- TimeRAG API 使用演示 ---")
    
    # 檢查 API 金鑰是否存在
    if not os.getenv("OPENAI_API_KEY"):
        print("錯誤: 請在 .env 檔案中設定您的 OPENAI_API_KEY")
        return
    
    # 1. 初始化系統
    # 所有模型和金鑰都在初始化時設定
    print("1. 正在初始化 GraphRAGSystem...")
    rag = GraphRAGSystem(
        llm_func=gpt_4o_mini_llm,
        embedding_func=openai_embedding_func
    )
    print("系統初始化完成。")
    
    # 2. 索引文件
    print("2. 正在索引文件...")
    documents = [
        {"text": "蘋果公司在2023年Q4的iPhone銷量達到8000萬部。", "doc_id": "apple_q4_2023", "metadata": {"quarter": "2023Q4"}},
        {"text": "到了2024年Q1，蘋果的iPhone銷量因新機型發布，增長至9000萬部。", "doc_id": "apple_q1_2024", "metadata": {"quarter": "2024Q1"}},
        {"text": "微軟在2024年Q1的雲端業務收入大幅增長了30%。", "doc_id": "ms_q1_2024", "metadata": {"quarter": "2024Q1"}}
    ]
    
    for doc in documents:
        rag.insert(doc["text"], doc["doc_id"], doc["metadata"])
        print(f"  - 已索引: {doc['doc_id']}")
    
    # 3. 建立時間連結
    # 這是完成圖譜建構的關鍵步驟
    print("\n3. 正在建立時間連結...")
    rag.build_temporal_links()
    print("圖譜建構完成。")
    
    # 4. 提出問題
    question = "蘋果iPhone銷量的趨勢如何？"
    print(f"4. 提出查詢: {question}\n")
    
    # 可選：為本次查詢定義特定參數
    custom_query_params = QueryParams(
        top_k=5,
        similarity_threshold=0.4
    )
    
    result = rag.query(question, query_params=custom_query_params)
    
    # 5. 顯示結果
    print("--- 查詢結果 ---")
    print(f"答案: {result.get('answer')}")
    print(f"\nToken 使用統計: {result.get('token_stats')}")
    print("--- 演示完成 ---")


if __name__ == "__main__":
    main()
