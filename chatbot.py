import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from zhipuai import ZhipuAI
import uuid
import asyncio
import jieba

app = FastAPI()

class KnowledgeBase:
    def __init__(self):
        self.qa_pairs = []
        self.vectorizer = TfidfVectorizer(tokenizer=self.chinese_tokenizer, token_pattern=None)
        self.vectors = None

    def chinese_tokenizer(self, text):
        return list(jieba.cut(text))

    def load_qa_pairs(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            self.qa_pairs = json.load(f)
        print(f"从 {filename} 加载了 {len(self.qa_pairs)} 个 QA 对")

    def vectorize(self):
        questions = [qa['question'] for qa in self.qa_pairs]
        self.vectors = self.vectorizer.fit_transform(questions)
        print(f"向量化了 {len(questions)} 个问题")
        print(f"词汇表大小: {len(self.vectorizer.vocabulary_)}")
        print(f"向量形状: {self.vectors.shape}")

    def search(self, query, top_k=5):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        # 获取相似度排序后的索引
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        seen_questions = set()
        
        for idx in sorted_indices:
            if len(results) >= top_k:
                break
            
            question = self.qa_pairs[idx]["question"]
            answer = self.qa_pairs[idx]["answer"]
            
            # 检查是否已经有相同的问题
            if question not in seen_questions:
                results.append({
                    "question": question,
                    "answer": answer,
                    "similarity": float(similarities[idx])
                })
                seen_questions.add(question)
        
        return results

class ZhipuAIWrapper:
    def __init__(self, api_key):
        self.client = ZhipuAI(api_key=api_key)

    def generate_response(self, query, context):
        prompt = f"上下文:\n{context}\n\n问题: {query}\n\n回答:"

        response = self.client.chat.completions.create(
            model="glm-4-0520",
            messages=[
                {"role": "system", "content": "你是一个有帮助的助手。使用提供的上下文来回答问题。"},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )

        return response

class ChatRequest(BaseModel):
    query: str

# 初始化知识库和ZhipuAI
kb = KnowledgeBase()
kb.load_qa_pairs("qa_pairs_10000.json")
kb.vectorize()

zhipuai_api_key = "3e6112341c28fb8f79f9a04fcd855881.8kHsVkSV3Z6GuVtA"
if not zhipuai_api_key:
    raise ValueError("请设置 ZHIPUAI_API_KEY 环境变量")
zhipuai_wrapper = ZhipuAIWrapper(zhipuai_api_key)

@app.post("/knowledge/chat")
async def chat(request: ChatRequest):
    # 在知识库中搜索相关问答对
    results = kb.search(request.query, top_k=5)

    # 准备上下文，只使用相似度大于 0.3 的结果
    context = "\n\n".join([f"Q: {r['question']}\nA: {r['answer']}" for r in results if r['similarity'] > 0.3])

    # 生成响应
    response = zhipuai_wrapper.generate_response(request.query, context)

    message_id = str(uuid.uuid4())

    async def stream_response():
        try:
            for chunk in response:
                if hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        yield f"data: {json.dumps({'message_id': message_id, 'text': content}, ensure_ascii=False)}\n\n"
            
            # 在流式响应的最后发送检索到的知识
            yield f"data: {json.dumps({'message_id': message_id, 'retrieved_knowledge': results}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            print(f"流式响应中出现错误: {str(e)}")
            yield f"data: {json.dumps({'message_id': message_id, 'error': str(e)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
