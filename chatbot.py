import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from fastapi import FastAPI, Query
from pydantic import BaseModel
from zhipuai import ZhipuAI
import uuid
import jieba
from fastapi.responses import StreamingResponse
import asyncio
import re
import time

DEBUG = 1
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
        
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        seen_questions = set()
        
        for idx in sorted_indices:
            if len(results) >= top_k:
                break
            
            question = self.qa_pairs[idx]["question"]
            answer = self.qa_pairs[idx]["answer"]
            
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

    async def generate_response(self, query, context):
        prompt = f"上下文:\n{context}\n\n问题: {query}\n\n回答:"

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="glm-4-0520",
                messages=[
                    {"role": "system", "content": "你是一个有帮助的助手,来自九天大模型。使用提供的上下文来回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            return response
        except Exception as e:
            print(f"ZhipuAI API error: {str(e)}")
            return None

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
    search_start = time.time()
    results = kb.search(request.query, top_k=5)
    search_end = time.time()
    search_time = search_end - search_start
    
    context = "\n\n".join([f"Q: {r['question']}\nA: {r['answer']}" for r in results if r['similarity'] > 0.5])

    prompt = f"""基于以下上下文回答问题。请直接给出答案，不要重复问题。

上下文:
{context}

问题: {request.query}

回答:"""

    inference_start = time.time()
    response = await zhipuai_wrapper.generate_response(request.query, prompt)
    inference_end = time.time()
    inference_time = inference_end - inference_start

    message_id = str(uuid.uuid4())

    if DEBUG:
        print(f"检索到的知识:")
        for r in results:
            if r['similarity'] > 0.5:
                print(f"问题: {r['question']}")
                print(f"答案: {r['answer']}")
                print(f"相似度: {r['similarity']}")
                print()
        print(f"搜索时间: {search_time:.4f}秒")
        print(f"推理时间: {inference_time:.4f}秒")
        print(f"总时间: {search_time + inference_time:.4f}秒")

    async def stream_response():
        try:
            if response is None:
                fallback_response = "抱歉，我现在无法连接到AI服务。根据知识库，以下是一些相关信息：\n\n"
                for r in results[:3]:
                    fallback_response += f"- {r['answer']}\n"
                fallback_response += "\n希望这些信息对您有帮助。如果您有更具体的问题，请随时告诉我。"
                yield f"data: {json.dumps({'message_id': message_id, 'text': fallback_response}, ensure_ascii=False)}\n\n"
            else:
                full_response = ""
                for chunk in response:
                    if hasattr(chunk.choices[0].delta, 'content'):
                        content = chunk.choices[0].delta.content
                        if content:
                            full_response += content
                
               
                full_response = re.sub(r'\n{3,}', '\n\n', full_response)
                full_response = re.sub(r'(^|\n)- ', '\n• ', full_response)
                
                yield f"data: {json.dumps({'message_id': message_id, 'text': full_response.strip()}, ensure_ascii=False)}\n\n"
        
        except Exception as e:
            print(f"流式响应中出现错误: {str(e)}")
            yield f"data: {json.dumps({'message_id': message_id, 'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
