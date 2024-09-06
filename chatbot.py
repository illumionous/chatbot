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
        print(f"Loaded {len(self.qa_pairs)} QA pairs from {filename}")

    def vectorize(self):
        questions = [qa['question'] for qa in self.qa_pairs]
        self.vectors = self.vectorizer.fit_transform(questions)
        print(f"Vectorized {len(questions)} questions")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"Vector shape: {self.vectors.shape}")

    def search(self, query, top_k=5):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        # 获取相似度排序后的索引
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        seen_concepts = set()
        
        for idx in sorted_indices:
            if len(results) >= top_k:
                break
            
            question = self.qa_pairs[idx]["question"]
            answer = self.qa_pairs[idx]["answer"]
            
            # 提取问题中的概念（假设概念总是在问题的开头）
            concept = question.split()[0] if question else ""
            
            # 如果这个概念还没有被选中，就添加到结果中
            if concept not in seen_concepts:
                results.append({
                    "question": question,
                    "answer": answer,
                    "similarity": float(similarities[idx])
                })
                seen_concepts.add(concept)
        
        return results
    
class ZhipuAIWrapper:
    def __init__(self, api_key):
        self.client = ZhipuAI(api_key=api_key)

    def generate_response(self, query, context):
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        response = self.client.chat.completions.create(
            model="glm-4-0520",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the question."},
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
    raise ValueError("Please set the ZHIPUAI_API_KEY environment variable")
zhipuai_wrapper = ZhipuAIWrapper(zhipuai_api_key)

@app.post("/knowledge/chat")
async def chat(request: ChatRequest):
    # 在知识库中搜索相关问答对
    results = kb.search(request.query, top_k=5)

    # 准备上下文，只使用相似度大于 0.3 的结果
    context = "\n\n".join([f"Q: {r['question']}\nA: {r['answer']}" for r in results])
    print("Retrieved results:")
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Question: {result['question']}")
        print(f"  Answer: {result['answer']}")
        print(f"  Similarity: {result['similarity']}")
    # 生成响应
    response = zhipuai_wrapper.generate_response(request.query, context)

    message_id = str(uuid.uuid4())

    async def stream_response():
        try:
            full_response = ""
            for chunk in response:
                if hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        yield f"data: {json.dumps({'message_id': message_id, 'text': content}, ensure_ascii=False)}\n\n"
            
            yield f"data: {json.dumps({'message_id': message_id, 'retrieved_knowledge': results}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            print(f"Error in stream_response: {str(e)}")
            yield f"data: {json.dumps({'message_id': message_id, 'error': str(e)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(stream_response())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)