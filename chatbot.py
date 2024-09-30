import logging
from fastapi import FastAPI, Query
from pydantic import BaseModel
from zhipuai import ZhipuAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

class KnowledgeBase:
    def __init__(self):
        self.questions = []
        self.qa_map = {}
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.embeddings = None

    def load_questions(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            self.questions = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(self.questions)} questions from {filename}")
        self.embeddings = self.model.encode(self.questions)

    def load_qa_map(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            exec(f.read(), globals())
            self.qa_map = globals().get('QA_MAP', {})
        logger.info(f"Loaded {len(self.qa_map)} QA pairs from {filename}")

    async def retrieve(self, query: str):
        start_time = time.time()
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        max_similarity = float(np.max(similarities))  # Convert to Python float
        max_index = int(np.argmax(similarities))  # Convert to Python int
        
        if query in self.questions:
            result = self.qa_map.get(query, "")
        elif max_similarity >= 0.92:
            result = self.qa_map.get(self.questions[max_index], "")
        else:
            result = None
        
        end_time = time.time()
        logger.debug(f"Retrieval time: {end_time - start_time:.4f} seconds")
        return result, max_similarity

class ZhipuAIWrapper:
    def __init__(self, api_key):
        self.client = ZhipuAI(api_key=api_key)

    async def generate_response(self, query, context):
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        start_time = time.time()
        response = await self.client.chat.completions.create(
            model="glm-4-0520",
            messages=[
                {"role": "system", "content": "You are a helpful assistant created by ZhipuAI."},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        end_time = time.time()
        logger.debug(f"Initial inference time: {end_time - start_time:.4f} seconds")
        return response

class ChatRequest(BaseModel):
    query: str

kb = KnowledgeBase()
kb.load_questions("questions_only.txt")
kb.load_qa_map("qamap.py")

zhipuai_api_key = "3e6112341c28fb8f79f9a04fcd855881.8kHsVkSV3Z6GuVtA"
zhipuai_wrapper = ZhipuAIWrapper(zhipuai_api_key)

@app.post("/knowledge/chat")
async def chat(request: ChatRequest):
    start_time = time.time()
    result, similarity = await kb.retrieve(request.query)
    
    if result:
        return {"answer": result, "similarity": similarity}
    
    context = "\n\n".join([f"Q: {q}\nA: {kb.qa_map.get(q, '')}" for q in kb.questions[:3]])
    response = await zhipuai_wrapper.generate_response(request.query, context)
    
    async def generate():
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    end_time = time.time()
    logger.debug(f"Total processing time: {end_time - start_time:.4f} seconds")
    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
