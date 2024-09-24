import csv
import json
import logging
from llama_index.retrievers.bm25 import BM25Retriever
from fastapi import FastAPI
from pydantic import BaseModel
from zhipuai import ZhipuAI
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from fastapi.responses import StreamingResponse
import time
import asyncio

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

class KnowledgeBase:
    def __init__(self):
        self.questions = []
        self.bm25_retriever = None
        self.qa_map = {}

    def load_questions(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            self.questions = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(self.questions)} questions from {filename}")

    def load_qa_map(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.debug(f"Content of {filename}: {content[:500]}...")  # Log the first 500 characters of the file
                exec(content, globals())
                self.qa_map = globals().get('QA_MAP', {})
            logger.info(f"Loaded {len(self.qa_map)} QA pairs from {filename}")
            logger.debug(f"First few QA pairs: {list(self.qa_map.items())[:3]}")
            logger.debug(f"Keys in QA_MAP: {list(self.qa_map.keys())[:10]}")  # Log the first 10 keys
        except Exception as e:
            logger.error(f"Error loading QA_MAP from {filename}: {str(e)}")
            self.qa_map = {}

    def vectorize(self):
        documents = [Document(text=question) for question in self.questions]
        parser = SimpleNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents(documents)
        self.bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
        logger.info(f"Indexed {len(self.questions)} questions using BM25")

    def retrieve(self, query: str):
        start_time = time.time()
        results = self.bm25_retriever.retrieve(query)
        
        if results:
            retrieved_doc = {
                "question": results[0].node.text,
                "answer": self.qa_map.get(results[0].node.text, ""),
                "score": results[0].score
            }
            logger.debug(f"resultsare: {results[0].node.text}")
            logger.debug(f"Retrieved document: Question: {retrieved_doc['question'][:30]}... | Score: {retrieved_doc['score']:.4f}")
            logger.debug(f"Retrieved answer: {retrieved_doc['answer'][:100]}...")  
            
            if not retrieved_doc['answer']:
                logger.warning(f"No answer found in QA_MAP for question: {retrieved_doc['question']}")
        else:
            retrieved_doc = None
            logger.debug("No relevant document found")
        
        end_time = time.time()
        logger.debug(f"Retrieval time: {end_time - start_time:.4f} seconds")
        return retrieved_doc


class ZhipuAIWrapper:
    def __init__(self, api_key):
        self.client = ZhipuAI(api_key=api_key)

    def generate_response(self, query, context):
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        start_time = time.time()
        response = self.client.chat.completions.create(
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
kb.vectorize()

zhipuai_api_key = "3e6112341c28fb8f79f9a04fcd855881.8kHsVkSV3Z6GuVtA"
zhipuai_wrapper = ZhipuAIWrapper(zhipuai_api_key)

@app.post("/knowledge/chat")
async def chat(request: ChatRequest):
    start_time = time.time()
    retrieved_doc = kb.retrieve(request.query)
    
    if retrieved_doc is None:
        return {"answer": "抱歉，我没有找到与您问题相关的信息。您能否换个方式提问，或者提供更多细节？"}
    
    if retrieved_doc['score'] >= 0.9 or retrieved_doc['question'] == request.query:
        logger.debug(f"Found high relevance document: {retrieved_doc['question']} | Score: {retrieved_doc['score']:.4f}")
        logger.debug(f"Returning answer: {retrieved_doc['answer']}")  # 添加这行
        return {"answer": retrieved_doc['answer']}
    
    
    context = f"Q: {retrieved_doc['question']}\nA: {retrieved_doc['answer']}"
    prompt = f"Context:\n{context}\n\nQuestion: {request.query}\n\nAnswer:"
    
    response = await asyncio.to_thread(zhipuai_wrapper.generate_response, request.query, prompt)
    
    async def generate():
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    end_time = time.time()
    logger.debug(f"Total processing time: {end_time - start_time:.4f} seconds")
    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
