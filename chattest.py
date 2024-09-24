import aiohttp
import asyncio
import json

async def chat(url, query):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={"query": query}) as response:
            if response.status != 200:
                print(f"Error: Received status code {response.status}")
                return

            content_type = response.headers.get('content-type', '')
            if 'application/json' in content_type:
                data = await response.json()
                print(f"Answer: {data['answer']}")
            elif 'text/plain' in content_type:
                async for chunk in response.content:
                    print(chunk.decode(), end='', flush=True)
            else:
                print(f"Unexpected content type: {content_type}")

async def main():
    url = "http://localhost:8000/knowledge/chat"
    query = "请介绍一下中国移动的AI审计能力,可以吗"

    print(f"Sending query: {query}")
    await chat(url, query)

if __name__ == "__main__":
    asyncio.run(main())
