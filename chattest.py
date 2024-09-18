import aiohttp
import json
import sys
import asyncio
import time

# 设置调试模式（0：关闭，1：开启）
DEBUG = 1

async def parse_sse(response):
    buffer = ""
    async for line in response.content:
        line = line.decode('utf-8')
        if line.startswith('data:'):
            buffer += line[5:]
        elif line.strip() == '' and buffer:
            yield buffer.strip()
            buffer = ""
    if buffer:
        yield buffer.strip()

async def print_gradually(text, delay=0.01):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        await asyncio.sleep(delay)
    sys.stdout.write('\n')

async def test_knowledge_chat():
    url = "http://localhost:8000/knowledge/chat"
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    data = {
        "query": "请问审计有哪些规则？"
    }

    print("发送请求到 /knowledge/chat...")
    print(f"问题: {data['query']}")
    print("回答:")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                response.raise_for_status()
                async for event_data in parse_sse(response):
                    try:
                        event_json = json.loads(event_data)
                        if "text" in event_json:
                            await print_gradually(event_json["text"])
                        elif "error" in event_json:
                            print(f"\n错误: {event_json['error']}")
                    except json.JSONDecodeError:
                        print(f"\n警告: 无法解析JSON: {event_data}")

    except aiohttp.ClientError as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    asyncio.run(test_knowledge_chat())
