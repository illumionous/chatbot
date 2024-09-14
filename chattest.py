import requests
import json
import sys

def test_knowledge_chat():
    url = "http://localhost:8000/knowledge/chat"
    query = "请问文化大革命是什么？"

    print("发送请求到 /knowledge/chat...")
    print(f"问题: {query}")
    print("回答:")

    try:
        response = requests.get(url, params={"query": query}, stream=True)
        response.raise_for_status()

        full_response = ""
        retrieved_knowledge = None

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    event_data = json.loads(decoded_line[6:])
                    if "text" in event_data:
                        sys.stdout.write(event_data["text"])
                        sys.stdout.flush()
                        full_response += event_data["text"]
                    elif "retrieved_knowledge" in event_data:
                        retrieved_knowledge = event_data["retrieved_knowledge"]
                elif decoded_line.startswith("event: close"):
                    break
                elif decoded_line.startswith("event: error"):
                    error_data = json.loads(next(response.iter_lines()).decode('utf-8')[6:])
                    print(f"\n错误: {error_data['error']}")
                    break

        print("\n\n完整回答:")
        print(full_response)

        if retrieved_knowledge:
            print("\n检索到的知识:")
            for i, item in enumerate(retrieved_knowledge, 1):
                print(f"\n项目 {i}:")
                print(f"问题: {item['question']}")
                print(f"答案: {item['answer']}")
                print(f"相似度: {item['similarity']:.6f}")

    except requests.exceptions.RequestException as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    test_knowledge_chat()
