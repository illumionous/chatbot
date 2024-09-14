import requests
import json
import sys
import time
def test_knowledge_chat():
    url = "http://localhost:8000/knowledge/chat"
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    data = {
        "query": "请问审计确认单需要哪些材料附件？"
    }

    print("发送请求到 /knowledge/chat...")
    print(f"问题: {data['query']}")
    print("回答:")

    try:
        response = requests.post(url, json=data, headers=headers, stream=True)
        response.raise_for_status()

        full_response = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    try:
                        event_data = json.loads(decoded_line[6:])
                        if "text" in event_data:
                            text = event_data["text"]
                            sys.stdout.write(text)
                            sys.stdout.flush()
                            time.sleep(0.05)  # 添加小延迟，使输出更容易观察
                            full_response += text
                        elif "echarts" in event_data:
                            print("\n\n图表数据:")
                            print(event_data["echarts"])
                        elif "error" in event_data:
                            print(f"\n错误: {event_data['error']}")
                    except json.JSONDecodeError:
                        print(f"\n警告: 无法解析JSON: {decoded_line}")

        print("\n\n完整回答:")
        print(full_response)

    except requests.exceptions.RequestException as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    test_knowledge_chat()
