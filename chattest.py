import requests
import json

def test_knowledge_chat():
    url = "http://localhost:8000/knowledge/chat"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "query": "请问文化大革命是什么？"
    }

    print("Sending request to /knowledge/chat...")
    print(f"Query: {data['query']}")
    print("Response:")

    try:
        response = requests.post(url, json=data, headers=headers, stream=True)
        response.raise_for_status()

        full_response = ""
        retrieved_knowledge = None
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    try:
                        data = json.loads(decoded_line[6:])
                        if "text" in data:
                            full_response += data["text"]
                            print(data["text"], end="", flush=True)
                        elif "retrieved_knowledge" in data:
                            retrieved_knowledge = data["retrieved_knowledge"]
                        elif "error" in data:
                            print(f"\nError: {data['error']}")
                    except json.JSONDecodeError:
                        if decoded_line.strip() == "data: [DONE]":
                            break
                        else:
                            print(f"\nWarning: Could not decode JSON from line: {decoded_line}")

        print("\n\nFull response:")
        print(full_response)

        if retrieved_knowledge:
            print("\nRetrieved Knowledge:")
            for i, item in enumerate(retrieved_knowledge, 1):
                print(f"\nItem {i}:")
                print(f"Question: {item['question']}")
                print(f"Answer: {item['answer']}")
                print(f"Similarity: {item['similarity']:.6f}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_knowledge_chat()