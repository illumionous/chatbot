项目包含以下主要文件：

generate.py: 用于生成历史概念的问答对
chatbot.py: 实现了知识库和问答功能的主要服务器
chattest.py: 用于测试问答系统的客户端脚本

依赖
本项目依赖以下Python库：

numpy
scikit-learn
fastapi
pydantic
zhipuai
jieba

使用以下命令安装这些依赖：
pip install numpy scikit-learn fastapi pydantic zhipuai jieba sse-starlette
使用方法

生成问答对：
运行 generate.py 脚本来生成问答对并保存到 qa_pairs_10000.json 文件中：
python generate.py

启动服务器：
运行 chatbot.py 脚本来启动问答服务器：
python chatbot.py

测试系统：
运行 chattest.py 脚本来测试问答系统：
python chattest.py
