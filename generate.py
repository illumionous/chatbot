import json
import random
from itertools import product

# 定义一些历史概念和它们的解释
historical_concepts = {
    "文艺复兴": "14至17世纪在欧洲兴起的文化运动，标志着中世纪向现代社会的过渡。",
    "工业革命": "18世纪中叶开始于英国的技术经济和社会变革，导致了从手工业和农业经济向以机器为基础的工业经济的转变。",
    "启蒙运动": "18世纪欧洲的一场哲学、政治、文化运动，强调理性和个人主义。",
    "法国大革命": "1789年至1799年期间发生在法国的一系列激进的社会和政治动荡。",
    "冷战": "第二次世界大战后美国和苏联之间的地缘政治紧张状态，持续到1991年苏联解体。",
    "文化大革命": "1966年至1976年在中国发生的社会政治运动，对中国社会产生了深远影响。",
    "资本主义": "一种经济体系，其特征是生产资料私有制和以追求利润为目的的市场经济。",
    "社会主义": "一种社会和经济体系，主张生产资料公有制和平等分配社会财富。",
    "民主制": "一种政治制度，人民通过选举的代表或直接参与来行使国家权力。",
    "君主制": "一种政体，国家元首世袭的政治制度。",
    "人权": "每个人与生俱来的基本权利和自由，包括生命权、自由权和平等权等。",
    "工业化": "社会经济从以农业为主向以工业为主转变的过程。",
    "全球化": "世界各国在经济、文化、政治等方面日益紧密联系和相互依存的过程。",
    "帝国主义": "19世纪末20世纪初一些国家通过军事、政治、经济手段扩张海外势力范围的政策。",
    "民族主义": "以民族认同为核心的政治意识形态，强调民族的独立、统一和发展。"
}

question_templates = [
    "什么是{concept}？",
    "{concept}的定义是什么？",
    "请解释一下{concept}这个历史概念。",
    "在历史上，{concept}是指什么？",
    "{concept}在历史中有何重要性？",
    "你能简要介绍一下{concept}吗？",
    "{concept}这个历史概念的主要特点是什么？",
    "历史学家如何定义{concept}？"
]

answer_templates = [
    "{concept}是{explanation}",
    "{concept}的定义是：{explanation}",
    "{concept}这个历史概念指的是{explanation}",
    "在历史上，{concept}指的是{explanation}",
    "{concept}在历史中很重要，它{explanation}这一点体现了它的重要性。",
    "简要来说，{concept}是{explanation}",
    "{concept}的主要特点是{explanation}",
    "历史学家通常将{concept}定义为{explanation}"
]

def generate_qa_pairs():
    qa_pairs = []
    for concept, explanation in historical_concepts.items():
        for q_template, a_template in product(question_templates, answer_templates):
            question = q_template.format(concept=concept)
            answer = a_template.format(concept=concept, explanation=explanation)
            qa_pairs.append({"question": question, "answer": answer})
    return qa_pairs

# 生成问答对
qa_pairs = generate_qa_pairs()

# 如果生成的问答对少于10000个，随机重复一些问答对来达到10000个
if len(qa_pairs) < 10000:
    additional_pairs = random.choices(qa_pairs, k=10000-len(qa_pairs))
    qa_pairs.extend(additional_pairs)

# 打乱问答对的顺序
random.shuffle(qa_pairs)

# 只保留前10000个问答对
qa_pairs = qa_pairs[:10000]

# 将问答对保存为JSON文件
with open("qa_pairs_10000.json", "w", encoding="utf-8") as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

print(f"已生成{len(qa_pairs)}个问答对并保存到qa_pairs_10000.json文件中。")