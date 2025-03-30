from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
import requests
import json
import os

# 设置路径
path = r"./guides/"
# path = r"./"
with np.load(f'{path}guide2num.npz') as data:
    guide2num = dict(data.items())
with np.load(f'{path}dia2guide.npz') as data:
    dia2guide = dict(data.items())


url = "https://9zekn5682505.vicp.fun/v1/chat/completions"
headers = {
    'Content-Type': 'application/json'
}

# 预先加载所有指南内容到内存中
guides_content = {} #存储指南内容
for guide_key, guide_file in guide2num.items():
    file_path = f'{path}{guide_file}.txt'
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            guides_content[guide_key] = f.read()
    else:
        print(f"[警告] 文件 {file_path} 不存在，跳过加载。")


def remove_similar_paragraphs(text):
    paragraphs = re.split(r'\n+', text)
    newparagraphs = []
    for i in paragraphs:
        if len(i) > 0:
            newparagraphs.append(i)
    paragraphs = newparagraphs
    vectorizer = TfidfVectorizer().fit_transform(paragraphs)
    similarity_matrix = cosine_similarity(vectorizer, vectorizer)
    to_keep = []
    for i in range(len(paragraphs)):
        flag = 1
        for j in to_keep:
            if similarity_matrix[i, j] > 0.25:
                flag = 0
        if flag:
            to_keep.append(i)
    if len(to_keep) > 1 and re.match(r'[^\w\s]', paragraphs[to_keep[-1]][-1]) is None:
        cleaned_paragraphs = [paragraphs[i] for i in range(len(paragraphs)) if i in to_keep[:-1]]
    else:
        cleaned_paragraphs = [paragraphs[i] for i in range(len(paragraphs)) if i in to_keep]
    text = "\n".join(cleaned_paragraphs)
    return text


app = Flask(__name__)
app.json.ensure_ascii = False


@app.route('/api', methods=['POST'])
def api_endpoint():
    if request.method == 'POST':
        data = request.json
        diagnosis = data.get('diagnosis', '')  # 获取诊断
        guide = data.get('guide', '')  # 获取指南
        question = data.get('question', '')  # 获取用户输入内容

        # 使用预加载的指南内容
        guide_content = guides_content.get(guide, '')

        if guide_content:
            # 生成请求内容
            content = ''
            content += 'Step 1. 你是一名康复医疗领域的专业医师；\n\n'
            content += 'Step 2. 现有一名患者或其家属来咨询，你拥有一份诊疗指南作为参考，请根据指南内容对患者的情况进行分析，解答患者问题；\n\n'
            content += f'Step 3. 你所参考的诊疗指南内容如下：\n{guide_content}\n'
            content += f'Step 4. 患者描述如下：\n{question}\n\n'
            
            if diagnosis in dia2guide:
                content += f'Step 5. 考虑患者可能的诊断为{diagnosis}，请结合诊疗指南内容，针对患者的实际情况，依次进行：\n一.分析患者病情并得出最终诊断，\n二.给出具体的治疗措施或进一步检查方案。'
            else:
                content += 'Step 5. 请结合诊疗指南内容，针对患者的实际情况，依次进行：\n一.分析患者病情并得出最终诊断，\n二.给出具体的治疗措施或进一步检查方案。'

            # 发送请求到大模型
            payload = json.dumps({
                "model": "BUPTmodel-70B",
                "messages": [
                    {"role": "user", "content": content}
                ],
                "temperature": 0.3,
                "max_tokens": 768,
                "top_p": 0.95
            })
            response = requests.post(url, headers=headers, data=payload)
            data = response.json()
            answer = data['choices'][0]['message']['content']
            answer = remove_similar_paragraphs(answer)

            return jsonify({"question": question, "diagnosis": diagnosis, "guide_input": answer})
        else:
            return jsonify({"question": question, "diagnosis": diagnosis, "guide_input": []})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
