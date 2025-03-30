from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
import requests
import json
import os
import time

path = r"./guides/"
with np.load('guide2num.npz') as data:
    guide2num = dict(data.items())
with np.load('dia2guide.npz') as data:
    dia2guide = dict(data.items())
print(len(guide2num))
print(len(dia2guide))
url = "http://localhost:8080/v1/chat/completions"
# url = "https://9zekn5682505.vicp.fun/v1/chat/completions"
headers = {
    'Content-Type': 'application/json'
}
model_name = "BUPTmodel-70B"
maxnum = 90
maxnum2 = 45


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
    print('----------------------------------------')
    print(text)
    paragraphs = re.split(r'[、，。]', text)
    newparagraphs = []
    for i in paragraphs:
        if len(i) > 0:
            newparagraphs.append(i)
    paragraphs = newparagraphs
    if len(paragraphs) > 2:
        vectorizer = TfidfVectorizer().fit_transform(paragraphs)
        similarity_matrix = cosine_similarity(vectorizer, vectorizer)
        to_keep = [0, 1]
        for i in range(2, len(paragraphs)):
            flag = 1
            for j in to_keep[-2:]:
                if similarity_matrix[i, j] > 0.75:
                    flag = 0
                    break
            if flag:
                to_keep.append(i)
        cleaned_paragraphs = [paragraphs[i] for i in range(len(paragraphs)) if i in to_keep]
    else:
        cleaned_paragraphs = paragraphs
    return "，".join(cleaned_paragraphs)


# def remove_similar_paragraphs(text, similarity_threshold=0.8):
#     def calculate_overlap_ratio(str1, str2):
#         set1, set2 = set(str1), set(str2)
#         common_chars = len(set1 & set2)
#         longest_length = max(len(set1), len(set2))
#         return common_chars / longest_length
#
#     paragraphs = re.split(r'\n+', text)
#     newparagraphs = []
#     for i in paragraphs:
#         if len(i) > 0:
#             newparagraphs.append(i)
#     paragraphs = newparagraphs
#     cleaned_paragraphs = paragraphs[:3]
#     for paragraph in paragraphs[3:]:
#         is_similar = False
#         for existing_string in cleaned_paragraphs[-3:]:
#             if calculate_overlap_ratio(paragraph, existing_string) > similarity_threshold:
#                 is_similar = True
#                 break
#         if not is_similar:
#             cleaned_paragraphs.append(paragraph)
#     if len(cleaned_paragraphs) > 1 and re.match(r'[^\w\s]', cleaned_paragraphs[-1][-1]) is None:
#         cleaned_paragraphs = cleaned_paragraphs[:-1]
#     text = "\n".join(cleaned_paragraphs)
#
#     paragraphs = re.split(r'[、，。]', text)
#     newparagraphs = []
#     for i in paragraphs:
#         if len(i) > 0:
#             newparagraphs.append(i)
#     paragraphs = newparagraphs
#     cleaned_paragraphs = paragraphs[:3]
#     for paragraph in paragraphs[3:]:
#         is_similar = False
#         for existing_string in cleaned_paragraphs[-3:]:
#             if calculate_overlap_ratio(paragraph, existing_string) > similarity_threshold:
#                 is_similar = True
#                 break
#         if not is_similar:
#             cleaned_paragraphs.append(paragraph)
#     return "，".join(cleaned_paragraphs)

app = Flask(__name__)
app.json.ensure_ascii = False
CORS(app)


@app.route('/v1/chat/completions', methods=['POST'])
def api_endpoint():
    if request.method == 'POST':
        start_time = time.time()
        data = request.json
        # question = data['question']  # 获取前文得出的诊断
        if '使用四到五个字直接返回这句话的简要主题，不要解释、不要标点、不要语气词、不要多余文本，不要加粗，如果没有主题' in \
                data['messages'][-1]['content']:
            return '新的聊天'
        if len(data['messages']) > 1:
            payload = json.dumps({
                "model": model_name,
                "messages": data['messages'],
                "temperature": 0.3,
                "max_tokens": 1024,
                "top_p": 0.95
            })
            response = requests.post(url, headers=headers, data=payload)
            data = response.json()
            anwser = data['choices'][0]['message']['content']
            print(time.time() - start_time)
            return anwser
        question = data['messages'][-1]['content']
        print(question)
        content = ''
        content = content + 'Step 1. 你是一名康复医疗领域的专业医师；\n\n'
        content = content + 'Step 2. 现有一名患者或其家属来咨询，请你根据患者描述，给出一个最有可能的诊断；\n\n'
        content = content + f'Step 3. 患者描述如下：\n{question}\n\n'
        content = content + 'Step 4. 可能的诊断如下：\n'
        stcontent = content
        num = 1
        allchoices = []
        for dia in dia2guide:
            content = content + f'({num}).' + dia + '\n'
            num += 1
            if num > maxnum:
                content = content + '\nStep 5. 请根据患者的描述，从上述可能的诊断中得出一个最有可能的诊断，如有符合患者情况的，请直接输出对应诊断，无需添加其他内容，若有多种诊断均与患者情况相符，请直接输出最有可能的一个诊断内容，如果上述诊断都与患者情况不符，请输出“不适用”。'
                payload = json.dumps({
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": content}],
                    "temperature": 0.0,
                    "max_tokens": 64
                })
                response = requests.post(url, headers=headers, data=payload)
                data = response.json()
                nowchoice = data['choices'][0]['message']['content']
                nowchoice = re.sub(r'\([0-9]+\)\.', '', nowchoice)
                allchoices = allchoices + nowchoice.split('\n')
                num = 1
                content = stcontent
        if num > 1:
            content = content + '\nStep 5. 请根据患者的描述，从上述可能的诊断中得出一个最有可能的诊断，如有符合患者情况的，请直接输出对应诊断，无需添加其他内容，若有多种诊断均与患者情况相符，请直接输出最有可能的一个诊断内容，如果上述诊断都与患者情况不符，请输出“不适用”。'
            payload = json.dumps({
                "model": model_name,
                "messages": [
                    {"role": "user", "content": content}],
                "temperature": 0.0,
                "max_tokens": 64
            })
            response = requests.post(url, headers=headers, data=payload)
            data = response.json()
            nowchoice = data['choices'][0]['message']['content']
            nowchoice = re.sub(r'\([0-9]+\)\.', '', nowchoice)
            allchoices = allchoices + nowchoice.split('\n')
        cleanchoices = []
        for c in allchoices:
            if c in dia2guide:
                cleanchoices.append(c)
        allchoices = cleanchoices
        if len(allchoices) > 1:
            content = stcontent
            num = 1
            for dia in allchoices:
                content = content + f'({num}).' + dia + '\n'
                num += 1
            content = content + '\nStep 5. 请根据患者的描述，从上述可能的诊断中得出一个最有可能的诊断，如有符合患者情况的，请直接输出对应诊断，无需添加其他内容，若有多种诊断均与患者情况相符，请直接输出最有可能的一个诊断内容，如果上述诊断都与患者情况不符，请输出“不适用”。'
            payload = json.dumps({
                "model": model_name,
                "messages": [
                    {"role": "user", "content": content}],
                "temperature": 0.0,
                "max_tokens": 64
            })
            response = requests.post(url, headers=headers, data=payload)
            data = response.json()
            nowchoice = data['choices'][0]['message']['content']
            nowchoice = re.sub(r'\([0-9]+\)\.', '', nowchoice).split('\n')[0]
            if nowchoice not in dia2guide:
                nowchoice = allchoices[-1]
        elif len(allchoices) > 0:
            nowchoice = allchoices[0]
        else:
            nowchoice = "不适用"
        diagnosis = nowchoice
        print(diagnosis)
        if (diagnosis in dia2guide) and (len(dia2guide[diagnosis]) > 0):
            guide_list = list(dia2guide[diagnosis])
        else:
            guide_list = list(guide2num.keys())

        content = ''
        content = content + 'Step 1. 你是一名康复医疗领域的专业医师；\n\n'
        content = content + 'Step 2. 现有一名患者或其家属来咨询，请你根据患者描述，选择合适的诊疗指南用以辅助对其进行诊断与治疗；\n\n'
        content = content + f'Step 3. 患者描述如下：\n{question}\n\n'
        content = content + 'Step 4. 你可以选择的诊疗指南名称如下：\n'
        stcontent = content
        num = 1
        allchoices = []
        for file in guide_list:
            content = content + f'({num}).' + file + '\n'
            num += 1
            if num > maxnum2:
                if diagnosis == "不适用":
                    content = content + '\nStep 5. 请根据患者的描述，在所给的诊疗指南范围内选出合适的一篇指南辅助医生对患者进行诊断与治疗，如果有合适的指南，直接输出对应诊疗指南的名称，无需添加其他内容，若有多本指南均可用作参考，请选择一本最为合适，直接输出其对应的名称，如果没有合适的指南，请输出“不适用”。'
                else:
                    content = content + f'\nStep 5. 请根据患者的描述，考虑患者的诊断为{diagnosis}，在所给的诊疗指南范围内选出合适的一篇指南辅助医生对患者进行诊断与治疗，如果有合适的指南，直接输出对应诊疗指南的名称，无需添加其他内容，若有多本指南均可用作参考，请选择一本最为合适，直接输出其对应的名称，如果没有合适的指南，请输出“不适用”。'
                payload = json.dumps({
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": content}],
                    "temperature": 0.0,
                    "max_tokens": 256
                })
                response = requests.post(url, headers=headers, data=payload)
                data = response.json()
                nowchoice = data['choices'][0]['message']['content']
                nowchoice = re.sub(r'\([0-9]+\)\.', '', nowchoice)
                allchoices = allchoices + nowchoice.split('\n')
                num = 1
                content = stcontent
        if num > 1:
            if diagnosis == "不适用":
                content = content + '\nStep 5. 请根据患者的描述，在所给的诊疗指南范围内选出合适的一篇指南辅助医生对患者进行诊断与治疗，如果有合适的指南，直接输出对应诊疗指南的名称，无需添加其他内容，若有多本指南均可用作参考，请选择一本最为合适，直接输出其对应的名称，如果没有合适的指南，请输出“不适用”。'
            else:
                content = content + f'\nStep 5. 请根据患者的描述，考虑患者的诊断为{diagnosis}，在所给的诊疗指南范围内选出合适的一篇指南辅助医生对患者进行诊断与治疗，如果有合适的指南，直接输出对应诊疗指南的名称，无需添加其他内容，若有多本指南均可用作参考，请选择一本最为合适，直接输出其对应的名称，如果没有合适的指南，请输出“不适用”。'
            payload = json.dumps({
                "model": model_name,
                "messages": [
                    {"role": "user", "content": content}],
                "temperature": 0.0,
                "max_tokens": 256
            })
            response = requests.post(url, headers=headers, data=payload)
            data = response.json()
            nowchoice = data['choices'][0]['message']['content']
            nowchoice = re.sub(r'\([0-9]+\)\.', '', nowchoice)
            allchoices = allchoices + nowchoice.split('\n')
            num = 1
            content = stcontent
        cleanchoices = []
        for c in allchoices:
            if c in guide_list:
                cleanchoices.append(c)
        allchoices = cleanchoices
        if len(allchoices) > 1:
            content = stcontent
            num = 1
            for dia in allchoices:
                content = content + f'({num}).' + dia + '\n'
                num += 1
            if diagnosis == "不适用":
                content = content + '\nStep 5. 请根据患者的描述，在所给的诊疗指南范围内选出合适的一篇指南辅助医生对患者进行诊断与治疗，如果有合适的指南，直接输出对应诊疗指南的名称，无需添加其他内容，若有多本指南均可用作参考，请选择一本最为合适，直接输出其对应的名称，如果没有合适的指南，请输出“不适用”。'
            else:
                content = content + f'\nStep 5. 请根据患者的描述，考虑患者的诊断为{diagnosis}，在所给的诊疗指南范围内选出合适的一篇指南辅助医生对患者进行诊断与治疗，如果有合适的指南，直接输出对应诊疗指南的名称，无需添加其他内容，若有多本指南均可用作参考，请选择一本最为合适，直接输出其对应的名称，如果没有合适的指南，请输出“不适用”。'
            payload = json.dumps({
                "model": model_name,
                "messages": [
                    {"role": "user", "content": content}],
                "temperature": 0.0,
                "max_tokens": 256
            })
            response = requests.post(url, headers=headers, data=payload)
            data = response.json()
            nowchoice = data['choices'][0]['message']['content']
            nowchoice = re.sub(r'\([0-9]+\)\.', '', nowchoice).split('\n')[0]
            if nowchoice not in guide_list:
                nowchoice = allchoices[0]
        elif len(allchoices) > 0:
            nowchoice = allchoices[0]
        else:
            nowchoice = "不适用"

        guide = nowchoice
        print(guide)
        print('------------------------------------------------------------')

        if guide in guide2num:
            guide = guide2num[guide]
            # 若需要选取指南，则进行一轮对话提取指南内容
            with open(f'{path}{guide}.txt', 'r', encoding='utf-8') as f:
                guide_content = f.read()
            content = ''
            content = content + 'Step 1. 你是一名康复医疗领域的专业医师；\n\n'
            content = content + 'Step 2. 现有一名患者或其家属来咨询，你拥有一份诊疗指南作为参考，请根据指南内容对患者的情况进行分析，解答患者问题；\n\n'
            content = content + f'Step 3. 你所参考的诊疗指南内容如下：\n{guide_content}\n'
            content = content + f'Step 4. 患者描述如下：\n{question}\n\n'
            if diagnosis in dia2guide:
                content = content + f'Step 5. 考虑患者可能的诊断为{diagnosis}，请结合诊疗指南内容，针对患者的实际情况，依次进行：\n一.分析患者病情并得出最终诊断，\n二.给出具体的治疗措施或进一步检查方案。'
            else:
                content = content + f'Step 5. 请结合诊疗指南内容，针对患者的实际情况，依次进行：\n一.分析患者病情并得出最终诊断，\n二.给出具体的治疗措施或进一步检查方案。'
            payload = json.dumps({
                "model": "BUPTmodel-70B",
                "messages": [
                    {"role": "user", "content": content}],
                "temperature": 0.3,
                "max_tokens": 768,
                "top_p": 0.95
            })
            response = requests.post(url, headers=headers, data=payload)  # 获取大模型输出，这一块可能得改一改？我的url直接用的本地的
            data = response.json()
            anwser = data['choices'][0]['message']['content']
            print(anwser)
            anwser = remove_similar_paragraphs(anwser)
            guide_input = anwser
            print('------------------------------------------------------------')
            print(anwser)
        else:
            guide_input = []
            # print('跳过')
        if len(guide_input) > 0:
            content = ''
            content = content + 'Step 1. 你是一名康复医疗领域的专业医师；\n\n'
            content = content + 'Step 2. 现有一名患者或其家属来咨询，我将给你一份初步诊断与建议作为参考，请根据相关内容对患者的情况进行分析，解答患者问题；\n\n'
            content = content + f'Step 3. 患者描述如下：\n{question}\n\n'
            content = content + f'Step 4. 初步诊断与建议如下：\n{guide_input}\n'
            if diagnosis in dia2guide:
                content = content + f'Step 5. 考虑患者可能的诊断为{diagnosis}，请针对患者实际情况，结合初步诊断与建议中的内容给出具体详细的答复，你的回答要符合康复医疗领域的专业医师的身份，若患者有明确提出问题，请直接对所提问题进行回答。'
            else:
                content = content + f'Step 5. 请针对患者实际情况，结合初步诊断与建议中的内容给出具体详细的答复，你的回答要符合康复医疗领域的专业医师的身份，若患者有明确提出问题，请直接对所提问题进行回答。'
            payload = json.dumps({
                "model": model_name,
                "messages": [
                    {"role": "user", "content": content}],
                "temperature": 0.3,
                "max_tokens": 1024,
                "top_p": 0.95
            })
            response = requests.post(url, headers=headers, data=payload)
            data = response.json()
            anwser = data['choices'][0]['message']['content']
        else:
            if diagnosis not in dia2guide:
                payload = json.dumps({
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": question}],
                    "temperature": 0.3,
                    "max_tokens": 1024,
                    "top_p": 0.95
                })
                response = requests.post(url, headers=headers, data=payload)
                data = response.json()
                anwser = data['choices'][0]['message']['content']
            else:
                content = ''
                content = content + 'Step 1. 你是一名康复医疗领域的专业医师；\n\n'
                content = content + f'Step 2. 现有一名患者前来咨询，患者描述如下；\n{question}\n\n'
                content = content + f'Step 3. 考虑患者可能的诊断为{diagnosis}，请对患者的情况进行分析，对患者问题进行解答。'
                payload = json.dumps({
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": content}],
                    "temperature": 0.3,
                    "max_tokens": 1024,
                    "top_p": 0.95
                })
                response = requests.post(url, headers=headers, data=payload)
                data = response.json()
                anwser = data['choices'][0]['message']['content']
        print(time.time() - start_time)
        return anwser


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)
