# import requests
# import json

# # API 地址
# url = "http://localhost:5001/api"  # 使用 Step1.5.py 的端口 5001

# # 测试数据：患者描述和诊断
# test_data = {
#     "question": "患者感觉胸部持续性疼痛，偶尔会出现呼吸困难，已经持续三天。",
#     "diagnosis": "不适用"  # 如果诊断是“未确定”，则API会尝试生成引导性问题
# }

# # 发送POST请求
# response = requests.post(url, json=test_data)

# # 检查响应
# if response.status_code == 200:
#     response_data = response.json()
#     print("引导性问题:", response_data.get("guidance_question"))
# else:
#     print(f"请求失败，状态码: {response.status_code}")
import requests
import json

# API 地址
url_step1 = "http://localhost:5000/api"  # Step1 服务地址
url_step1_5 = "http://localhost:5001/api"  # Step1.5 服务地址

# 测试数据：患者描述和诊断
test_data = {
    "question": "患者感觉胸部持续性疼痛，偶尔会出现呼吸困难，已经持续三天。",
    "diagnosis": "不适用"  # 如果诊断是“未确定”，则API会尝试生成引导性问题
}

# Step 1: 向 Step1.py 请求诊断
response_step1 = requests.post(url_step1, json=test_data)

if response_step1.status_code == 200:
    result_step1 = response_step1.json()
    print("Step1 诊断结果:", result_step1)
    guidance_question = result_step1.get("guidance_question", "")
    
    # 如果 Step1 返回了引导性问题
    if guidance_question:
        print("引导性问题:", guidance_question)
        
        # Step 2: 用户回答引导性问题，并传回 Step1.py 进行第二轮判断
        test_data["question"] += "\n" + guidance_question
        response_step2 = requests.post(url_step1, json=test_data)
        
        if response_step2.status_code == 200:
            result_step2 = response_step2.json()
            print("Step2 诊断结果:", result_step2)
            if result_step2.get("guidance_question") != "不需要额外引导问题":
                print("第二轮引导性问题:", result_step2.get("guidance_question"))
                
                # Step 3: 用户回答第二轮问题，并传回 Step1.py 进行第三轮判断
                test_data["question"] += "\n" + result_step2.get("guidance_question")
                response_step3 = requests.post(url_step1, json=test_data)
                if response_step3.status_code == 200:
                    result_step3 = response_step3.json()
                    print("Step3 诊断结果:", result_step3)
else:
    print(f"请求失败，状态码: {response_step1.status_code}")
