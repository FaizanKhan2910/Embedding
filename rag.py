import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
load_dotenv()
my_api_key= os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key. kaha hain bhai  ")
client = Groq(api_key = my_api_key)
groqmodel = "openai/gpt-oss-120b"

 
documents = [
    "Employees receive 24 days of paid annual leave (vacation) per year.",
    
    "Employees receive 12 days of sick leave per year.",
    
    "Employees receive medical health insurance covering hospitalization and medical expenses.",
    
    "Employees receive dental and vision insurance as part of their healthcare benefits.",
    
    "Employees receive Rs 3000 per month for gym membership reimbursement.",
    
    "Employees can claim Rs 2000 per month for home internet expenses.",
    
    "Employees receive a company laptop and necessary accessories for work.",
    
    "Employees can work remotely on Monday and Friday.",
    
    "Employees are required to work from the office on Tuesday, Wednesday and Thursday.",
    
    "Employees receive free lunch, snacks and beverages at the office.",
    
    "Employees receive performance-based bonuses depending on individual and company performance.",
    
    "Employees are eligible for annual salary increments based on performance.",
    
    "Employees can participate in the company's Employee Stock Option Plan (ESOP).",
    
    "Employees receive 16 weeks of paid maternity leave, subject to company policy and applicable laws.",
    
    "Employees receive paid paternity leave according to company policy.",
    
    "Employees can access free online courses and professional certification programs.",
    
    "Employees receive financial assistance for attending technical conferences and workshops.",
    
    "Employees receive Rs 5000 per year for professional learning and development.",
    
    "Employees receive relocation assistance when moving to another city for work.",
    
    "Employees receive transportation allowance for commuting to the office.",
    
    "Employees have a 90-day notice period before leaving the company.",
    
    "Employees are eligible for gratuity according to applicable laws and company policy.",
    
    "Employees receive Provident Fund (PF) contributions as per applicable regulations.",
    
    "Employees can access mental health counselling and wellness programs.",
    
    "Employees receive 10 paid public holidays every year.",
    
    "Employees can take flexible working hours with manager approval.",
    
    "Employees are eligible for a yearly health checkup sponsored by the company.",
    
    "Employees receive company-sponsored team outings and annual celebrations.",
    
    "Employees can claim reimbursement for approved work-related expenses.",
    
    "Employees receive overtime compensation according to company policy and applicable laws."
]
document_embeddings = model.encode(documents)
sys.getsizeof(document_embeddings)
print(sys.getsizeof(document_embeddings))



def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )
def retrieve(qembedding):
    scores = []
    for document, embedding in zip(documents, document_embeddings):
        score = cosine_similarity(qembedding, embedding)
        scores.append((score, document))
    return max(scores, key=lambda item: item[0])

def ask_llm(question, context):
    

    sys_prompt=f"""answer in one line only .Answer only based on this context. do not hallicuniate. Context: {context}"""
    system_message={
        "role": "system",
        "content": sys_prompt

    }
    message={
        "role": "user",
        "content": question
    }
    messages=[system_message, message]
    response=client.chat.completions.create(model=groqmodel, messages=messages)
    answer= response.choices[0].message.content
    return answer 

query = "How much vacation do i get?"
qembedding= model.encode(query)
score, context = retrieve(qembedding)
answer= ask_llm(query , context)
print(answer)
