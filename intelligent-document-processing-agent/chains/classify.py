from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import yaml

def build_classifier():
    with open("prompts/classify.yaml") as f:
        data = yaml.safe_load(f)
        
    prompt = ChatPromptTemplate.from_template(data['template'])
    
    llm = ChatOpenAI(
        model = 'gpt-4.1-mini',
        temperature = 0.0
    )
    
    return prompt | llm