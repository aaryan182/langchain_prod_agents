from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import yaml

def build_analyzer_chain():
    with open("prompts/analyze.yaml") as f:
        prompt_data = yaml.safe_load(f)
        
    prompt = ChatPromptTemplate.from_template(prompt_data['template'])
    
    llm = ChatOpenAI(
        model= 'gpt-4.1-mini',
        temperature= 0.0
    )
    
    return prompt | llm