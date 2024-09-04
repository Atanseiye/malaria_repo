import openai
import re
import httpx
import os
import streamlit as st
import pickle as pkl
from joblib import load
import sklearn
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
import json



# -----------------------------------------------------------------------------------------------
# Encode
import sys
import json
# Define the maximum number of executions
MAX_EXECUTIONS = 3

# Define a function to load the expiration date from the JSON file
def load_expiration_date(file_path="config.json"):
    with open(file_path, "r") as file:
        config = json.load(file)
        expiration_date_str = config["timer_date"]
        return datetime.fromisoformat(expiration_date_str)

from datetime import datetime

# Define the cut-off date
# Load the expiration date from the external JSON file
CUTOFF_DATE = load_expiration_date()

def is_within_allowed_date():
    current_date = datetime.now()
    return current_date > CUTOFF_DATE

# Check if the current date is within the allowed date
# if is_within_allowed_date():
    # st.error("This app has expired and can no longer be accessed after 5 August 2024.")
# else:
    # Increment the counter and write it back to the file
    # pass


# -----------------------------------------------------------------------------------------------


chat = ChatOpenAI(temperature=0.5)
from dotenv import load_dotenv

from openai import OpenAI

openai.api_key = os.getenv("OPENAI_API_KEY")

# initialise the OpenAI client
client = OpenAI()

# Create a class -Agent- to handle the LLMs
class Agent:
    def __init__(self, system=''):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role":'system', 'content':system})
            
            
    def __call__(self, message):
        self.messages.append({'role':'user', 'content':message})
        result = self.execute()
        self.messages.append({'role':'assistant', 'content':result})
        return result
    
    def execute(self):
        chat_completion = client.chat.completions.create(
            model='gpt-4o',
            temperature=0,
            messages = self.messages
        )
        
        return chat_completion.choices[0].message.content
    
# Open the documents
with open('System prompt.txt') as file:
    system_prompt = file.read()

with open('Health Recommendation Doc.txt') as file:
    health_prompt = file.read()
    

# Function to Process the text coming from the LLM to extract the JSON
def extract_users_details(response):
    import json as js
    import re
    pattern = r"\{(.*?)\}"
    
    match = re.search(pattern, response, re.DOTALL)
    
    if match:
        json_str = match.group()
        json_obj = js.loads(json_str)
        return json_obj
    # else:
    #     return False 

# Function to call the built model and makes prediction with it
def predict(llm_json):
    import pickle as pkl
    import numpy as np

    
    job_model = load('Job_model.joblib')
    pkl_ = load('model.pkl')
        
    # get the users data
    users_details = extract_users_details(llm_json)
    input_data = np.array([list(users_details.values())])

    # Do prediction
    prediction = job_model.predict(input_data)
    
    # return prediction
    return prediction


# Function to get the user's data
def get_data(users_complain):
    interact = Agent(system_prompt)
    get_complain = interact(users_complain)
    return get_complain

def recommendation(prediction):
    prescribe = Agent(health_prompt)
    input_ = f"The prediction made by the model from the patient's data is {prediction}, can you give a recommendation based on this?"
    give_prescription = prescribe(input_)
    st.write(give_prescription)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Function to extract JSON from the string
def extract_json_from_string(s):
    start = s.find("") + 3
    end = s.rfind("")
    if start != -1 and end != -1:
        json_str = s[start:end]
        return json.loads(json_str.strip())
    return None

# Given string
input_string = "Here is the results \njson\n{'name':'Kolade'}\nYou can go ahead for further processing."





# ------------Streamlite Code -------------------
st.title('Maleria Test and Recommendation System')
if 'input' not in st.session_state:
    st.session_state['input'] = [
        SystemMessage(content=system_prompt)
    ]

users_complain = st.text_area('Please, give your complain here')

run = st.button('Check')
if run:
    
    complain = users_complain
    st.session_state['input'].append(HumanMessage(content=users_complain))
    response = chat(st.session_state['input'])
    st.session_state['input'].append(AIMessage(content=response.content))


    # st.write(extract_users_details(response.content))


    # st.write(extract_users_details(users_complain))



   extracted_details = extract_users_details(response.content)
    if extracted_details:
        prediction = predict(response.content)
        if prediction == 1:
            st.write(f'Model\'s prediction is: Malaria')
        else:
            st.write(f'Model\'s prediction is: No Malaria')
        recommend = recommendation(predict(response.content))
        st.write(recommend)
    else:
        st.write(response.content)
  

            

       
    
    # recommend = recommendation(predict(st.session_state['input']))
    # st.write(recommend)
