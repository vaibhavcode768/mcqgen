import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging
import streamlit as st


#imporing necessary packages packages from langchain
from langchain_ai21 import ChatAI21
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain


# Load environment variables from the .env file
load_dotenv()

# Access the environment variables just like you would with os.environ
key=os.getenv("AI21_API_KEY")


llm = ChatAI21(api_key = key, model="jamba-large", temperature=0.7)

template="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}

"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=template)


quiz_chain=LLMChain(llm=llm,prompt=quiz_generation_prompt,output_key="quiz",verbose=True)


template2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""


quiz_evaluation_prompt=PromptTemplate(input_variables=["subject", "quiz"], template=template2)

review_chain=LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)


# This is an Overall Chain where we run the two chains in Sequence

generate_evaluate_chain=SequentialChain(chains=[quiz_chain, review_chain], input_variables=["text", "number", "subject", "tone", "response_json"],
                                        output_variables=["quiz", "review"], verbose=True)


with open("C:/Users/jainv/mcqgen/experiment/response.json", 'r') as file:
    RESPONSE_JSON = json.load(file)


#creating a title for the app
st.title("MCQs Creator Application with LangChain 🦜⛓️")

#Create a form using st.form
with st.form("user_inputs"):

    #File Upload
    uploaded_file = st.file_uploader("Upload a PDF or txt file")

    #Input Fields
    mcq_count = st.number_input("No. of MCQs", min_value=3, max_value=50)
    subject = st.text_input("Subject", max_chars=20)
    tone = st.text_input("Complexity Level Of Questions", max_chars=20, placeholder="Simple")
    
# Add Button
    button = st.form_submit_button("Create MCQs")

# Check if the button is clicked and all fields have input
    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("looking..."):
            try:
                text = read_file(uploaded_file)
                response = generate_evaluate_chain(
                    {
                "text": text,
                "number": mcq_count,
                "subject":subject,
                "tone": tone,
                "response_json": json.dumps(RESPONSE_JSON)
                    }
                )

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error")

            else:
                print("Task Completed")
                if isinstance(response, dict):
        # Extract the quiz data from the response
                    quiz = response.get("quiz")
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1
                            st.table(df)
                # Display the review in a text box as well
                            st.text_area(label="Review", value=response["review"])
                        else:
                            st.error("Error in the table data")
    
                else:
                    st.write(response)

