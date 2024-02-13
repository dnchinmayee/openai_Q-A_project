import os
import json
import traceback
import pandas as pd
from src.mcqgenrator.utils import read_file,get_table_data
import streamlit as st
from langchain.callbacks import get_openai_callback
from src.mcqgenrator.mcqgenerator import generate_evaluate_chain
from src.mcqgenrator.logger import logging

uploaded_file = './data.txt'
mcq_count = 2
subject = "Astronomy"
tone = "positive"

#loading json file
with open('Response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)
    logging.info("Response.json loaded: "+str(RESPONSE_JSON))

    try:
        with open(uploaded_file, 'r') as fh:
            text=read_file(fh)
            #Count tokens and the cost of API call
            logging.info("Text: "+str(text))
            with get_openai_callback() as cb:
                response=generate_evaluate_chain(
                    {
                        "text": text,
                        "number": mcq_count,
                        "subject":subject,
                        "tone": tone,
                        "RESPONSE_JSON": json.dumps(RESPONSE_JSON)
                    }
                )
            #st.write(response)
                logging.info("Response: "+str(response))

    except Exception as e:
        logging.error("Error: "+str(e))

    else:
        logging.info(f"Total Tokens:{cb.total_tokens}")
        logging.info(f"Prompt Tokens:{cb.prompt_tokens}")
        logging.info(f"Completion Tokens:{cb.completion_tokens}")
        logging.info(f"Total Cost:{cb.total_cost}")
        if isinstance(response, dict):
            #Extract the quiz data from the response
            # quiz=response.get("quiz", None)
            # if quiz is not None:
            #     table_data=get_table_data(quiz)
            #     if table_data is not None:
            #         df=pd.DataFrame(table_data)
            #         df.index=df.index+1
            #         st.table(df)
            #         #Display the review in atext box as well
            #         st.text_area(label="Review", value=response["review"])
            #     else:
            #         st.error("Error in the table data")
            logging.info("Response: "+str(response))

        else:
            logging.error("Error in the response"+str(response))





