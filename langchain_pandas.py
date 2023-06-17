# import libraries

import numpy as np
import pandas as pd
from langchain.llms import OpenAI
from langchain.llms import VertexAI
from langchain.agents import create_pandas_dataframe_agent
import os
import yaml

# get config values from config file
current_path = os.getcwd()
print("current directory is: "+current_path)
path_to_yaml = os.path.join(current_path, 'langchain_df_config.yml')
print("path_to_yaml "+path_to_yaml)
try:
    with open (path_to_yaml, 'r') as c_file:
        config = yaml.safe_load(c_file)
except Exception as e:
    print('Error reading the config file ', e)


# read data file
df = pd.read_csv(config['general']['data_file'])

# use standard pandas approach to answer the questions
print("df.shape[0] \n",df.shape[0])
print("df[df.neighbourhood_group == \"Manhattan\"].shape[0] \n",df[df.neighbourhood_group == "Manhattan"].shape[0])
print("columns with missing values: \n",df.columns[df.isnull().any()])
print("df[df.minimum_nights >= 30].shape[0] \n",df[df.minimum_nights >= 30].shape[0])
print("df[df.minimum_nights == 30].shape[0] \n",df[df.minimum_nights == 30].shape[0])

# define LLM objects
llm = VertexAI()
agent = create_pandas_dataframe_agent(llm, df, verbose=True)

# use LLM to answer the questions
for question in config['questions']:
    print(question)
    print(agent.run(question))  







