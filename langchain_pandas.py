# get config file

import numpy as np
import pandas as pd
from langchain.llms import OpenAI
from langchain.llms import VertexAI
from langchain.agents import create_pandas_dataframe_agent
import os
import yaml
current_path = os.getcwd()
print("current directory is: "+current_path)
path_to_yaml = os.path.join(current_path, 'langchain_df_config.yml')
print("path_to_yaml "+path_to_yaml)
try:
    with open (path_to_yaml, 'r') as c_file:
        config = yaml.safe_load(c_file)
except Exception as e:
    print('Error reading the config file ', e)

# get key
os.environ['OPENAI_API_KEY'] = config['general']['oai_key']

# read data file
df = pd.read_csv(config['general']['data_file'])

print("df.shape is ",df.shape)
print("df[neighbourhood_group].value_counts() \n",df["neighbourhood_group"].value_counts())
# df[df.last == 'smith'].shape[0]
print("df[df.minimum_nights >= 30].shape[0] \n",df[df.minimum_nights >= 30].shape[0])

llm = VertexAI()
agent = create_pandas_dataframe_agent(llm, df, verbose=True)

for question in config['questions']:
    print(question)
    print(agent.run(question))  







