# get config file

import numpy as np
import pandas as pd
from langchain.llms import OpenAI
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

agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

output1 = agent.run("how many rows are there?")

print("output1 is ",output1)







