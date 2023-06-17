# Pandas experiments with LangChain and Vertex AI Generative AI

Take advantage of the LangChain [create_pandas_dataframe_agent](https://python.langchain.com/docs/modules/agents/toolkits/pandas) API to use Vertex AI Generative AI in Google Cloud to answer English-language questions about Pandas dataframes.

[langchain_pandas.py](https://github.com/ryanmark1867/langchain-pandas/blob/main/langchain_pandas.py):
- loads required libraries
- reads set of question from a yaml config file
- answers the question using hardcoded, standard Pandas approach
- uses Vertex AI Generative AI + LangChain to answer the same questions

Thanks to [this example](https://github.com/bhattbhavesh91/langchain-crashcourse/blob/main/pandas-dataframe-agent-notebook.ipynb) for demonstrating some of the approaches used here.
