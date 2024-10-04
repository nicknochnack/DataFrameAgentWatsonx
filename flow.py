from dotenv import load_dotenv
import os
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_ibm import WatsonxLLM
from colorama import init, Fore

load_dotenv()
init()
# Decoding parameters
params = {
    "decoding_method": "sample",
    "max_new_tokens": 300,
    "min_new_tokens": 1,
    "top_k": 50,
}

# Create the LLM
llm = WatsonxLLM(
    model_id="meta-llama/llama-3-1-70b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id=os.environ["WATSONX_PROJECTID"],
    params=params,
)

# IMport the dataframe
df = pd.read_csv("train.csv")
# print(df.head())

# Create the agent
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True,
    max_iterations=30,
    max_execution_time=300,
    agent_executor_kwargs={"handle_parsing_errors": True},
)

# Run logic
if __name__ == "__main__":
    while True:
        prompt = input(
            Fore.YELLOW
            + "Ask your data question here (type /quit to exit)"
            + Fore.RESET
        )
        if prompt.lower() == "/quit":
            print(Fore.RED + "Catch ya later, exiting" + Fore.RESET)
            break
        # Run the prompt if the user hasn't entered quit
        response = agent.run(prompt)
        print(Fore.LIGHTMAGENTA_EX + response + Fore.RESET)
