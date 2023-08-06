import argparse
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv

load_dotenv()

def summarize_web_page(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

    prompt_template = """Write a concise summary of the following:\n{text}\nCONCISE SUMMARY IN CHINESE:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    # chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)

    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=PROMPT, combine_prompt=PROMPT)

    summarize = chain(docs, return_only_outputs=True)
    return summarize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", type=str)
    args = parser.parse_args()

    summary = summarize_web_page(args.url)
    print(summary)
