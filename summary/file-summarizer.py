from langchain import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import argparse
from dotenv import load_dotenv

load_dotenv()

def summarize_file(file_path):
    llm = OpenAI(temperature=0)

    text_splitter = CharacterTextSplitter()

    with open(file_path) as f:
        state_of_the_union = f.read()
    texts = text_splitter.split_text(state_of_the_union)
    docs = [Document(page_content=t) for t in texts[:3]]

    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(docs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    args = parser.parse_args()

    summary = summarize_file(args.file_path)
    print(summary)
