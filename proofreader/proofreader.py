from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import SpacyTextSplitter
from langchain.docstore.document import Document
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import argparse
from dotenv import load_dotenv

load_dotenv()


def proofread_file(file_path):
    with open(file_path) as f:
        state_of_the_union = f.read()
    
    text_splitter = SpacyTextSplitter(separator="\n\n", pipeline="zh_core_web_sm", chunk_size=200)
    docs = text_splitter.create_documents([state_of_the_union])
    print(docs)

    # prompt_template = """检查以下内容进行文本校对:
    # "{text}"
    # 校对后的文本:"""
    # prompt = PromptTemplate.from_template(prompt_template)

    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    # llm_chain = LLMChain(llm=llm, prompt=prompt)

    # chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

    # return chain(docs, return_only_outputs=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    args = parser.parse_args()

    proofread_content = proofread_file(args.file_path)
    print(proofread_content)

    output_file_path = "proofread_output.txt"
    with open(output_file_path, "w") as output_file:
        output_file.write(proofread_content["output_text"])

    print("Proofread content written to:", output_file_path)
