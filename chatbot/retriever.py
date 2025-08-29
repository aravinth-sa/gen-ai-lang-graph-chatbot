from config import Config
from pinecone import Pinecone

class PineconeRetriever:
    def __init__(self):
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY, environment=Config.PINECONE_ENVIRONMENT)
        self.index = self.pc.Index(Config.PINECONE_INDEX_NAME)

    def get_retriever(self):
        return self.index.as_retriever()


retriever = PineconeRetriever()