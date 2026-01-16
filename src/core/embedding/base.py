"""
Embedding 基础接口定义

TODO: 你需要实现这个接口的具体类
"""
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama  # 新版本
from langchain_openai import OpenAIEmbeddings
from ollama import embeddings
import os
import dotenv

dotenv.load_dotenv()

class BaseEmbedding(ABC):
    """Embedding 基础接口"""
    
    def __init__(self, model_name: str):
        """
        初始化 Embedding 模型
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        对单个查询文本进行向量化
        
        Args:
            text: 查询文本
            
        Returns:
            向量表示
        """
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        对文档列表进行向量化
        
        Args:
            texts: 文档文本列表
            
        Returns:
            向量列表
        """
        pass


# TODO: 实现 OpenAIEmbedding 类
# 提示：
# 1. 继承 BaseEmbedding
# 2. 使用 langchain_openai.OpenAIEmbeddings
# 3. 实现 embed_query 和 embed_documents 方法
# 4. 参考文档: https://python.langchain.com/docs/integrations/text_embedding/openai

class OpenAIEmbedding(BaseEmbedding):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        api_key=os.getenv("OPENAI_API_KEY")

        self.embeddings=OpenAIEmbeddings(
            api_key=api_key,
            model=model_name
        )

    def embed_query(self, text: str) -> List[float]:
        """
          对文档列表进行向量化

          Args:
              texts: 文档文本列表

          Returns:
              向量列表（每个文档对应一个向量）
          """
        try:
           vector =self.embeddings.embed_query(text)
           return vector
        except Exception as e:
            raise Exception(f"向量化文档失败: {str(e)}") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
            对文档列表进行向量化

            Args:
                texts: 文档文本列表

            Returns:
                向量列表（每个文档对应一个向量）
            """
        try:
            # 直接调用 OpenAIEmbeddings 的 embed_documents 方法
            vectors = self.embeddings.embed_documents(texts)
            return vectors
        except Exception as e:
            raise Exception(f"向量化文档失败: {str(e)}") from e

# TODO: 实现本地 Embedding 模型（可选）
# 提示：可以使用 sentence-transformers 或其他本地模型

class OllamaBaseEmbedding(BaseEmbedding):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.embeddings=OllamaEmbeddings(
            model=model_name,
        )

    def embed_query(self, text: str) -> List[float]:
        """
          对文档列表进行向量化

          Args:
              texts: 文档文本列表

          Returns:
              向量列表（每个文档对应一个向量）
          """
        try:
            vector = self.embeddings.embed_query(text)
            return vector
        except Exception as e:
            raise Exception(f"向量化文档失败: {str(e)}") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
            对文档列表进行向量化

            Args:
                texts: 文档文本列表

            Returns:
                向量列表（每个文档对应一个向量）
            """
        try:
            # 直接调用 OpenAIEmbeddings 的 embed_documents 方法
            vectors = self.embeddings.embed_documents(texts)
            return vectors
        except Exception as e:
            raise Exception(f"向量化文档失败: {str(e)}") from e

if __name__ == '__main__':
    def cosine_similarity(vec1, vec2):
        """计算余弦相似度"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # 创建实例
    embedding = OpenAIEmbedding(model_name="text-embedding-ada-002")

    # 测试 embed_query
    query = "什么是 Python？"
    query_vector = embedding.embed_query(query)
    print(f"查询向量维度: {len(query_vector)}")
    print(f"向量前5个值: {query_vector[:5]}")

    # 测试 embed_documents
    documents = [
        "Python 是一种编程语言",
        "Java 是一种编程语言",
        "今天天气真好"
    ]
    doc_vectors = embedding.embed_documents(documents)
    print(f"文档数量: {len(doc_vectors)}")
    print(f"每个向量维度: {len(doc_vectors[0])}")
# 测试相似度

    query = "什么是 Python？"
    doc1 = "Python 是一种编程语言"
    doc2 = "今天天气真好"

    query_vec = embedding.embed_query(query)
    doc1_vec = embedding.embed_query(doc1)  # 注意：也可以用 embed_query
    doc2_vec = embedding.embed_query(doc2)

    # 计算相似度
    similarity1 = cosine_similarity(query_vec, doc1_vec)
    similarity2 = cosine_similarity(query_vec, doc2_vec)

    print(f"查询与文档1的相似度: {similarity1:.4f}")  # 应该较高
    print(f"查询与文档2的相似度: {similarity2:.4f}")  # 应该较低

