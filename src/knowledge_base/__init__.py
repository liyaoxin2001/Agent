"""
知识库管理模块

本模块提供了知识库的管理功能，是 RAG 系统的核心组件之一。

主要类：
- KnowledgeBase: 单个知识库的封装，提供文档的增删查操作
- KnowledgeBaseManager: 管理多个知识库的创建、获取、删除

使用示例：
    >>> from src.knowledge_base import KnowledgeBase, KnowledgeBaseManager
    >>> from src.core.embedding import OpenAIEmbedding
    >>> from src.core.vectorstore import FAISSVectorStore
    >>> 
    >>> # 创建管理器
    >>> manager = KnowledgeBaseManager(root_path="./data/knowledge_base")
    >>> 
    >>> # 创建知识库
    >>> embedding = OpenAIEmbedding(model_name="text-embedding-ada-002")
    >>> vectorstore = FAISSVectorStore(
    ...     embedding=embedding,
    ...     persist_directory="./data/kb1"
    ... )
    >>> kb = manager.create_kb("技术文档", vectorstore, embedding)
    >>> 
    >>> # 添加文档
    >>> from langchain.schema import Document
    >>> docs = [Document(page_content="Python 教程")]
    >>> kb.add_documents(docs)
    >>> 
    >>> # 搜索
    >>> results = kb.search("Python 是什么？", k=3)
"""

from .kb_manager import (
    KnowledgeBase,
    KnowledgeBaseManager,
)

__all__ = [
    "KnowledgeBase",
    "KnowledgeBaseManager",
]

