"""
文档处理模块

这个模块提供了文档加载和文本切分的功能，是 RAG 系统中的重要组成部分。

主要功能：
1. 文档加载：支持多种格式（TXT、PDF、Markdown）的文档加载
2. 文本切分：将长文档切分成适合向量化的小块（Chunks）

使用示例：
    >>> from src.core.document import TextLoader, RecursiveTextSplitter
    >>> 
    >>> # 加载文档
    >>> loader = TextLoader()
    >>> documents = loader.load("example.txt")
    >>> 
    >>> # 切分文档
    >>> splitter = RecursiveTextSplitter(chunk_size=500, chunk_overlap=50)
    >>> chunks = splitter.split_documents(documents)
"""

from .loader import (
    BaseDocumentLoader,
    TextLoader,
    PDFLoader,
    MarkdownLoader,
    DocumentLoaderFactory,
)

from .splitter import (
    BaseTextSplitter,
    RecursiveTextSplitter,
    ChineseTextSplitter,
    TextSplitterFactory,
)

__all__ = [
    # 文档加载器
    "BaseDocumentLoader",
    "TextLoader",
    "PDFLoader",
    "MarkdownLoader",
    "DocxLoader",
    "DocumentLoaderFactory",
    # 文本切分器
    "BaseTextSplitter",
    "RecursiveTextSplitter",
    "ChineseTextSplitter",
    "TextSplitterFactory",
]


