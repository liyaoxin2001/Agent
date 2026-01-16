"""
文档加载器模块

本模块实现了多种文档加载器，用于将不同格式的文件转换为 LangChain 的 Document 对象。

Document 对象结构：
    - page_content: 文档的文本内容（字符串）
    - metadata: 文档的元数据（字典），如来源、页码、作者等

设计模式：
    - 策略模式：不同的加载器实现相同的接口，可以灵活切换
    - 工厂模式：通过文件扩展名自动选择合适的加载器

支持的格式：
    - TXT: 纯文本文件
    - PDF: PDF 文档
    - Markdown: Markdown 格式文档
"""

from abc import ABC, abstractmethod
from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader as LCTextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)


class BaseDocumentLoader(ABC):
    """
    文档加载器抽象基类
    
    这个基类定义了所有文档加载器必须实现的接口。
    通过继承这个基类，可以轻松扩展支持新的文档格式。
    
    设计理念：
        - 单一职责：每个加载器只负责一种格式
        - 开闭原则：对扩展开放，对修改关闭
        - 依赖倒置：依赖抽象而不是具体实现
    """
    
    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """
        加载文档并返回 Document 对象列表
        
        Args:
            file_path: 文档文件路径（相对路径或绝对路径）
            
        Returns:
            List[Document]: Document 对象列表，每个对象包含：
                - page_content: 文档内容（字符串）
                - metadata: 元数据字典，如 {"source": "file.txt"}
                
        Raises:
            Exception: 文档加载失败时抛出异常
            
        注意：
            - 某些格式（如 PDF）可能会将每一页作为一个 Document
            - 某些格式（如 TXT）通常整个文件是一个 Document
        """
        pass


class TextLoader(BaseDocumentLoader):
    """
    文本文件加载器
    
    用于加载纯文本文件（.txt），支持多种编码格式。
    
    特点：
        - 简单高效：直接读取文本内容
        - 编码兼容：支持 UTF-8、GBK 等多种编码
        - 自动检测：如果 UTF-8 失败，会尝试其他编码
        
    使用场景：
        - 加载纯文本文档
        - 加载日志文件
        - 加载代码文件
    """
    
    def __init__(self, encoding: str = "utf-8", autodetect_encoding: bool = True):
        """
        初始化文本加载器
        
        Args:
            encoding: 文件编码格式，默认 "utf-8"
            autodetect_encoding: 是否自动检测编码，默认 True
                - True: 如果指定编码失败，会尝试其他常见编码
                - False: 只使用指定的编码
        """
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding
    
    def load(self, file_path: str) -> List[Document]:
        """
        加载文本文件
        
        Args:
            file_path: 文本文件路径
            
        Returns:
            List[Document]: 包含一个 Document 对象的列表
            
        工作流程：
            1. 检查文件是否存在
            2. 使用 LangChain 的 TextLoader 加载文件
            3. 捕获异常并提供友好的错误信息
            
        注意：
            - LangChain 的 TextLoader 会自动处理编码问题
            - 返回的列表通常只包含一个 Document（整个文件）
        """
        try:
            # 检查文件是否存在
            if not Path(file_path).exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 使用 LangChain 的 TextLoader 加载文件
            # LangChain 会自动处理编码检测和文件读取
            loader = LCTextLoader(
                file_path=file_path,
                encoding=self.encoding,
                autodetect_encoding=self.autodetect_encoding
            )
            
            # load() 返回 List[Document]
            documents = loader.load()
            
            return documents
            
        except FileNotFoundError as e:
            # 文件不存在的错误
            raise Exception(f"文件加载失败: {str(e)}") from e
        except UnicodeDecodeError as e:
            # 编码错误
            raise Exception(f"文件编码错误，请尝试指定正确的编码格式: {str(e)}") from e
        except Exception as e:
            # 其他未知错误
            raise Exception(f"文本文件加载失败: {str(e)}") from e


class PDFLoader(BaseDocumentLoader):
    """
    PDF 文档加载器
    
    用于加载 PDF 文件，自动提取文本内容。
    
    特点：
        - 按页加载：每一页作为一个独立的 Document
        - 保留元数据：包含页码、来源等信息
        - 自动提取：无需手动处理 PDF 格式
        
    使用场景：
        - 加载学术论文
        - 加载技术文档
        - 加载电子书
        
    注意事项：
        - 扫描版 PDF（图片）可能无法提取文字
        - 复杂排版可能影响提取质量
        - 需要安装 pypdf 依赖：pip install pypdf
    """
    
    def load(self, file_path: str) -> List[Document]:
        """
        加载 PDF 文件
        
        Args:
            file_path: PDF 文件路径
            
        Returns:
            List[Document]: Document 对象列表，每个对象对应 PDF 的一页
                每个 Document 的 metadata 包含：
                - source: 文件路径
                - page: 页码（从 0 开始）
                
        工作流程：
            1. 检查文件是否存在且为 PDF 格式
            2. 使用 PyPDFLoader 逐页提取文本
            3. 每一页生成一个 Document 对象
            
        示例：
            >>> loader = PDFLoader()
            >>> docs = loader.load("paper.pdf")
            >>> print(f"PDF 共 {len(docs)} 页")
            >>> print(f"第一页内容: {docs[0].page_content[:100]}")
            >>> print(f"第一页元数据: {docs[0].metadata}")
        """
        try:
            # 检查文件是否存在
            if not Path(file_path).exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 检查文件扩展名
            if not file_path.lower().endswith('.pdf'):
                raise ValueError(f"不是有效的 PDF 文件: {file_path}")
            
            # 使用 PyPDFLoader 加载 PDF
            # PyPDFLoader 会自动逐页提取文本
            loader = PyPDFLoader(file_path)
            
            # load() 返回 List[Document]，每页一个 Document
            documents = loader.load()
            
            # 如果 PDF 为空
            if not documents:
                raise ValueError(f"PDF 文件为空或无法提取文本: {file_path}")
            
            return documents
            
        except FileNotFoundError as e:
            raise Exception(f"PDF 加载失败: {str(e)}") from e
        except ValueError as e:
            raise Exception(f"PDF 加载失败: {str(e)}") from e
        except Exception as e:
            raise Exception(f"PDF 文件加载失败: {str(e)}") from e


class MarkdownLoader(BaseDocumentLoader):
    """
    Markdown 文档加载器
    
    用于加载 Markdown 格式的文档（.md）。
    
    特点：
        - 保留结构：尽可能保留 Markdown 的格式结构
        - 智能解析：自动处理标题、列表、代码块等
        - 纯文本输出：转换为纯文本便于向量化
        
    使用场景：
        - 加载技术文档
        - 加载笔记
        - 加载 README 文件
        
    注意事项：
        - 需要安装依赖：pip install unstructured markdown
        - 某些复杂格式可能无法完美保留
    """
    
    def __init__(self, mode: str = "single"):
        """
        初始化 Markdown 加载器
        
        Args:
            mode: 加载模式
                - "single": 整个文件作为一个 Document（默认）
                - "elements": 按元素分割（标题、段落等）
        """
        self.mode = mode
    
    def load(self, file_path: str) -> List[Document]:
        """
        加载 Markdown 文件
        
        Args:
            file_path: Markdown 文件路径
            
        Returns:
            List[Document]: Document 对象列表
            
        工作流程：
            1. 检查文件是否存在且为 .md 格式
            2. 使用 UnstructuredMarkdownLoader 解析
            3. 返回 Document 列表
            
        示例：
            >>> loader = MarkdownLoader()
            >>> docs = loader.load("README.md")
            >>> print(docs[0].page_content)
        """
        try:
            # 检查文件是否存在
            if not Path(file_path).exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 检查文件扩展名
            if not file_path.lower().endswith(('.md', '.markdown')):
                raise ValueError(f"不是有效的 Markdown 文件: {file_path}")
            
            # 使用 UnstructuredMarkdownLoader 加载
            loader = UnstructuredMarkdownLoader(
                file_path=file_path,
                mode=self.mode
            )
            
            # 加载文档
            documents = loader.load()
            
            return documents
            
        except FileNotFoundError as e:
            raise Exception(f"Markdown 加载失败: {str(e)}") from e
        except ValueError as e:
            raise Exception(f"Markdown 加载失败: {str(e)}") from e
        except Exception as e:
            raise Exception(f"Markdown 文件加载失败: {str(e)}") from e


class DocxLoader(BaseDocumentLoader):
    """
    Word 文档加载器
    
    用于加载 Microsoft Word 文档（.docx）。
    
    特点：
        - 支持 .docx 格式（不支持旧的 .doc 格式）
        - 提取文本内容和基本格式
        - 保留段落结构
        
    使用场景：
        - 加载 Word 文档
        - 处理 Office 文档
        
    注意事项：
        - 需要安装依赖：pip install python-docx unstructured
        - 不支持 .doc 格式（旧版 Word 格式）
    """
    
    def load(self, file_path: str) -> List[Document]:
        """
        加载 Word 文档
        
        Args:
            file_path: Word 文档路径
            
        Returns:
            List[Document]: Document 对象列表
            
        工作流程：
            1. 检查文件是否存在且为 .docx 格式
            2. 使用 UnstructuredWordDocumentLoader 解析
            3. 返回 Document 列表
        """
        try:
            # 检查文件是否存在
            if not Path(file_path).exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 检查文件扩展名
            if not file_path.lower().endswith('.docx'):
                raise ValueError(f"不是有效的 Word 文档: {file_path}（仅支持 .docx 格式）")
            
            # 使用 UnstructuredWordDocumentLoader 加载
            loader = UnstructuredWordDocumentLoader(file_path=file_path)
            
            # 加载文档
            documents = loader.load()
            
            if not documents:
                raise ValueError(f"Word 文档为空或无法提取文本: {file_path}")
            
            return documents
            
        except FileNotFoundError as e:
            raise Exception(f"Word 文档加载失败: {str(e)}") from e
        except ValueError as e:
            raise Exception(f"Word 文档加载失败: {str(e)}") from e
        except Exception as e:
            raise Exception(f"Word 文档加载失败: {str(e)}。请确保已安装 python-docx 和 unstructured: pip install python-docx unstructured") from e


class DocumentLoaderFactory:
    """
    文档加载器工厂类
    
    根据文件扩展名自动选择合适的加载器，简化使用流程。
    
    设计模式：工厂模式
        - 封装对象创建逻辑
        - 根据条件返回不同的加载器实例
        - 客户端无需关心具体加载器类型
        
    使用示例：
        >>> # 不需要手动判断文件类型
        >>> loader = DocumentLoaderFactory.get_loader("example.pdf")
        >>> documents = loader.load("example.pdf")
    """
    
    # 支持的文件类型映射
    _loaders = {
        '.txt': TextLoader,
        '.pdf': PDFLoader,
        '.md': MarkdownLoader,
        '.markdown': MarkdownLoader,
        '.docx': DocxLoader,
    }
    
    @classmethod
    def get_loader(cls, file_path: str) -> BaseDocumentLoader:
        """
        根据文件扩展名获取对应的加载器
        
        Args:
            file_path: 文件路径
            
        Returns:
            BaseDocumentLoader: 对应的加载器实例
            
        Raises:
            ValueError: 不支持的文件格式
            
        示例：
            >>> loader = DocumentLoaderFactory.get_loader("doc.pdf")
            >>> type(loader)  # PDFLoader
        """
        # 获取文件扩展名（小写）
        ext = Path(file_path).suffix.lower()
        
        # 查找对应的加载器类
        loader_class = cls._loaders.get(ext)
        
        if loader_class is None:
            # 不支持的格式
            supported_formats = ', '.join(cls._loaders.keys())
            raise ValueError(
                f"不支持的文件格式: {ext}\n"
                f"支持的格式: {supported_formats}"
            )
        
        # 返回加载器实例
        return loader_class()
    
    @classmethod
    def load(cls, file_path: str) -> List[Document]:
        """
        便捷方法：直接加载文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            List[Document]: 加载的文档列表
            
        这是一个便捷方法，等价于：
            loader = DocumentLoaderFactory.get_loader(file_path)
            documents = loader.load(file_path)
            
        示例：
            >>> # 最简单的使用方式
            >>> documents = DocumentLoaderFactory.load("example.pdf")
        """
        loader = cls.get_loader(file_path)
        return loader.load(file_path)


