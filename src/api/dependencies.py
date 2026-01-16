"""
API 依赖注入

提供全局依赖项，如 KnowledgeBaseManager, LLM, Embedding 等。
"""
from functools import lru_cache
from typing import Optional
import os
from dotenv import load_dotenv

from src.knowledge_base.kb_manager import KnowledgeBaseManager
from src.core.llm.base import OpenAILLM
from src.core.embedding.base import OpenAIEmbedding

# 加载环境变量
load_dotenv()


class ServiceContainer:
    """
    服务容器（单例模式）
    
    管理应用的全局服务实例，如 KnowledgeBaseManager, LLM 等。
    """
    _instance: Optional['ServiceContainer'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # 初始化标志
        self._initialized = True
        
        # 知识库根目录
        self.kb_root_path = os.getenv("KB_ROOT_PATH", "./data/knowledge_base")
        
        # 初始化 KnowledgeBaseManager
        self.kb_manager: Optional[KnowledgeBaseManager] = None
        
        # 初始化 LLM
        self.llm: Optional[OpenAILLM] = None
        
        # 初始化 Embedding
        self.embedding: Optional[OpenAIEmbedding] = None
        
        print(f"📦 ServiceContainer 初始化完成")
    
    def init_services(self):
        """
        初始化所有服务
        """
        try:
            # 初始化 KnowledgeBaseManager
            if self.kb_manager is None:
                self.kb_manager = KnowledgeBaseManager(root_path=self.kb_root_path)
                print(f"✅ KnowledgeBaseManager 初始化成功，根目录: {self.kb_root_path}")
            
            # 初始化 LLM
            if self.llm is None:
                model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
                self.llm = OpenAILLM(model_name=model_name)
                vision_status = "✅ 支持视觉" if self.llm.supports_vision else "❌ 不支持视觉"
                print(f"✅ LLM 初始化成功，模型: {model_name} ({vision_status})")
            
            # 初始化 Embedding
            if self.embedding is None:
                embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
                self.embedding = OpenAIEmbedding(model_name=embedding_model)
                print(f"✅ Embedding 初始化成功，模型: {embedding_model}")
            
            return True
        except Exception as e:
            print(f"❌ 服务初始化失败: {e}")
            return False
    
    def get_kb_manager(self) -> KnowledgeBaseManager:
        """获取 KnowledgeBaseManager 实例"""
        if self.kb_manager is None:
            self.init_services()
        return self.kb_manager
    
    def get_llm(self) -> OpenAILLM:
        """获取 LLM 实例"""
        if self.llm is None:
            self.init_services()
        return self.llm
    
    def get_embedding(self) -> OpenAIEmbedding:
        """获取 Embedding 实例"""
        if self.embedding is None:
            self.init_services()
        return self.embedding


# 全局服务容器实例
@lru_cache()
def get_service_container() -> ServiceContainer:
    """
    获取服务容器单例
    
    使用 FastAPI 的依赖注入系统
    """
    return ServiceContainer()


# ============================================================
# FastAPI 依赖函数
# ============================================================

def get_kb_manager() -> KnowledgeBaseManager:
    """
    获取 KnowledgeBaseManager（FastAPI 依赖）
    
    用法:
        @app.get("/api/v1/kb/list")
        def list_kb(kb_manager: KnowledgeBaseManager = Depends(get_kb_manager)):
            ...
    """
    container = get_service_container()
    return container.get_kb_manager()


def get_llm() -> OpenAILLM:
    """
    获取 LLM 实例（FastAPI 依赖）
    """
    container = get_service_container()
    return container.get_llm()


def get_embedding() -> OpenAIEmbedding:
    """
    获取 Embedding 实例（FastAPI 依赖）
    """
    container = get_service_container()
    return container.get_embedding()
