"""
API 数据模型

使用 Pydantic 定义请求和响应的数据结构。
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ============================================================
# 聊天相关模型
# ============================================================

class ChatRequest(BaseModel):
    """
    聊天请求模型
    
    用于 /api/v1/chat 端点
    """
    question: str = Field(..., description="用户的问题", min_length=1, max_length=1000)
    kb_name: Optional[str] = Field(None, description="知识库名称（RAG模式必需）", min_length=1)
    conversation_id: Optional[str] = Field(None, description="对话ID，用于多轮对话")
    stream: bool = Field(False, description="是否使用流式输出")
    max_steps: int = Field(5, description="Agent 最大执行步骤", ge=1, le=20)
    image_paths: Optional[List[str]] = Field(None, description="图片路径列表，用于视觉理解（已废弃，使用images）")
    images: Optional[List[str]] = Field(None, description="图片base64数据列表，格式：data:image/jpeg;base64,...")
    model_name: Optional[str] = Field(None, description="模型名称，如果不提供则使用默认模型")
    conversation_history: Optional[List[dict]] = Field(None, description="对话历史，格式：[{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "Python 是什么？",
                    "kb_name": "tech_kb",
                    "conversation_id": "conv-123",
                    "stream": False,
                    "max_steps": 5
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """
    聊天响应模型
    """
    answer: str = Field(..., description="生成的答案")
    conversation_id: str = Field(..., description="对话ID")
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list, description="检索到的文档")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据（检索分数、执行步骤等）")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "Python 是一种高级编程语言...",
                    "conversation_id": "conv-123",
                    "retrieved_docs": [
                        {
                            "content": "Python 相关内容",
                            "source": "doc1.txt",
                            "score": 0.95
                        }
                    ],
                    "metadata": {
                        "retrieval_score": 0.91,
                        "confidence_score": 0.88,
                        "step_count": 2
                    }
                }
            ]
        }
    }


# ============================================================
# 知识库管理模型
# ============================================================

class KnowledgeBaseCreate(BaseModel):
    """
    创建知识库请求模型
    """
    name: str = Field(..., description="知识库名称", min_length=1, max_length=100)
    description: Optional[str] = Field(None, description="知识库描述", max_length=500)
    embedding_model: str = Field("text-embedding-ada-002", description="Embedding 模型名称")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "tech_kb",
                    "description": "技术知识库",
                    "embedding_model": "text-embedding-ada-002"
                }
            ]
        }
    }


class KnowledgeBaseInfo(BaseModel):
    """
    知识库信息响应模型
    """
    name: str = Field(..., description="知识库名称")
    description: Optional[str] = Field(None, description="知识库描述")
    document_count: int = Field(0, description="文档数量")
    created_at: Optional[str] = Field(None, description="创建时间")
    embedding_model: Optional[str] = Field(None, description="Embedding 模型")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "tech_kb",
                    "description": "技术知识库",
                    "document_count": 15,
                    "created_at": "2026-01-15 10:30:00",
                    "embedding_model": "text-embedding-ada-002"
                }
            ]
        }
    }


class KnowledgeBaseList(BaseModel):
    """
    知识库列表响应模型
    """
    knowledge_bases: List[KnowledgeBaseInfo] = Field(default_factory=list, description="知识库列表")
    total: int = Field(0, description="知识库总数")


class DocumentUpload(BaseModel):
    """
    文档上传请求模型（用于文本内容）
    """
    kb_name: str = Field(..., description="知识库名称")
    content: str = Field(..., description="文档内容", min_length=1)
    filename: str = Field(..., description="文件名")
    metadata: Optional[Dict[str, Any]] = Field(None, description="文档元数据")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "kb_name": "tech_kb",
                    "content": "Python 是一种高级编程语言...",
                    "filename": "python_intro.txt",
                    "metadata": {"author": "张三", "category": "编程"}
                }
            ]
        }
    }


class DocumentInfo(BaseModel):
    """
    文档信息响应模型
    """
    filename: str = Field(..., description="文件名")
    source: str = Field(..., description="文档来源路径")
    chunk_count: int = Field(0, description="分块数量")
    upload_time: Optional[str] = Field(None, description="上传时间")
    metadata: Optional[Dict[str, Any]] = Field(None, description="文档元数据")


class DocumentList(BaseModel):
    """
    文档列表响应模型
    """
    documents: List[DocumentInfo] = Field(default_factory=list, description="文档列表")
    total: int = Field(0, description="文档总数")


# ============================================================
# 通用响应模型
# ============================================================

class SuccessResponse(BaseModel):
    """
    成功响应模型
    """
    success: bool = Field(True, description="是否成功")
    message: str = Field(..., description="响应消息")
    data: Optional[Any] = Field(None, description="响应数据")


class ErrorResponse(BaseModel):
    """
    错误响应模型
    """
    success: bool = Field(False, description="是否成功")
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误消息")
    detail: Optional[Any] = Field(None, description="错误详情")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": False,
                    "error": "KnowledgeBaseNotFound",
                    "message": "知识库 'tech_kb' 不存在",
                    "detail": None
                }
            ]
        }
    }


# ============================================================
# 健康检查模型
# ============================================================

class HealthResponse(BaseModel):
    """
    健康检查响应模型
    """
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="API 版本")
    timestamp: str = Field(..., description="当前时间")
    components: Dict[str, str] = Field(default_factory=dict, description="组件状态")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "version": "1.0.0",
                    "timestamp": "2026-01-15T10:30:00",
                    "components": {
                        "llm": "ok",
                        "embedding": "ok",
                        "vectorstore": "ok"
                    }
                }
            ]
        }
    }
