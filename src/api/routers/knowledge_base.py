"""
知识库管理 API 路由

提供知识库的 CRUD 操作和文档管理功能。
"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import List, Optional

from src.api.models import (
    KnowledgeBaseCreate,
    KnowledgeBaseInfo,
    KnowledgeBaseList,
    DocumentUpload,
    DocumentInfo,
    DocumentList,
    SuccessResponse,
    ErrorResponse
)
from src.api.dependencies import get_kb_manager, get_embedding
from src.knowledge_base.kb_manager import KnowledgeBaseManager
from src.core.embedding.base import OpenAIEmbedding, BaseEmbedding
from src.core.vectorstore.base import FAISSVectorStore
import os


router = APIRouter(
    prefix="/api/v1/kb",
    tags=["Knowledge Base"],
    responses={404: {"model": ErrorResponse}}
)


@router.post(
    "/create",
    response_model=SuccessResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建知识库",
    description="创建一个新的知识库"
)
async def create_knowledge_base(
    request: KnowledgeBaseCreate,
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager),
    embedding_instance: BaseEmbedding = Depends(get_embedding)
) -> SuccessResponse:
    """
    创建知识库
    
    参数:
        request: 知识库创建请求（名称、描述、embedding 模型）
        kb_manager: 知识库管理器（依赖注入）
        embedding_instance: Embedding 实例（依赖注入）
    
    返回:
        SuccessResponse: 成功消息
    
    异常:
        400: 知识库已存在
        500: 创建失败
    """
    try:
        # 检查知识库是否已存在
        if request.name in kb_manager.knowledge_bases:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"知识库 '{request.name}' 已存在"
            )
        
        # 创建知识库路径
        kb_path = os.path.join(kb_manager.root_path, request.name)
        os.makedirs(kb_path, exist_ok=True)
        
        # 创建 Embedding（如果需要不同的模型，可以创建新实例）
        if request.embedding_model != embedding_instance.model_name:
            embedding = OpenAIEmbedding(model_name=request.embedding_model)
        else:
            embedding = embedding_instance
        
        # 创建 VectorStore
        vectorstore = FAISSVectorStore(
            persist_directory=kb_path,
            embedding=embedding
        )
        
        # 创建知识库
        kb = kb_manager.create_kb(
            name=request.name,
            vectorstore=vectorstore,
            embedding=embedding
        )
        
        # 保存描述信息（如果有）
        if request.description:
            import json
            metadata_file = os.path.join(kb_path, "metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "name": request.name,
                    "description": request.description,
                    "embedding_model": request.embedding_model
                }, f, ensure_ascii=False, indent=2)
        
        return SuccessResponse(
            success=True,
            message=f"知识库 '{request.name}' 创建成功",
            data={"name": request.name}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建知识库时出错: {str(e)}"
        )


@router.get(
    "/list",
    response_model=KnowledgeBaseList,
    summary="列出所有知识库",
    description="获取所有知识库的列表和详细信息"
)
async def list_knowledge_bases(
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager)
) -> KnowledgeBaseList:
    """
    列出所有知识库
    
    参数:
        kb_manager: 知识库管理器（依赖注入）
    
    返回:
        KnowledgeBaseList: 知识库列表及总数
    """
    try:
        kb_names = kb_manager.list_kb()
        kb_list = []
        
        for name in kb_names:
            try:
                kb = kb_manager.get_kb(name)
                info = kb_manager.get_kb_info(name)
                
                kb_list.append(KnowledgeBaseInfo(
                    name=name,
                    description=info.get("description"),
                    document_count=info.get("document_count", 0),
                    created_at=info.get("created_at"),
                    embedding_model=info.get("embedding_model")
                ))
            except Exception as e:
                print(f"获取知识库 '{name}' 信息失败: {e}")
                continue
        
        return KnowledgeBaseList(
            knowledge_bases=kb_list,
            total=len(kb_list)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取知识库列表时出错: {str(e)}"
        )


@router.get(
    "/{kb_name}",
    response_model=KnowledgeBaseInfo,
    summary="获取知识库详情",
    description="获取指定知识库的详细信息"
)
async def get_knowledge_base(
    kb_name: str,
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager)
) -> KnowledgeBaseInfo:
    """
    获取知识库详情
    
    参数:
        kb_name: 知识库名称
        kb_manager: 知识库管理器（依赖注入）
    
    返回:
        KnowledgeBaseInfo: 知识库详细信息
    
    异常:
        404: 知识库不存在
    """
    try:
        if kb_name not in kb_manager.knowledge_bases:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"知识库 '{kb_name}' 不存在"
            )
        
        info = kb_manager.get_kb_info(kb_name)
        
        return KnowledgeBaseInfo(
            name=kb_name,
            description=info.get("description"),
            document_count=info.get("document_count", 0),
            created_at=info.get("created_at"),
            embedding_model=info.get("embedding_model")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取知识库信息时出错: {str(e)}"
        )


@router.delete(
    "/{kb_name}",
    response_model=SuccessResponse,
    summary="删除知识库",
    description="删除指定的知识库及其所有文档"
)
async def delete_knowledge_base(
    kb_name: str,
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager)
) -> SuccessResponse:
    """
    删除知识库
    
    参数:
        kb_name: 知识库名称
        kb_manager: 知识库管理器（依赖注入）
    
    返回:
        SuccessResponse: 成功消息
    
    异常:
        404: 知识库不存在
        500: 删除失败
    """
    try:
        if kb_name not in kb_manager.knowledge_bases:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"知识库 '{kb_name}' 不存在"
            )
        
        kb_manager.delete_kb(kb_name)
        
        return SuccessResponse(
            success=True,
            message=f"知识库 '{kb_name}' 删除成功",
            data={"name": kb_name}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除知识库时出错: {str(e)}"
        )


# ============================================================
# 文档管理端点
# ============================================================

@router.post(
    "/{kb_name}/upload",
    response_model=SuccessResponse,
    status_code=status.HTTP_201_CREATED,
    summary="上传文档",
    description="上传文件到指定知识库"
)
async def upload_document(
    kb_name: str,
    file: UploadFile = File(..., description="要上传的文件"),
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager)
) -> SuccessResponse:
    """
    上传文档到知识库
    
    参数:
        kb_name: 知识库名称
        file: 上传的文件
        kb_manager: 知识库管理器（依赖注入）
    
    返回:
        SuccessResponse: 成功消息及文档信息
    
    异常:
        404: 知识库不存在
        400: 不支持的文件类型
        500: 上传失败
    """
    try:
        # 检查文件类型
        filename = file.filename
        if not filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="文件名不能为空"
            )
        
        # 支持的文件类型
        supported_extensions = ['.txt', '.pdf', '.md', '.markdown', '.docx']
        # 图片类型（暂时只保存，不处理内容）
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        
        file_ext = None
        is_image = False
        
        # 检查是否为图片
        for ext in image_extensions:
            if filename.lower().endswith(ext):
                is_image = True
                file_ext = ext
                break
        
        # 检查是否为支持的文档类型
        if not is_image:
            for ext in supported_extensions:
                if filename.lower().endswith(ext):
                    file_ext = ext
                    break
        
        if not file_ext:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的文件类型: {filename}。支持的类型: {', '.join(supported_extensions + image_extensions)}"
            )
        
        # 如果是图片，保存文件并返回路径用于视觉理解
        if is_image:
            import shutil
            # 如果知识库不存在，自动创建
            if kb_name not in kb_manager.knowledge_bases:
                from src.api.dependencies import get_embedding
                from src.core.embedding.base import BaseEmbedding
                embedding_instance: BaseEmbedding = get_embedding()
                kb_path = os.path.join(kb_manager.root_path, kb_name)
                os.makedirs(kb_path, exist_ok=True)
                
                from src.core.vectorstore.base import FAISSVectorStore
                vectorstore = FAISSVectorStore(
                    persist_directory=kb_path,
                    embedding=embedding_instance
                )
                kb_manager.create_kb(
                    name=kb_name,
                    vectorstore=vectorstore,
                    embedding=embedding_instance
                )
            
            kb = kb_manager.get_kb(kb_name)
            kb_path = os.path.join(kb_manager.root_path, kb_name)
            images_dir = os.path.join(kb_path, 'images')
            os.makedirs(images_dir, exist_ok=True)
            
            # 保存图片文件
            image_path = os.path.join(images_dir, filename)
            with open(image_path, 'wb') as f:
                shutil.copyfileobj(file.file, f)
            
            return SuccessResponse(
                success=True,
                message=f"图片 '{filename}' 已保存",
                data={
                    "filename": filename,
                    "type": "image",
                    "image_path": image_path  # 返回图片路径用于视觉理解
                }
            )
        
        # 对于非图片文件，检查知识库是否存在
        if kb_name not in kb_manager.knowledge_bases:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"知识库 '{kb_name}' 不存在"
            )
        
        # 保存文件到临时位置（保留原始文件名）
        import tempfile
        import shutil
        
        # 创建临时目录，使用原始文件名
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)  # 使用原始文件名
        
        try:
            # 读取文件内容
            content = await file.read()
            
            # 写入临时文件（使用原始文件名）
            with open(temp_path, 'wb') as f:
                f.write(content)
            
            # 上传文件到知识库
            kb = kb_manager.get_kb(kb_name)
            chunk_count = kb.upload_file(temp_path)
            
            return SuccessResponse(
                success=True,
                message=f"文件 '{filename}' 上传成功",
                data={
                    "filename": filename,
                    "chunk_count": chunk_count,
                    "kb_name": kb_name
                }
            )
        finally:
            # 清理临时文件和目录
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as e:
                print(f"⚠️ 清理临时文件失败: {e}")
    
    except HTTPException:
        raise
    except Exception as e:
        # 提供更详细的错误信息
        error_detail = str(e)
        if "不支持的文件格式" in error_detail:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_detail
            )
        elif "加载失败" in error_detail or "加载器" in error_detail or "python-docx" in error_detail:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"文档处理失败: {error_detail}\n\n提示：\n- Word文档需要安装: pip install python-docx unstructured\n- PDF文档需要安装: pip install pypdf\n- 请检查文件是否损坏"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"上传文档时出错: {error_detail}"
            )


@router.get(
    "/{kb_name}/documents",
    response_model=DocumentList,
    summary="列出知识库文档",
    description="获取指定知识库中的所有文档"
)
async def list_documents(
    kb_name: str,
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager)
) -> DocumentList:
    """
    列出知识库中的所有文档
    
    参数:
        kb_name: 知识库名称
        kb_manager: 知识库管理器（依赖注入）
    
    返回:
        DocumentList: 文档列表及总数
    
    异常:
        404: 知识库不存在
    """
    try:
        if kb_name not in kb_manager.knowledge_bases:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"知识库 '{kb_name}' 不存在"
            )
        
        kb = kb_manager.get_kb(kb_name)
        docs = kb.list_documents()
        
        doc_list = []
        for doc in docs:
            source = doc.get("source", "")
            # 从source路径中提取文件名
            import os
            filename = os.path.basename(source) if source else "unknown"
            
            doc_list.append(DocumentInfo(
                filename=filename,
                source=source,
                chunk_count=doc.get("chunk_count", 0),
                upload_time=doc.get("added_at") or doc.get("upload_time"),
                metadata=doc.get("metadata")
            ))
        
        return DocumentList(
            documents=doc_list,
            total=len(doc_list)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取文档列表时出错: {str(e)}"
        )


@router.delete(
    "/{kb_name}/documents/{filename}",
    response_model=SuccessResponse,
    summary="删除文档",
    description="从知识库中删除指定文档"
)
async def delete_document(
    kb_name: str,
    filename: str,
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager)
) -> SuccessResponse:
    """
    删除文档
    
    参数:
        kb_name: 知识库名称
        filename: 文件名
        kb_manager: 知识库管理器（依赖注入）
    
    返回:
        SuccessResponse: 成功消息
    
    异常:
        404: 知识库或文档不存在
        500: 删除失败
    """
    try:
        if kb_name not in kb_manager.knowledge_bases:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"知识库 '{kb_name}' 不存在"
            )
        
        kb = kb_manager.get_kb(kb_name)
        
        # 通过文件名找到对应的source
        docs = kb.list_documents()
        source_to_delete = None
        
        for doc in docs:
            import os
            doc_filename = os.path.basename(doc.get("source", ""))
            if doc_filename == filename:
                source_to_delete = doc.get("source")
                break
        
        if not source_to_delete:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"文档 '{filename}' 不存在于知识库中"
            )
        
        kb.delete_document(source_to_delete)
        
        return SuccessResponse(
            success=True,
            message=f"文档 '{filename}' 删除成功",
            data={"filename": filename, "kb_name": kb_name}
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除文档时出错: {str(e)}"
        )
