"""
聊天相关 API 路由

提供对话问答功能。
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from typing import AsyncIterator
import uuid
import json

from src.api.models import ChatRequest, ChatResponse, ErrorResponse
from src.api.dependencies import get_kb_manager, get_llm
from src.knowledge_base.kb_manager import KnowledgeBaseManager
from src.core.llm.base import OpenAILLM
from src.agent.state import create_initial_state
from src.agent.graph import create_simple_rag_agent


router = APIRouter(
    prefix="/api/v1/chat",
    tags=["Chat"],
    responses={404: {"model": ErrorResponse}}
)


@router.get(
    "/default-model",
    summary="获取默认模型",
    description="获取当前配置的默认模型名称"
)
async def get_default_model(
    default_llm: OpenAILLM = Depends(get_llm)
) -> dict:
    """
    获取默认模型信息
    
    返回:
        dict: 包含默认模型名称和是否支持视觉
    """
    import os
    env_model = os.getenv("OPENAI_MODEL")
    return {
        "model_name": default_llm.model_name,
        "supports_vision": default_llm.supports_vision,
        "from_env": env_model is not None,
        "env_model": env_model,
    }


@router.post(
    "",
    response_model=ChatResponse,
    summary="发送聊天消息",
    description="向指定知识库发送问题，获取 AI 生成的答案"
)
async def chat(
    request: ChatRequest,
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager),
    llm: OpenAILLM = Depends(get_llm)
) -> ChatResponse:
    """
    聊天端点
    
    参数:
        request: 聊天请求（包含问题、知识库名称等）
        kb_manager: 知识库管理器（依赖注入）
        llm: LLM 实例（依赖注入）
    
    返回:
        ChatResponse: 包含答案、检索文档、元数据等
    
    异常:
        404: 知识库不存在
        500: 服务器内部错误
    """
    try:
        # 1. 检查知识库是否存在（RAG 模式必需）
        if not request.kb_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="RAG 模式需要指定知识库名称"
            )
        if request.kb_name not in kb_manager.knowledge_bases:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"知识库 '{request.kb_name}' 不存在"
            )
        
        # 2. 获取知识库
        kb = kb_manager.get_kb(request.kb_name)
        
        # 3. 生成对话 ID（如果没有提供）
        conversation_id = request.conversation_id or f"conv-{uuid.uuid4().hex[:8]}"
        
        # 4. 创建 Agent
        agent = create_simple_rag_agent(
            llm=llm,
            vectorstore=kb.vectorstore,
            k=4
        )
        
        # 5. 创建初始状态
        state = create_initial_state(
            question=request.question,
            max_steps=request.max_steps,
            conversation_id=conversation_id
        )
        
        # 6. 运行 Agent
        result = agent.invoke(state)
        
        # 7. 构建响应
        retrieved_docs = []
        if result.get('retrieved_docs'):
            for doc in result['retrieved_docs']:
                retrieved_docs.append({
                    "content": doc.page_content[:200],  # 限制长度
                    "source": doc.metadata.get("source", "unknown"),
                    "score": doc.metadata.get("score", 0.0)
                })
        
        metadata = {
            "retrieval_score": result.get("retrieval_score"),
            "confidence_score": result.get("confidence_score"),
            "step_count": result.get("step_count", 0),
            "kb_name": request.kb_name
        }
        
        return ChatResponse(
            answer=result.get("answer", "抱歉，我无法回答这个问题。"),
            conversation_id=conversation_id,
            retrieved_docs=retrieved_docs,
            metadata=metadata
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理聊天请求时出错: {str(e)}"
        )


@router.post(
    "/general",
    response_model=ChatResponse,
    summary="通用对话",
    description="通用对话模式，直接调用 LLM，不使用知识库"
)
async def chat_general(
        request: ChatRequest,
        default_llm: OpenAILLM = Depends(get_llm)
    ) -> ChatResponse:
        """
        通用对话端点
        
        直接调用 LLM 生成回答，不使用知识库检索。
        支持动态选择模型。
        """
        try:
            # 如果请求中指定了模型，使用指定的模型；否则使用默认模型
            if request.model_name and request.model_name != default_llm.model_name:
                # 动态创建指定模型的LLM实例
                llm = OpenAILLM(model_name=request.model_name)
            else:
                llm = default_llm
            
            # 支持图片输入（优先使用images，兼容image_paths）
            images = getattr(request, 'images', None) or getattr(request, 'image_paths', None) or []
            # 获取对话历史
            conversation_history = getattr(request, 'conversation_history', None) or []
            answer = llm.generate(
                prompt=request.question,
                images=images if images else [],
                conversation_history=conversation_history
            )
            
            return ChatResponse(
                answer=answer,
                conversation_id=request.conversation_id or f"conv-{uuid.uuid4().hex[:8]}",
                retrieved_docs=[],
                metadata={
                    "mode": "general",
                    "has_images": bool(images),
                    "model_used": llm.model_name,
                    "supports_vision": llm.supports_vision
                }
            )
        except ValueError as e:
            # 模型不支持视觉的错误，返回400而不是500
            error_msg = str(e)
            if "不支持视觉功能" in error_msg:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_msg
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"通用对话失败: {error_msg}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"通用对话失败: {str(e)}"
            )


@router.post(
    "/search",
    response_model=ChatResponse,
    summary="联网搜索",
    description="联网搜索 + AI 总结"
)
async def chat_search(
    request: ChatRequest,
    llm: OpenAILLM = Depends(get_llm)
) -> ChatResponse:
    """
    联网搜索端点
    
    使用 DuckDuckGo 搜索，然后用 AI 总结结果。
    """
    try:
        from duckduckgo_search import DDGS
        
        # 1. 搜索
        search_results = []
        with DDGS() as ddgs:
            for r in ddgs.text(request.question, max_results=5):
                search_results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", "")
                })
        
        if not search_results:
            return ChatResponse(
                answer="抱歉，未找到相关搜索结果。",
                conversation_id=request.conversation_id or f"conv-{uuid.uuid4().hex[:8]}",
                retrieved_docs=[],
                metadata={"mode": "search"}
            )
        
        # 2. 构建上下文
        context = "\n\n".join([
            f"**{r['title']}**\n{r['snippet']}\n来源: {r['url']}"
            for r in search_results[:3]
        ])
        
        # 3. AI 总结
        prompt = f"""基于以下搜索结果回答问题。

问题: {request.question}

搜索结果:
{context}

请基于上述信息给出准确的答案。"""
        
        answer = llm.generate(prompt)
        
        # 4. 添加引用来源
        answer += "\n\n**🔍 参考来源:**\n"
        for i, r in enumerate(search_results[:3], 1):
            answer += f"{i}. [{r['title']}]({r['url']})\n"
        
        return ChatResponse(
            answer=answer,
            conversation_id=request.conversation_id or f"conv-{uuid.uuid4().hex[:8]}",
            retrieved_docs=[{
                "content": r['snippet'],
                "source": r['url'],
                "title": r['title']
            } for r in search_results[:3]],
            metadata={"mode": "search", "result_count": len(search_results)}
        )
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="搜索功能需要安装 duckduckgo-search: pip install duckduckgo-search"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"搜索失败: {str(e)}"
        )


@router.post(
    "/stream",
    summary="流式聊天",
    description="使用服务器发送事件（SSE）流式返回答案"
)
async def chat_stream(
    request: ChatRequest,
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager),
    llm: OpenAILLM = Depends(get_llm)
):
    """
    流式聊天端点
    
    使用 SSE (Server-Sent Events) 流式返回生成的答案。
    
    TODO: 需要实现流式生成节点
    """
    # 检查知识库是否存在
    if request.kb_name not in kb_manager.knowledge_bases:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"知识库 '{request.kb_name}' 不存在"
        )
    
    async def event_generator() -> AsyncIterator[str]:
        """生成 SSE 事件流"""
        try:
            # 获取知识库
            kb = kb_manager.get_kb(request.kb_name)
            
            # 生成对话 ID
            conversation_id = request.conversation_id or f"conv-{uuid.uuid4().hex[:8]}"
            
            # 发送开始事件
            yield f"data: {json.dumps({'type': 'start', 'conversation_id': conversation_id})}\n\n"
            
            # TODO: 使用流式生成节点
            # 这里使用简单的模拟
            answer = f"这是对问题 '{request.question}' 的流式回答..."
            for char in answer:
                yield f"data: {json.dumps({'type': 'token', 'content': char})}\n\n"
            
            # 发送结束事件
            yield f"data: {json.dumps({'type': 'end', 'conversation_id': conversation_id})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
