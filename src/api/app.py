"""
FastAPI 应用主文件

定义和配置 FastAPI 应用，包括路由、中间件、错误处理等。
"""
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from datetime import datetime
import traceback

from src.api.routers import chat, knowledge_base
from src.api.models import ErrorResponse, HealthResponse
from src.api.dependencies import get_service_container


# ============================================================
# 创建 FastAPI 应用
# ============================================================

app = FastAPI(
    title="HuahuaChat RAG API",
    description="""
    **HuahuaChat** 是一个基于 RAG（检索增强生成）的企业级智能问答系统。
    
    ## 功能特性
    
    - 🤖 **智能对话**: 基于知识库的 AI 问答
    - 📚 **知识库管理**: 创建、删除、查询知识库
    - 📄 **文档管理**: 上传、删除、列出文档
    - 🔍 **语义检索**: 基于向量相似度的文档检索
    - 🌊 **流式输出**: 支持流式生成答案（TODO）
    
    ## 技术栈
    
    - **框架**: FastAPI, LangChain, LangGraph
    - **LLM**: OpenAI GPT-3.5/4
    - **向量库**: FAISS
    - **文档处理**: PyPDF, UnstructuredIO
    
    ## API 版本
    
    当前版本: **v1.0.0**
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# ============================================================
# 中间件配置
# ============================================================

# CORS 中间件（允许跨域请求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 全局异常处理
# ============================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    处理请求验证错误
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            success=False,
            error="ValidationError",
            message="请求参数验证失败",
            detail=exc.errors()
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    处理未捕获的异常
    """
    # 打印完整的错误堆栈
    print(f"\n{'='*70}")
    print(f"❌ 未处理的异常:")
    print(f"{'='*70}")
    traceback.print_exc()
    print(f"{'='*70}\n")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            success=False,
            error="InternalServerError",
            message="服务器内部错误",
            detail=str(exc) if app.debug else None
        ).model_dump()
    )


# ============================================================
# 应用生命周期事件
# ============================================================

@app.on_event("startup")
async def startup_event():
    """
    应用启动时执行
    """
    print("\n" + "="*70)
    print("🚀 HuahuaChat API 正在启动...")
    print("="*70)
    
    # 初始化服务容器
    container = get_service_container()
    success = container.init_services()
    
    if success:
        print("✅ 所有服务初始化成功")
    else:
        print("⚠️  部分服务初始化失败，请检查配置")
    
    print("="*70)
    print("📖 API 文档: http://localhost:8000/docs")
    print("📖 ReDoc 文档: http://localhost:8000/redoc")
    print("="*70 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """
    应用关闭时执行
    """
    print("\n" + "="*70)
    print("👋 HuahuaChat API 正在关闭...")
    print("="*70 + "\n")


# ============================================================
# 路由注册
# ============================================================

# 注册聊天路由
app.include_router(chat.router)

# 注册知识库管理路由
app.include_router(knowledge_base.router)


# ============================================================
# 根端点和健康检查
# ============================================================

@app.get(
    "/",
    summary="API 根端点",
    description="返回 API 基本信息",
    tags=["Health"]
)
async def root():
    """
    API 根端点
    """
    return {
        "name": "HuahuaChat RAG API",
        "version": "1.0.0",
        "description": "企业级 RAG 智能问答系统",
        "docs": "/docs",
        "health": "/health"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="健康检查",
    description="检查 API 和各组件的运行状态",
    tags=["Health"]
)
async def health_check():
    """
    健康检查端点
    
    返回 API 和各组件的状态。
    """
    container = get_service_container()
    
    # 检查各组件状态
    components = {}
    
    # 检查 LLM
    try:
        llm = container.get_llm()
        components["llm"] = "ok" if llm else "unavailable"
    except Exception:
        components["llm"] = "error"
    
    # 检查 Embedding
    try:
        embedding = container.get_embedding()
        components["embedding"] = "ok" if embedding else "unavailable"
    except Exception:
        components["embedding"] = "error"
    
    # 检查 KnowledgeBaseManager
    try:
        kb_manager = container.get_kb_manager()
        components["kb_manager"] = "ok" if kb_manager else "unavailable"
    except Exception:
        components["kb_manager"] = "error"
    
    # 整体状态
    overall_status = "healthy" if all(
        status == "ok" for status in components.values()
    ) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        components=components
    )


# ============================================================
# 调试信息（仅开发环境）
# ============================================================

if app.debug:
    @app.get("/debug/routes", tags=["Debug"])
    async def debug_routes():
        """
        列出所有注册的路由（仅开发环境）
        """
        routes = []
        for route in app.routes:
            if hasattr(route, "methods"):
                routes.append({
                    "path": route.path,
                    "methods": list(route.methods),
                    "name": route.name
                })
        return {"routes": routes, "total": len(routes)}
