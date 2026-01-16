"""
FastAPI 应用启动脚本

使用 uvicorn 启动 FastAPI 应用。

运行方式:
    python run_api.py
    
    或指定端口:
    python run_api.py --port 8080
"""
import sys
import io
from pathlib import Path
import argparse

# 设置 Windows 控制台 UTF-8 编码
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent)
sys.path.insert(0, project_root)


def main():
    """
    启动 FastAPI 应用
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="启动 HuahuaChat API 服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（默认: 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8000, help="监听端口（默认: 8000）")
    parser.add_argument("--reload", action="store_true", help="启用自动重载（开发模式）")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数（默认: 1）")
    
    args = parser.parse_args()
    
    # 导入 uvicorn
    try:
        import uvicorn
    except ImportError:
        print("❌ 错误: 未找到 uvicorn")
        print("请运行: pip install uvicorn")
        sys.exit(1)
    
    # 启动配置
    print(f"\n{'='*70}")
    print(f"🚀 启动 HuahuaChat API 服务")
    print(f"{'='*70}")
    print(f"  监听地址: {args.host}:{args.port}")
    print(f"  自动重载: {'是' if args.reload else '否'}")
    print(f"  工作进程: {args.workers}")
    print(f"{'='*70}\n")
    
    # 启动服务
    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,  # reload 模式下只能单进程
        log_level="info"
    )


if __name__ == "__main__":
    main()
