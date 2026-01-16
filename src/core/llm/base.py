"""
LLM 基础接口定义

TODO: 你需要实现这个接口的具体类
参考 LangChain 的 LLM 接口设计
"""
from abc import ABC, abstractmethod
from typing import Iterator, Optional
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama  # 新版本
import os
import dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List, Union
import base64
from pathlib import Path

dotenv.load_dotenv()


class BaseLLM(ABC):# ABC抽象类，其中包括两个方法generate和stream_generate，任何继承这个接口的方法都要实现这两个方法
    """LLM 基础接口"""
    
    def __init__(self, model_name: str, temperature: float = 0.7):
        """
        初始化 LLM
        
        Args:
            model_name: 模型名称
            temperature: 温度参数
        """
        self.model_name = model_name
        self.temperature = temperature
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成回答
        
        Args:
            prompt: 输入提示词
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        pass
    
    @abstractmethod
    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        流式生成回答
        
        Args:
            prompt: 输入提示词
            **kwargs: 其他参数
            
        Yields:
            生成的文本片段
        """
        pass




# TODO: 实现 OpenAILLM 类
# 提示：
# 1. 继承 BaseLLM
# 2. 使用 langchain_openai.ChatOpenAI
# 3. 实现 generate 和 stream_generate 方法
# 4. 参考文档: https://python.langchain.com/docs/integrations/chat/openai

class OpenAILLM(BaseLLM):
    # 支持视觉功能的模型列表
    VISION_MODELS = [
        "gpt-4-vision-preview",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-1106-preview",
        "gpt-4-0125-preview",
    ]
    
    def __init__(self, model_name: str, temperature: float = 0.7):
        #初始化
        super().__init__(model_name, temperature)#是调用父类（BaseLLM）的构造函数。这样做的目的是为了确保父类中定义的初始化操作得以执行，即设置 self.model_name 和 self.temperature。
        # 初始化OpenAI客户端
        # 需要设置api_key等参数
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        
        # 检查模型是否支持视觉功能
        self.supports_vision = self._check_vision_support(model_name)
        
        self.llm = ChatOpenAI(
            # 必要的三个参数
            model=model_name,  # 模型名称 不设置默认使用 gpt-3.5-turbo
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
            model_kwargs={"max_tokens": 1000}  # 视觉模型需要更多token
        )
        
        # 如果模型不支持视觉但尝试使用图片，给出警告
        if not self.supports_vision:
            print(f"⚠️  警告: 模型 '{model_name}' 不支持视觉功能。如需使用图片理解，请切换到支持视觉的模型（如 gpt-4-turbo 或 gpt-4o）")
    
    @classmethod
    def _check_vision_support(cls, model_name: str) -> bool:
        """
        检查模型是否支持视觉功能
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 是否支持视觉功能
        """
        # 检查是否在支持视觉的模型列表中
        for vision_model in cls.VISION_MODELS:
            if vision_model.lower() in model_name.lower():
                return True
        return False
    def generate(self, prompt: str, images: List[str] = None, conversation_history: List[dict] = None, **kwargs) -> str:
        """
        生成回答
        
        Args:
            prompt: 输入提示词
            images: 图片数据列表（可选），支持：
                - base64数据URI格式：data:image/jpeg;base64,...
                - 本地文件路径（已废弃，建议使用base64）
            conversation_history: 对话历史（可选），格式：
                [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
            **kwargs: 其他参数
            
        Returns:
            生成的文本
            
        Raises:
            ValueError: 如果模型不支持视觉但提供了图片
        """
        # 检查是否提供了图片但模型不支持视觉
        if images and not self.supports_vision:
            raise ValueError(
                f"模型 '{self.model_name}' 不支持视觉功能。"
                f"请切换到支持视觉的模型（如 gpt-4-turbo 或 gpt-4o）。"
                f"支持的模型: {', '.join(self.VISION_MODELS)}"
            )
        
        try:
            # 构建消息内容
            content: List[Union[str, dict]] = []
            
            # 添加文本
            if prompt:
                content.append(prompt)
            
            # 添加图片（如果提供）
            if images:
                for image_data in images:
                    # 检查是否是base64数据URI格式
                    if image_data.startswith('data:image/'):
                        # 直接使用base64数据URI
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": image_data
                            }
                        })
                    elif Path(image_data).exists():
                        # 兼容旧版本：从文件路径读取（已废弃）
                        with open(image_data, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode('utf-8')
                        
                        # 获取图片MIME类型
                        ext = Path(image_data).suffix.lower()
                        mime_type = {
                            '.jpg': 'image/jpeg',
                            '.jpeg': 'image/jpeg',
                            '.png': 'image/png',
                            '.gif': 'image/gif',
                            '.webp': 'image/webp'
                        }.get(ext, 'image/jpeg')
                        
                        # 添加图片到消息内容
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{img_data}"
                            }
                        })
            
            # 构建消息列表
            messages = []
            
            # 添加系统提示，告诉AI它的名字是"花花"
            system_prompt = """你是一个友好的AI助手，名字叫"花花"。当用户问你的名字时，你应该回答"我叫花花"或"我是花花"。请用友好、自然的语气与用户交流。"""
            messages.append(SystemMessage(content=system_prompt))
            
            # 如果有对话历史，先添加历史消息
            if conversation_history:
                from langchain_core.messages import AIMessage
                for hist_msg in conversation_history:
                    role = hist_msg.get('role', '')
                    hist_content = hist_msg.get('content', '')
                    hist_images = hist_msg.get('images', [])
                    
                    if role == 'user':
                        # 构建用户消息（可能包含图片）
                        hist_content_list: List[Union[str, dict]] = []
                        if hist_content:
                            hist_content_list.append(hist_content)
                        if hist_images:
                            for img_data in hist_images:
                                if img_data.startswith('data:image/'):
                                    hist_content_list.append({
                                        "type": "image_url",
                                        "image_url": {"url": img_data}
                                    })
                        if hist_content_list:
                            messages.append(HumanMessage(content=hist_content_list if len(hist_content_list) > 1 else hist_content_list[0]))
                    elif role == 'assistant':
                        # 构建助手消息
                        if hist_content:
                            messages.append(AIMessage(content=hist_content))
            
            # 创建当前消息
            # 如果有图片，使用列表格式（包含文本和图片）
            # 如果只有文本，直接使用字符串
            if images and len(content) > 0:
                current_message = HumanMessage(content=content)
            elif content:
                current_message = HumanMessage(content=content[0] if isinstance(content, list) else content)
            else:
                current_message = HumanMessage(content=prompt)
            messages.append(current_message)
            
            # 使用 self.llm.invoke() 方法
            response = self.llm.invoke(messages)
            # response 是 AIMessage 对象，需要提取 content
            answer = response.content
            if not answer:
                raise ValueError("AI回答为空")
            return answer

        except Exception as e:
            # 捕获所有异常，包装成更友好的错误信息
            raise Exception(f"生成回答失败: {str(e)}") from e

    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
            """
              流式生成回答

              与 generate() 的区别：
              - 使用 stream() 而不是 invoke()
              - 使用 yield 而不是 return
              - 返回迭代器，可以逐步获取文本片段
              """
            try:
            # 步骤 1: 转换消息（与 generate 相同）
                # 添加系统提示，告诉AI它的名字是"花花"
                system_prompt = """你是一个友好的AI助手，名字叫"花花"。当用户问你的名字时，你应该回答"我叫花花"或"我是花花"。请用友好、自然的语气与用户交流。"""
                message = HumanMessage(content=prompt)
                messages = [SystemMessage(content=system_prompt), message]

            # 步骤 2: 使用 stream() 获取迭代器
            # stream() 返回一个迭代器，每次迭代得到一个 chunk
                for chunk in self.llm.stream(messages):
                    # chunk 是 AIMessageChunk 对象
                    # 步骤 3: 提取内容并 yield（不是 return）
                    yield chunk.content
                    # yield 是生成器的关键字，每次返回一个片段
                    # 函数不会结束，会继续执行下一次循环
            except Exception as e:
                # 捕获所有异常，包装成更友好的错误信息
                raise Exception(f"生成回答失败: {str(e)}") from e

# TODO: 实现 OllamaLLM 类（可选）
# 提示：
# 1. 继承 BaseLLM
# 2. 使用 langchain_community.llms.Ollama
# 3. 需要本地运行 Ollama 服务

class OllamaLLM(BaseLLM):
     def __init__(self, model_name: str, temperature: float = 0.7):
            super().__init__(model_name, temperature)
            # Ollama 默认运行在 http://localhost:11434
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.llm =ChatOllama(
                model=model_name,
                temperature=temperature,
                base_url=base_url
            )
     def generate(self, prompt: str, **kwargs) -> str:
        try:
             message = HumanMessage(content=prompt)
             messages = [message]
             response = self.llm.invoke(messages)
             answer = response.content
             if not answer:
                raise ValueError("AI回答为空")
             return answer
        except Exception as e:
            # 捕获所有异常，包装成更友好的错误信息
            raise Exception(f"生成回答失败: {str(e)}") from e

     def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        """
                     流式生成回答

                     与 generate() 的区别：
                     - 使用 stream() 而不是 invoke()
                     - 使用 yield 而不是 return
                     - 返回迭代器，可以逐步获取文本片段
                     """
        try:
            # 步骤 1: 转换消息（与 generate 相同）
            message = HumanMessage(content=prompt)
            messages = [message]

            # 步骤 2: 使用 stream() 获取迭代器
            # stream() 返回一个迭代器，每次迭代得到一个 chunk
            for chunk in self.llm.stream(messages):
                # chunk 是 AIMessageChunk 对象
                # 步骤 3: 提取内容并 yield（不是 return）
                yield chunk.content
                # yield 是生成器的关键字，每次返回一个片段
                # 函数不会结束，会继续执行下一次循环
        except Exception as e:
            # 捕获所有异常，包装成更友好的错误信息
            raise Exception(f"生成回答失败: {str(e)}") from e

# 测试类
if __name__ == '__main__':
    # 调用对话模型
    chat_model1 = OpenAILLM(
        # 必要的三个参数
        model_name="gpt-4o-mini",  # 模型名称 不设置默认使用 gpt-3.5-turbo
        # openai系列的sdk，包括langchain，都需要这个/v1的后缀
    )
    response1=chat_model1.generate("什么是langchain？")
    print(response1)

    print("--------------------------")
    response2=chat_model1.stream_generate("什么是langchain？")
    for chunk in response2:  # ✅ 遍历迭代器
        print(chunk, end="", flush=True)  # 逐个打印片段

    print("--------------------------")

    chat_model2 = OllamaLLM(
        model_name="qwen2.5:7b",
        temperature=0.7
    )

    response3=chat_model2.generate("什么是langchain？")
    print(response3)

    print("--------------------------")
    response4=chat_model2.stream_generate("什么是langchain？")
    for chunk in response4:
        print(chunk, end="", flush=True)