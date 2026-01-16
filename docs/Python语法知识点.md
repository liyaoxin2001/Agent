# Python 语法知识点详解

## 一、类和对象相关

### 1. `super()` 函数

```python
super().__init__(model_name, temperature)
```

**知识点**：
- `super()` 返回父类的代理对象
- 用于调用父类的方法
- 在子类中，`super().__init__()` 调用父类的构造函数
- 这是 Python 中实现继承的标准方式

**为什么使用**：
- 确保父类的初始化代码被执行
- 避免重复代码
- 遵循面向对象的设计原则

**示例对比**：
```python
# 不使用 super()（不推荐）
class Child(Parent):
    def __init__(self):
        Parent.__init__(self)  # 直接调用父类

# 使用 super()（推荐）
class Child(Parent):
    def __init__(self):
        super().__init__()  # 更灵活，支持多继承
```

### 2. `self` 关键字

```python
self.llm = ChatOpenAI(...)
self.model_name = model_name
```

**知识点**：
- `self` 是实例的引用
- 通过 `self` 访问实例变量和方法
- Python 约定使用 `self`，不是关键字，可以改名但不推荐

**示例**：
```python
class MyClass:
    def __init__(self, name):
        self.name = name  # 实例变量
    
    def get_name(self):
        return self.name  # 通过 self 访问实例变量
```

### 3. `@abstractmethod` 装饰器

```python
@abstractmethod
def generate(self, prompt: str) -> str:
    pass
```

**知识点**：
- `@abstractmethod` 是装饰器（decorator）
- 用于定义抽象方法（必须被子类实现）
- 如果子类不实现抽象方法，实例化时会报错
- 需要配合 `ABC`（抽象基类）使用

**示例**：
```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass  # 子类必须实现这个方法

class Dog(Animal):
    def make_sound(self):
        return "汪汪"  # 必须实现，否则无法实例化
```

## 二、类型注解

### 1. 函数参数类型注解

```python
def generate(self, prompt: str, **kwargs) -> str:
```

**知识点**：
- `prompt: str` 表示参数 `prompt` 的类型是 `str`
- `-> str` 表示返回值类型是 `str`
- 类型注解是可选的，但推荐使用（提高代码可读性）
- Python 3.5+ 支持类型注解

**示例**：
```python
# 有类型注解
def add(a: int, b: int) -> int:
    return a + b

# 没有类型注解（也可以工作）
def add(a, b):
    return a + b
```

### 2. `**kwargs` 参数

```python
def generate(self, prompt: str, **kwargs) -> str:
```

**知识点**：
- `**kwargs` 表示接收任意数量的关键字参数
- `**` 是解包操作符
- `kwargs` 是一个字典，包含所有传入的关键字参数
- 常用于函数需要接受额外参数但不确定具体是什么参数时

**示例**：
```python
def my_function(name: str, **kwargs):
    print(f"Name: {name}")
    print(f"Other args: {kwargs}")

my_function("Alice", age=25, city="Beijing")
# 输出：
# Name: Alice
# Other args: {'age': 25, 'city': 'Beijing'}
```

## 三、列表和数据结构

### 1. 列表创建

```python
messages = [message]
```

**知识点**：
- `[]` 是列表字面量语法
- `[message]` 创建一个包含一个元素的列表
- 列表是可变的（mutable），可以添加、删除元素

**示例**：
```python
# 创建空列表
empty_list = []

# 创建包含元素的列表
numbers = [1, 2, 3]
names = ["Alice", "Bob"]

# 创建包含一个元素的列表
single = [message]
```

### 2. 为什么需要列表？

```python
response = self.llm.invoke([message])  # 需要列表，不是单个消息
```

**知识点**：
- `invoke()` 方法接受消息列表，支持多轮对话
- 即使只有一个消息，也要放在列表中
- 这样可以支持对话历史

**示例**：
```python
# 单轮对话
messages = [HumanMessage(content="你好")]

# 多轮对话
messages = [
    HumanMessage(content="你好"),
    AIMessage(content="你好！有什么可以帮助你的吗？"),
    HumanMessage(content="介绍一下 Python")
]
```

## 四、对象属性和方法调用

### 1. 点号操作符

```python
response.content
self.llm.invoke(messages)
```

**知识点**：
- `.` 是属性访问操作符
- `object.attribute` 访问对象的属性
- `object.method()` 调用对象的方法
- 这是面向对象编程的基础

**示例**：
```python
class Person:
    def __init__(self, name):
        self.name = name  # 属性
    
    def greet(self):  # 方法
        return f"Hello, I'm {self.name}"

person = Person("Alice")
print(person.name)  # 访问属性
print(person.greet())  # 调用方法
```

### 2. 方法链式调用

```python
message = HumanMessage(content=prompt)
```

**知识点**：
- `HumanMessage` 是类的构造函数调用
- `content=prompt` 是关键字参数
- 创建对象后赋值给变量

**示例**：
```python
# 创建对象
message = HumanMessage(content="Hello")

# 等价于
message = HumanMessage(content="Hello")
```

## 五、返回语句

### 1. `return` 语句

```python
return answer
```

**知识点**：
- `return` 用于从函数返回一个值
- 函数执行到 `return` 就结束
- 可以返回任何类型的值
- 如果没有 `return`，函数返回 `None`

**示例**：
```python
def add(a, b):
    return a + b  # 返回计算结果

result = add(1, 2)  # result = 3

def no_return():
    print("Hello")  # 没有 return，返回 None
```

## 六、环境变量和配置

### 1. `os.getenv()`

```python
api_key=os.getenv("OPENAI_API_KEY")
```

**知识点**：
- `os.getenv()` 获取环境变量的值
- 第一个参数是环境变量名（字符串）
- 如果环境变量不存在，返回 `None`
- 可以设置默认值：`os.getenv("KEY", "default")`

**示例**：
```python
import os

# 获取环境变量
api_key = os.getenv("OPENAI_API_KEY")

# 带默认值
api_key = os.getenv("OPENAI_API_KEY", "default_key")

# 检查是否存在
if api_key:
    print("API Key 已设置")
else:
    print("API Key 未设置")
```

### 2. `dotenv.load_dotenv()`

```python
dotenv.load_dotenv()
```

**知识点**：
- `python-dotenv` 库用于从 `.env` 文件加载环境变量
- `.env` 文件通常包含敏感信息（如 API Key）
- 不应该将 `.env` 文件提交到 Git

**`.env` 文件格式**：
```
OPENAI_API_KEY=sk-xxxxx
OPENAI_BASE_URL=https://api.openai.com/v1
```

## 七、异常处理（建议添加）

### 1. `try-except` 语句

```python
try:
    # 可能出错的代码
    response = self.llm.invoke(messages)
except Exception as e:
    # 处理错误
    raise Exception(f"生成失败: {str(e)}")
```

**知识点**：
- `try` 块包含可能出错的代码
- `except` 块处理异常
- `Exception as e` 捕获异常并赋值给变量 `e`
- `raise` 重新抛出异常或抛出新异常

**示例**：
```python
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"错误: {e}")  # 错误: division by zero

try:
    api_call()
except Exception as e:
    print(f"API 调用失败: {e}")
    raise  # 重新抛出异常
```

## 八、代码组织最佳实践

### 1. 导入顺序

```python
# 标准库
import os

# 第三方库
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import dotenv

# 本地模块
# from src.xxx import xxx
```

**知识点**：
- 按照标准库、第三方库、本地模块的顺序导入
- 每组之间用空行分隔
- 提高代码可读性

### 2. 文档字符串（Docstring）

```python
def generate(self, prompt: str, **kwargs) -> str:
    """
    生成回答
    
    Args:
        prompt: 输入提示词
        **kwargs: 其他参数
        
    Returns:
        生成的文本
    """
```

**知识点**：
- 三引号 `"""` 用于多行字符串
- 文档字符串描述函数的功能、参数、返回值
- 可以使用工具自动生成文档
- 遵循 Google 或 NumPy 风格的文档字符串格式

## 九、总结

### 关键 Python 概念回顾

1. **类和对象**：`class`, `self`, `super()`
2. **类型注解**：`str`, `-> str`, `**kwargs`
3. **数据结构**：列表 `[]`
4. **属性访问**：`.` 操作符
5. **返回语句**：`return`
6. **环境变量**：`os.getenv()`, `dotenv`
7. **异常处理**：`try-except`
8. **文档字符串**：`"""..."""`

### 学习建议

1. **理解概念**：每个语法点都有其设计目的
2. **多练习**：通过编写代码加深理解
3. **查文档**：遇到不理解的语法，查看 Python 官方文档
4. **看示例**：参考其他项目的代码风格

