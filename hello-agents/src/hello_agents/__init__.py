from .llm import HelloAgentsLLM

llm = HelloAgentsLLM()

messages = [
    {"role": "system", "content": "You are a helpful assistant that writes Python code."},
    {"role": "user", "content": "写一个快速排序算法"}
]

llm.think(messages)
