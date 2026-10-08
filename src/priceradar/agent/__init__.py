"""Agent 层（骨架先行）：工具注册表 + 单轮管线。

- tools.py：确定性能力的统一封装（未来 LangGraph 节点与 MCP Server 共用同一套工具）
- pipeline.py：单轮管线的确定性版本（关键词 → 卡）；LLM 节点（意图/报告）在 M1 接入

说明：本包不在此处预导入子模块，避免 `python -m priceradar.agent.pipeline` 触发
"module found in sys.modules" 警告；按需直接导入子模块即可。
"""
