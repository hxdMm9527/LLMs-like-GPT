# LLMs-like-GPT

从零复现 GPT 类大模型的核心组件，学习《LLMs-from-scratch》后的代码实践，包含斯坦福 CS336 课程作业的完整实现。

## 实现内容

- **BPE 分词器**：训练与推理（train BPE / tokenize）
- **Transformer 架构**：Embedding、Multi-Head Attention、前馈网络、LayerNorm、RoPE 旋转位置编码
- **注意力机制**：scaled dot-product attention（含 KV Cache 优化）
- **优化器**：AdamW 实现
- **模型序列化**：checkpoint 保存 / 加载
- **文本采样**：temperature / top-k 解码生成

## 运行

```bash
pip install -e .
pytest tests/       # 完整测试套件（含快照测试）
```

## 参考

- 《Build a Large Language Model (From Scratch)》（LLMs-from-scratch）
- Stanford CS336: Language Modeling from Scratch
