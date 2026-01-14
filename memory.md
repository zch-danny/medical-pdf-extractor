# Medical PDF Extractor - 项目记忆文档

## 项目概述
医学PDF结构化信息提取系统，使用Qwen3-8B模型从医学文献中提取结构化信息。

## 当前状态
- **版本**: v7.9 分流架构版
- **vLLM配置**: Qwen3-8B, fp16, max_model_len=16384
- **主文件**: `production_extractor_v79.py`

## 架构设计

### 分流策略
```
文档页数 → 策略选择
  ≤50页  → single_short (智能选页，一次性提取)
  51-80页 → single_long (选60页，一次性提取)
  >80页  → mapreduce (先选80页，分块提取后合并)
```

### 页面选择算法
- 标题信号: Abstract/Introduction/Methods/Results/Conclusion等
- 关键词信号: recommend/should/Class I/Level A等（按文档类型）
- 密度信号: 字符数、段落数
- 位置加分: 前5页、后3页

## 测试结果（同一批20个PDF）

| 版本 | 平均评分 | ≥8分 | 耗时 | 短文档 | 长文档 |
|------|---------|------|------|--------|--------|
| v7.2 | 7.40 | 50% | 51s | - | - |
| v7.3 | 7.35 | 50% | 46s | 8.36 | 5.33 |
| v7.6 | 7.50 | 42% | 41s | - | - |
| v7.9 | 6.33 | 0% | 60s | 6.50 | 6.00 |

**关键发现**：
- 短文档效果最好（v7.3达8.36分）
- 长文档始终是瓶颈（4-6分）
- 32K上下文对准确度无明显提升，只增加覆盖率

## 待优化方向

### 高优先级：Milvus语义检索集成
**背景**：用户已有Milvus+BGE-M3向量数据库，后续会换成Qwen3-embedding

**优化方案**：
```
当前：PDF → 启发式关键词选页 → LLM提取
优化：PDF → Milvus语义检索选页 → LLM提取
```

**待确认**：
1. Milvus存储粒度（整页/段落/固定chunk）
2. 是否存储页码信息
3. Collection结构

**预期收益**：长文档准确度+1~2分

### 中优先级：提示词优化
1. **Quote-then-Structure**：先摘录原文+页码，再转结构化JSON
2. **JSON Schema约束**：明确必填字段和格式
3. **思维链引导**：让模型先分析再输出

### 低优先级：长文档分块策略
1. 先选后分块（已实现，效果一般）
2. 锚点窗口检索
3. 自适应chunk大小
4. Map→验证补全→Reduce

## 关键配置

### vLLM启动
```bash
# 16K版本（当前使用）
bash /root/autodl-tmp/vllm_server_v13.sh

# 32K版本（备用，已测试无明显准确度提升）
bash /root/autodl-tmp/vllm_server_v14_32k.sh
```

### 测试命令
```bash
python3 test_v79.py /root/autodl-tmp/pdf_input/pdf_input_09-1125/ 20
```

### GPT评估API
- URL: https://api.bltcy.ai/v1/chat/completions
- Model: gpt-5.2

## 历史决策

1. **放弃32K上下文**：测试显示相同输入下，32K不提升准确度，只增加耗时
2. **分流架构**：短文档用v7.3策略效果最好，长文档用MapReduce
3. **先选后分块**：对长文档效果有限，核心问题是页面选择不够精准

## 下一步行动
1. 等待用户确认Milvus结构后，开发语义检索集成
2. 优化提示词（Quote-then-Structure）
