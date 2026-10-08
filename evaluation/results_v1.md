# 同款匹配评测报告 v1（构造集）

数据集：`evaluation/dataset_v1.jsonl`（构造法 + 真实标题种子，方法论见 build_dataset.py 文件头）（218 对）

## 总体指标

| 指标 | 值 |
|---|---|
| 总体准确率 | 100.0%（218/218） |
| 同款检出 Precision | 100.0%（TP=56, FP=0） |
| 同款检出 Recall | 100.0%（FN=0） |

## 分类目明细

| kind | 总数 | 正确 | 准确率 |
|---|---|---|---|
| accessory_suffix | 10 | 10 | 100% |
| accessory_wrap | 10 | 10 | 100% |
| brand_alias_swap | 5 | 5 | 100% |
| brand_swap | 4 | 4 | 100% |
| candidate_missing_model | 3 | 3 | 100% |
| capacity_swap | 2 | 2 | 100% |
| fluff_injection | 18 | 18 | 100% |
| fullwidth_variant | 18 | 18 | 100% |
| model_swap | 5 | 5 | 100% |
| no_spec_cap_b | 3 | 3 | 100% |
| prefix_noise | 18 | 18 | 100% |
| punctuation_variant | 18 | 18 | 100% |
| real_accessory_pair | 1 | 1 | 100% |
| real_cross_platform | 1 | 1 | 100% |
| segment_reorder | 18 | 18 | 100% |
| suffix_fluff | 18 | 18 | 100% |
| suspicious_price | 3 | 3 | 100% |
| unrelated | 60 | 60 | 100% |
| year_swap | 3 | 3 | 100% |

## 错例清单（0 条）

（无）

> 说明：v1 为构造评测集，指标用于回归护栏与已知错误类型的强弱分析；
> 真实人工标注集（v2，200-500 对）按 PRD §5.7 计划扩充。
