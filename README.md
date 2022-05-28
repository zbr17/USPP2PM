# U.S. Patent Phrase to Phrase Matching

| date | pretrain model | loss | method | wd | lr | bs | cards | fold | CV | Pub |
| - | - | - | - | - | - | - | - | - | - | - |
| 2022-05-15 | deberta-v3-large | mse | combined | 0.01 | 0.00002 | 16 | 2 | 5 | 0.8539 | 0.8326 |
| 2022-05-16 | deberta-v3-large | mse | combined | 0.01 | 0.00002 | 16 | 2 | 10 | 0.8588 | 0.8351 |
| **2022-05-26** | **deberta-v3-base** | **mse** | **combined** | **0.01** | **0.00002** | **48** | **1** | **5** | **0.8516** | **0.8214** | 

## 计划


- [ ] split + 1层全连接（3->1)
- [ ] split + n层全连接（3->1)

- [ ] 固定主干网络finetune
- [ ] 设计串联集成模型（基于adaboost）（比较有希望的一种设计）
- [ ] 实现连续度量学习的损失函数
- [ ] 尝试其他的网络结构设计
- [ ] 使用adaboost进行集成