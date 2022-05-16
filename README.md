# U.S. Patent Phrase to Phrase Matching

| date | pretrain model | loss | method | fold | CV | Pub |
| - | - | - | - | - | - | - | 
| 2022-05-15 | deberta-v3-large | mse | combined | 5 | 0.8539 | 0.8326 |
| 2022-05-16 | deberta-v3-large | mse | combined | 10 | 0.8588 | 0.8351 |
| 2022-05-17 | 

## 说明

- `combine`: baseline方法
- `split`：

## 计划


- [ ] split + 1层全连接（3->1)
- [ ] split + n层全连接（3->1)

- [ ] 固定主干网络finetune
- 实现连续度量学习的损失函数
- 尝试其他的网络结构设计
- 使用adaboost进行集成