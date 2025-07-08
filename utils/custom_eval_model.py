# custom_eval_model.py

import torch
import torch.nn as nn

def set_to_eval(model):
    """自定义的函数，用于设置模型为评估模式"""
    model.eval()  # 仍然调用 PyTorch 的 eval()，但是我们用自定义的函数包装
    torch.cuda.empty_cache()  # 释放缓存，避免内存泄漏
    return model

# 示例模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

if __name__ == "__main__":
    model = SimpleModel()  # 创建一个简单模型
    model = set_to_eval(model)  # 调用自定义的评估模式设置函数

    # 假设你已经有一些训练数据，并想保存模型
    try:
        torch.save(model.state_dict(), "model.pth")
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {e}")