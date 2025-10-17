
import penman
from transition_amr_parser.parse import AMRParser
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import AutoTokenizer, AutoModel
from requests.adapters import DEFAULT_RETRIES
import requests

requests.adapters.DEFAULT_RETRIES = 5  # 设置重试次数
requests.adapters.DEFAULT_TIMEOUT = 1200  # 设置超时时间，单位为秒




def AMR(parser,text):
    # Download and save a model named AMR3.0 to cache

    tokens, positions = parser.tokenize(text)

    # Use parse_sentence() for single sentences or parse_sentences() for a batch
    annotations, machines = parser.parse_sentence(tokens)

    # Print Penman notation
    return machines

def encode_node(text,model,tokenizer):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=10)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1)

def amr_to_graph(amr, model,tokenizer):
    """
    将 AMR 文本解析为 PyTorch Geometric 的图数据格式，并统一节点特征维度。
    """
    # 使用 Penman 解析 AMR 文本
    node_features = {}
    graph=amr.nodes
    edges=amr.edges
    for node, text in graph.items():
        node_features[node] = encode_node(text,model,tokenizer)

    # 将节点嵌入整理为特征矩阵，并移除多余维度
    node_index = {node: idx for idx, node in enumerate(graph.keys())}
    x = torch.cat([node_features[node] for node in graph.keys()], dim=0)  # 合并嵌入
    x = x.squeeze(1)  # 移除多余的维度，确保x是二维的 [num_nodes, num_features]

    # Step 3: 构建图的拓扑结构
    edge_index = []
    for src, _, dst in edges:
        edge_index.append([node_index[src], node_index[dst]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # 返回 PyTorch Geometric 的 Data 对象
    return Data(x=x, edge_index=edge_index)





# 定义 GNN 模型
class GNNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x


