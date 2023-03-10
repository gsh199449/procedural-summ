import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from gat import GATConv


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)

class HANLayer(nn.Module):
    """
    HAN layer.
    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

    def forward(self, g, h):
        semantic_embeddings = []

        for i, meta_path in enumerate(self.meta_paths):
            if len(meta_path) != 2:
                new_g = dgl.metapath_reachable_graph(g, meta_path).to(g.device)
                semantic_embeddings.append(self.gat_layers[i](new_g, h, new=False).flatten(1))
            else:
                feature1 = []
                feature2 = []
                src = []
                tgt = []
                
                g_1 = g[meta_path[0]]
                g_2 = g[meta_path[1]]
                for j,s in enumerate(g_1.edges()[0]):
                    for t in torch.where(g_2.edges()[0] == g_1.edges()[1][j])[0]:
                        src.append(s)
                        tgt.append(g_2.edges()[1][t])
                        feature1.append(g.edges['contain'].data['sta'][j])
                        feature2.append(g.edges['contained'].data['stb'][t])
                
                new_g = dgl.graph((torch.tensor(src), torch.tensor(tgt)), num_nodes=h.shape[0]).to(g.device)
                new_g.edata['st'] = torch.cat((torch.stack(feature1), torch.stack(feature2)), dim=1)
                    
                semantic_embeddings.append(self.gat_layers[i](new_g, h, new=True).flatten(1))

        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)

class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        self.output = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        return self.output(h)
