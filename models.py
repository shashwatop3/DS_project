import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import os
from torch_geometric.nn import GATConv
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Subset
from collections import defaultdict


HIDDEN_SIZE = 16

class HierarchicalStockDataset(Dataset):
    def __init__(self, multiindex_df, sequence_length=5):
        self.sequence_length = sequence_length
        self.samples = []


        self.industry_map = {industry: idx for idx, industry in enumerate(multiindex_df.columns.levels[0])}
        self.company_map = {company: idx for idx, company in enumerate(multiindex_df.columns.levels[1])}

        for industry_id, industry in enumerate(multiindex_df.columns.levels[0]):
            for company_id, company in enumerate(multiindex_df.columns.levels[1]):
                if industry not in self.industry_map:
                    self.industry_map[industry] = industry_id
                if company not in self.company_map:
                    self.company_map[company] = company_id
                
                try:

                    company_df = multiindex_df.xs((industry, company), axis=1, level=[0,1]).copy()

                    if 'return_ratio' not in company_df.columns:
                      
                        
                        continue
                        
                    features = company_df.drop('return_ratio', axis=1).values
                    labels = company_df['return_ratio'].values
                    

                    for i in range(len(features) - self.sequence_length-1):
                        self.samples.append({
                            'features': features[i:i+self.sequence_length],
                            'industry': self.industry_map[industry],
                            'company': self.company_map[company],
                            'return_ratio': labels[i + self.sequence_length+1],
                            'movements': (labels[i + self.sequence_length + 1]>0),
                        })
                except KeyError:
                    print(f"Could not find data for industry={industry}, company={company}")
                    continue

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample['features'], dtype=torch.float32),  
            torch.tensor(sample['industry'], dtype=torch.long),
            torch.tensor(sample['company'], dtype=torch.long),
            torch.tensor(sample['return_ratio'], dtype=torch.float32),
            torch.tensor(sample['movements'], dtype=torch.float32)  
        )
        
    def __len__(self):
        return len(self.samples)

class TransformerSequentialLearner(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        
       
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, hidden_size)) 
        
        self.input_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead, 
            dim_feedforward=hidden_size*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):

        seq_len = x.size(1)
        
        x = self.input_proj(x)

        x = x + self.pos_encoder[:, :seq_len, :]
        
        transformer_output = self.transformer_encoder(x)
        

        context = torch.mean(transformer_output, dim=1)
        
        return context, transformer_output
    
class IntraSectorGAT(torch.nn.Module):
    def __init__(self, HIDDEN_SIZE=HIDDEN_SIZE):
        super().__init__()
        
        self.model_registry = {}
        self.out_dim = HIDDEN_SIZE  
    
    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        if x.dim() != 2:
            x = x.reshape(-1, x.size(-1))

        in_channels = x.size(1)
        
        if in_channels not in self.model_registry:
            print(f"Creating new GATConv model for input dimension {in_channels}")
            conv = GATConv(
                in_channels=in_channels,
                out_channels=self.out_dim // 4,
                heads=4,
                concat=True,
                negative_slope=0.2,
                dropout=0.1
            ).to(x.device)
            self.model_registry[in_channels] = conv

        conv = self.model_registry[in_channels]
        x = conv(x, edge_index)
        
        return x
    

class LongTermTransformerLearner(nn.Module):
    def __init__(self, input_size, hidden_size, lookback_weeks=4, num_layers=2, nhead=4):
        super().__init__()
        self.lookback = lookback_weeks
        self.transformer = TransformerSequentialLearner(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nhead=nhead
        )
        
    def forward(self, sequential_embeddings):

        seq_window = torch.stack(sequential_embeddings[-self.lookback:], dim=1)

        context, _ = self.transformer(seq_window)
        return context


class InterSectorGAT(torch.nn.Module):
    def __init__(self, in_channels=16, hidden_channels=16):
        super().__init__()
       
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels // 4,
            heads=4,
            concat=True
        )
        self.conv2 = GATConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            heads=1,
            concat=False
        )
    
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class EmbeddingFusion(nn.Module):
    def __init__(self, attentive_dim, graph_dim, sector_dim, output_dim=64):
        super().__init__()
        self.input_dim = attentive_dim + graph_dim + sector_dim
        self.fusion_layer = nn.Linear(self.input_dim, output_dim)
        
    def forward(self, attentive_emb, graph_emb, sector_emb):
        """
        Fuses three embedding types into a single representation
        """
        # Normalize dimensions
        if graph_emb.dim() > 2:
            graph_emb = graph_emb.squeeze(1)
        if attentive_emb.dim() > 2:
            attentive_emb = attentive_emb.squeeze(1)
        
        # Three-way concatenation
        concatenated = torch.cat([graph_emb, attentive_emb, sector_emb], dim=-1)
        
        # Apply fusion transformation with ReLU
        fused = F.relu(self.fusion_layer(concatenated))
        
        return fused
    
class FinGAT(nn.Module):
    def __init__(self, attentive_dim=16, graph_dim=16, sector_dim=16, hidden_dim=64):
        super().__init__()

        self.fusion = EmbeddingFusion(
            attentive_dim=attentive_dim,
            graph_dim=graph_dim,
            sector_dim=sector_dim,
            output_dim=hidden_dim
        )
        

        self.return_ratio_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

        self.movement_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        
    def forward(self, attentive_emb, graph_emb, sector_emb):

        fused_embedding = self.fusion(attentive_emb, graph_emb, sector_emb)

        return_ratio_pred = self.return_ratio_layer(fused_embedding).squeeze(-1)
        movement_pred = torch.sigmoid(self.movement_layer(fused_embedding)).squeeze(-1)

        return_ratio_pred = return_ratio_pred.view(-1)
        movement_pred = movement_pred.view(-1)
        
        return return_ratio_pred, movement_pred
    
class MultiTaskLoss:
    def __init__(self, alpha=0.5):
        """
        Multi-task loss without auxiliary return prediction
        Args:
            alpha: Weight between ranking loss and movement loss
        """
        self.alpha = alpha
        self.bce_loss = nn.BCELoss()
        
    def pairwise_ranking_loss(self, predictions, targets):
        # Same as before
        n = predictions.size(0)
        pairs_i, pairs_j = torch.triu_indices(n, n, offset=1)
        
        pred_diff = predictions[pairs_i] - predictions[pairs_j]
        target_diff = targets[pairs_i] - targets[pairs_j]
        
        target_sign = torch.sign(target_diff)
        
        margin = 0.1
        loss = F.relu(-target_sign * pred_diff + margin)
        
        return loss.mean()
    
    def __call__(self, return_preds, return_targets, move_preds, move_targets):
        """
        Calculate combined multi-task loss
        """
        # Ensure consistent shapes
        return_preds = return_preds.view(-1)
        return_targets = return_targets.view(-1)
        move_preds = move_preds.view(-1)
        move_targets = move_targets.view(-1)
        
        # Calculate main losses
        ranking_loss = self.pairwise_ranking_loss(return_preds, return_targets)
        movement_loss = self.bce_loss(move_preds, move_targets)
        
        # Combined loss
        combined_loss = self.alpha * ranking_loss + (1 - self.alpha) * movement_loss
        
        return combined_loss, ranking_loss, movement_loss