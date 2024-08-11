import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AdamW, get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import logging
import json
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from scipy.stats import beta
from torch.cuda.amp import autocast, GradScaler
import wandb
from captum.attr import IntegratedGradients
import networkx as nx
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import praw
from collections import Counter
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime

class RedditBehavioralModel(nn.Module):
    def __init__(self, config: Dict[str, Union[str, int, float]]):
        super(RedditBehavioralModel, self).__init__()
        self.text_encoder = AutoModel.from_pretrained(config['text_model_name'])
        
        self.subreddit_embedding = nn.Embedding(config['num_subreddits'], 64)
        self.karma_encoder = nn.Linear(1, 32)
        self.account_age_encoder = nn.Linear(1, 32)
        
        combined_dim = self.text_encoder.config.hidden_size + 64 + 32 + 32
        
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=combined_dim, nhead=8, dim_feedforward=2048),
            num_layers=4
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(512, config['num_labels'])
        )
        
        self.attention = nn.MultiheadAttention(combined_dim, num_heads=8)

    def forward(self, text_input_ids: torch.Tensor, text_attention_mask: torch.Tensor, 
                subreddit_ids: torch.Tensor, karma: torch.Tensor, account_age: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        text_output = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask).last_hidden_state
        subreddit_output = self.subreddit_embedding(subreddit_ids).unsqueeze(1)
        karma_output = self.karma_encoder(karma.unsqueeze(-1)).unsqueeze(1)
        account_age_output = self.account_age_encoder(account_age.unsqueeze(-1)).unsqueeze(1)
        
        combined_output = torch.cat((text_output, subreddit_output, karma_output, account_age_output), dim=1)
        
        fused_output = self.fusion_transformer(combined_output)
        
        attn_output, attn_weights = self.attention(fused_output, fused_output, fused_output)
        
        pooled_output = torch.mean(attn_output, dim=1)
        
        logits = self.classifier(pooled_output)
        
        return logits, attn_weights

class RedditDataset(Dataset):
    def __init__(self, reddit_data: List[Dict], tokenizer: AutoTokenizer, config: Dict[str, Union[str, int, float]]):
        self.reddit_data = reddit_data
        self.tokenizer = tokenizer
        self.config = config
        self.subreddit_to_id = {subreddit: idx for idx, subreddit in enumerate(config['subreddits'])}

    def __len__(self) -> int:
        return len(self.reddit_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.reddit_data[idx]
        
        encoding = self.tokenizer.encode_plus(
            item['text'],
            add_special_tokens=True,
            max_length=self.config['max_length'],
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'text_input_ids': encoding['input_ids'].flatten(),
            'text_attention_mask': encoding['attention_mask'].flatten(),
            'subreddit_id': torch.tensor(self.subreddit_to_id[item['subreddit']], dtype=torch.long),
            'karma': torch.tensor(item['karma'], dtype=torch.float32),
            'account_age': torch.tensor(item['account_age'], dtype=torch.float32),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }

def fetch_reddit_data(config: Dict[str, Union[str, int, float]]) -> List[Dict]:
    reddit = praw.Reddit(client_id=config['reddit_client_id'],
                         client_secret=config['reddit_client_secret'],
                         user_agent=config['reddit_user_agent'])
    
    data = []
    for subreddit_name in config['subreddits']:
        subreddit = reddit.subreddit(subreddit_name)
        for submission in subreddit.hot(limit=config['posts_per_subreddit']):
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                data.append({
                    'text': comment.body,
                    'subreddit': subreddit_name,
                    'karma': comment.score,
                    'account_age': (datetime.datetime.now() - datetime.datetime.fromtimestamp(comment.author.created_utc)).days,
                    'label': 0  # You'll need to define how to label the data
                })
    
    return data

def analyze_reddit_patterns(reddit_data: List[Dict]) -> Dict[str, Any]:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_embeddings = sentence_model.encode([item['text'] for item in reddit_data])
    text_clusters = DBSCAN(eps=0.5, min_samples=3).fit(text_embeddings)
    
    subreddit_counter = Counter([item['subreddit'] for item in reddit_data])
    
    karma_distribution = [item['karma'] for item in reddit_data]
    
    account_age_distribution = [item['account_age'] for item in reddit_data]
    
    sid = SentimentIntensityAnalyzer()
    sentiments = [sid.polarity_scores(item['text'])['compound'] for item in reddit_data]
    
    emoji_counter = Counter()
    for item in reddit_data:
        emoji_counter.update(emoji.emoji_list(item['text']))
    
    G = nx.Graph()
    for i, item in enumerate(reddit_data):
        G.add_node(i, text=item['text'], subreddit=item['subreddit'], karma=item['karma'], account_age=item['account_age'])
    
    for i in range(len(reddit_data)):
        for j in range(i+1, len(reddit_data)):
            if reddit_data[i]['subreddit'] == reddit_data[j]['subreddit']:
                G.add_edge(i, j, weight=1)
            if abs(reddit_data[i]['karma'] - reddit_data[j]['karma']) < 10:
                G.add_edge(i, j, weight=2)
            if abs(reddit_data[i]['account_age'] - reddit_data[j]['account_age']) < 30:
                G.add_edge(i, j, weight=3)
    
    return {
        'text_clusters': text_clusters.labels_,
        'subreddit_distribution': subreddit_counter,
        'karma_distribution': karma_distribution,
        'account_age_distribution': account_age_distribution,
        'sentiment_distribution': sentiments,
        'emoji_usage': emoji_counter,
        'behavioral_graph': G
    }

def interpret_reddit_model_decision(model: nn.Module, tokenizer: AutoTokenizer, reddit_item: Dict, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    ig = IntegratedGradients(model)
    
    inputs = tokenizer(reddit_item['text'], return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs['subreddit_id'] = torch.tensor([reddit_item['subreddit_id']], dtype=torch.long).to(device)
    inputs['karma'] = torch.tensor([reddit_item['karma']], dtype=torch.float32).to(device)
    inputs['account_age'] = torch.tensor([reddit_item['account_age']], dtype=torch.float32).to(device)
    
    attributions, delta = ig.attribute(inputs=inputs, target=1, return_convergence_delta=True)
    
    return {
        'text_attributions': attributions['text_input_ids'].sum(dim=2).squeeze(0).cpu().numpy(),
        'subreddit_attributions': attributions['subreddit_id'].squeeze(0).cpu().numpy(),
        'karma_attributions': attributions['karma'].squeeze(0).cpu().numpy(),
        'account_age_attributions': attributions['account_age'].squeeze(0).cpu().numpy()
    }

def generate_reddit_intervention(behavioral_profile: Dict[str, Any], attributions: Dict[str, np.ndarray]) -> str:
    generator = pipeline('text-generation', model='gpt2')
    
    prompt = f"Generate a Reddit-specific intervention message for a user with the following profile: {json.dumps(behavioral_profile)}"
    intervention = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    
    return intervention

def train_reddit_model(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, 
                       config: Dict[str, Union[str, int, float]], device: torch.device):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], correct_bias=False)
    total_steps = len(train_dataloader) * config['num_epochs']
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=total_steps)
    
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    best_val_auc = 0
    
    wandb.init(project="reddit-behavioral-detection", config=config)
    
    for epoch in range(config['num_epochs']):
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}"):
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            subreddit_ids = batch['subreddit_id'].to(device)
            karma = batch['karma'].to(device)
            account_age = batch['account_age'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                logits, _ = model(text_input_ids, text_attention_mask, subreddit_ids, karma, account_age)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in val_dataloader:
                text_input_ids = batch['text_input_ids'].to(device)
                text_attention_mask = batch['text_attention_mask'].to(device)
                subreddit_ids = batch['subreddit_id'].to(device)
                karma = batch['karma'].to(device)
                account_age = batch['account_age'].to(device)
                labels = batch['label'].to(device)
                
                logits, _ = model(text_input_ids, text_attention_mask, subreddit_ids, karma, account_age)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_auc = roc_auc_score(all_labels, all_preds)
        
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_auc": val_auc
        })
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_reddit_model.pth')
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}")
    
    wandb.finish()

if __name__ == "__main__":
    config = {
        'text_model_name': 'bert-base-uncased',
        'num_labels': 2,
        'max_length': 512,
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 2e-5,
        'warmup_steps': 500,
        'dropout': 0.3,
        'reddit_client_id': 'your_client_id',
        'reddit_client_secret': 'your_client_secret',
        'reddit_user_agent': 'your_user_agent',
        'subreddits': ['AskReddit', 'worldnews', 'funny', 'gaming', 'aww'],
        'posts_per_subreddit': 100,
        'num_subreddits': 5
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    reddit_data = fetch_reddit_data(config)
    tokenizer = AutoTokenizer.from_pretrained(config['text_model_name'])
    
    dataset = RedditDataset(reddit_data, tokenizer, config)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size