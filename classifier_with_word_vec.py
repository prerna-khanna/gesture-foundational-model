import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import argparse

from contrastive.augmenter import GestureAugmenter
from contrastive.losses import ContrastiveCombinedLoss
from contrastive.models import ContrastiveGRUClassifier

import train
from config import load_dataset_label_names
from embedding import load_embedding_label
from models import fetch_classifier
from plot import plot_matrix
from statistic import stat_acc_f1, stat_results
from utils import get_device, handle_argv, IMUDataset, load_classifier_config, prepare_classifier_dataset

def create_config_with_params(original_cfg, params):
    """
    Creates a new configuration by updating the original config with new parameters.
    """
    config_dict = {
        'seed': original_cfg.seed,
        'batch_size': params.get('batch_size', original_cfg.batch_size),
        'lr': params.get('lr', original_cfg.lr),
        'n_epochs': params.get('n_epochs', original_cfg.n_epochs),
        'warmup': params.get('warmup', original_cfg.warmup),
        'save_steps': original_cfg.save_steps,
        'total_steps': original_cfg.total_steps,
        'lambda1': original_cfg.lambda1,
        'lambda2': params.get('lambda2', original_cfg.lambda2),
        'pooling': original_cfg.pooling
    }
    return argparse.Namespace(**config_dict)

class SemanticLoss(nn.Module):
    def __init__(self, label_names, device, temperature=0.07):
        """
        Initialize SemanticLoss with both semantic and contrastive learning capabilities
        
        Args:
            label_names: List of gesture label names
            device: Device to run computations on
            temperature: Temperature parameter for contrastive loss scaling
        """
        super().__init__()
        self.device = device
        self.temperature = temperature
        
        # Initialize Word2Vec and embeddings
        self.word2vec = Word2Vec(vector_size=768, window=5, min_count=1, workers=4)
        
        # Projection networks for different spaces (keeping same dimensions as original)
        self.semantic_projection = nn.Linear(20, 32).to(device)  
        self.contrastive_projection = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        ).to(device)
        
        # Compute semantic similarities between gesture labels
        self.semantic_sims = self._compute_semantic_sims(label_names)
        
    def _compute_semantic_sims(self, label_names):
        """
        Compute semantic similarity matrix between gesture labels using Word2Vec
        """
        descriptions = [
            "Vertical upward motion starting from rest position, controlled ascent", 
            "Vertical downward motion with steady arm trajectory, controlled descent", 
            "Horizontal lateral movement to the left side, smooth arm translation", 
            "Horizontal lateral movement to the right side, smooth arm translation",
            "Circular wrist rotation moving clockwise, maintaining consistent radius",
            "Circular wrist rotation moving anticlockwise, maintaining consistent radius",
            "Rapid, sharp upward jerking motion with quick acceleration and immediate stop",
            "Rapid, sharp downward jerking motion with quick acceleration and immediate stop",
            "Abrupt lateral movement to the left with sudden acceleration and quick cessation",
            "Abrupt lateral movement to the right with sudden acceleration and quick cessation",
            "Angular path tracing four equal sides with crisp, precise 90-degree corner turns",
            "Smooth, continuous curved motion forming a perfect closed loop without corner breaks",
            "Geometric path creating three connected straight lines with distinct angular transitions",
            "Curved motion starting with an upward arc, then sharply hooking downward",
            "Continuous figure-eight path with smooth, symmetrical mid-point crossing"
        ]

        # Train Word2Vec on descriptions
        tokenized_descriptions = [desc.lower().split() for desc in descriptions]
        self.word2vec.build_vocab(tokenized_descriptions)
        self.word2vec.train(tokenized_descriptions, total_examples=len(tokenized_descriptions), epochs=100)

        # Get embeddings using different pooling strategies
        embeddings = []
        for desc in descriptions:
            tokens = desc.lower().split()
            token_embeddings = [torch.tensor(self.word2vec.wv[token]) for token in tokens]
            token_embeddings = torch.stack(token_embeddings)

            if self.pooling == "cls":
                # Simulate CLS token by using first token
                embedding = token_embeddings[0]
            elif self.pooling == "mean":
                # Mean pooling
                embedding = torch.mean(token_embeddings, dim=0)
            elif self.pooling == "max":
                # Max pooling
                embedding = torch.max(token_embeddings, dim=0)[0]

            embeddings.append(embedding)

        embeddings = torch.stack(embeddings).to(self.device)
        embeddings = F.normalize(embeddings, p=2, dim=1)
            
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.t())
        
        # Apply transformations (keeping same as original)
        sim_matrix = (sim_matrix - sim_matrix.min()) / (1 - sim_matrix.min())
        sim_matrix = torch.pow(sim_matrix, 3)
        
        very_similar = (sim_matrix > 0.95).float()
        somewhat_similar = ((sim_matrix > 0.85) & (sim_matrix <= 0.95)).float()
        different = (sim_matrix <= 0.85).float()
        
        sim_matrix = (very_similar * sim_matrix * 1.2 +
                    somewhat_similar * sim_matrix * 0.8 +
                    different * sim_matrix * 0.5)
        
        return sim_matrix
    
    def semantic_loss(self, embeddings, labels, base_margin=1.0):
        """
        Compute semantic loss with dynamic margins based on gesture similarity
        """
        embeddings = self.semantic_projection(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        dists = torch.cdist(embeddings, embeddings, p=2)
        loss = 0
        batch_size = embeddings.size(0)
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                sem_sim = self.semantic_sims[labels[i], labels[j]]
                
                margin = base_margin * (2.0 if sem_sim > 0.8 else 
                                      1.5 if sem_sim > 0.5 else 1.0)
                
                weight = torch.pow(sem_sim, 2) + 0.1
                loss += weight * torch.max(torch.tensor(0.0).to(self.device),
                                         margin - dists[i, j])
                    
        return loss / (batch_size * (batch_size - 1))
    
    def contrastive_loss(self, embeddings, labels):
        """
        Compute NT-Xent contrastive loss (keeping exactly same as original)
        """
        features = self.contrastive_projection(embeddings)
        features = F.normalize(features, dim=1)
        
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(mask.shape[0]).view(-1, 1).to(self.device),
            0
        )
        
        mask = mask * logits_mask
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
        
        return -mean_log_prob_pos.mean()
    
    def forward(self, embeddings, labels, epoch=0):
        """
        Combine semantic and contrastive losses with dynamic weighting (keeping same weights)
        """
        sem_loss = self.semantic_loss(embeddings, labels)
        cont_loss = self.contrastive_loss(embeddings, labels)
        
        semantic_weight = min(0.3, (epoch / 10) * 0.3)
        contrastive_weight = min(0.5, (epoch / 20) * 0.5)
        
        total_loss = sem_loss * semantic_weight + cont_loss * contrastive_weight
        
        return total_loss, {
            'semantic_loss': sem_loss.item(),
            'contrastive_loss': cont_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def __init__(self, label_names, device, temperature=0.07):
        """
        Initialize SemanticLoss with Word2Vec-based semantic embeddings
        
        Args:
            label_names: List of gesture label names
            device: Device to run computations on
            temperature: Temperature parameter for contrastive loss scaling
        """
        super().__init__()
        self.device = device
        self.temperature = temperature
        
        # Projection networks for different spaces
        self.semantic_projection = nn.Linear(20, 32).to(device)
        self.contrastive_projection = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        ).to(device)
        
        # Create rich descriptions for gestures
        self.descriptions = self._create_gesture_descriptions(label_names)
        
        # Train Word2Vec model on gesture descriptions
        self.word2vec_model = self._train_word2vec(self.descriptions)
        
        # Compute semantic similarities between gesture labels
        self.semantic_sims = self._compute_semantic_sims()
        
    def _create_gesture_descriptions(self, label_names):
        """
        Create detailed descriptions for each gesture to capture semantic meaning
        """
        descriptions = {
            'up': "vertical upward motion starting from rest position controlled ascent",
            'down': "vertical downward motion steady arm trajectory controlled descent",
            'left': "horizontal lateral movement left side smooth arm translation",
            'right': "horizontal lateral movement right side smooth arm translation", 
            'rotate_cw': "circular wrist rotation clockwise maintaining consistent radius",
            'rotate_ccw': "circular wrist rotation counterclockwise maintaining consistent radius",
            'jerk_up': "rapid sharp upward jerking motion quick acceleration immediate stop",
            'jerk_down': "rapid sharp downward jerking motion quick acceleration immediate stop",
            'jerk_left': "abrupt lateral movement left sudden acceleration quick cessation",
            'jerk_right': "abrupt lateral movement right sudden acceleration quick cessation",
            'square': "angular path tracing four equal sides crisp precise corner turns",
            'circle': "smooth continuous curved motion perfect closed loop without breaks",
            'triangle': "geometric path three connected straight lines angular transitions",
            'hook': "curved motion starting upward arc sharply hooking downward",
            'infinity': "continuous figure eight path smooth symmetrical crossing"
        }
        
        return [descriptions.get(name.lower(), f"{name} gesture") for name in label_names]

    def _train_word2vec(self, descriptions):
        """
        Train a Word2Vec model on gesture descriptions
        """
        # Tokenize descriptions
        sentences = [desc.lower().split() for desc in descriptions]
        
        # Train Word2Vec model
        model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
        
        return model

    def _compute_semantic_sims(self):
        """
        Compute semantic similarity matrix between gesture descriptions using Word2Vec
        """
        num_gestures = len(self.descriptions)
        sim_matrix = torch.zeros((num_gestures, num_gestures)).to(self.device)
        
        # Get document embeddings by averaging word vectors
        doc_vectors = []
        for desc in self.descriptions:
            words = desc.lower().split()
            word_vectors = [self.word2vec_model.wv[word] for word in words if word in self.word2vec_model.wv]
            doc_vector = np.mean(word_vectors, axis=0)
            doc_vectors.append(doc_vector)
        
        doc_vectors = np.array(doc_vectors)
        
        # Compute cosine similarities
        for i in range(num_gestures):
            for j in range(num_gestures):
                cos_sim = np.dot(doc_vectors[i], doc_vectors[j]) / (
                    np.linalg.norm(doc_vectors[i]) * np.linalg.norm(doc_vectors[j]))
                sim_matrix[i, j] = cos_sim
        
        # Apply transformations to enhance similarity structure
        sim_matrix = (sim_matrix - sim_matrix.min()) / (1 - sim_matrix.min())
        sim_matrix = torch.pow(sim_matrix, 3)
        
        # Threshold-based enhancement
        very_similar = (sim_matrix > 0.95).float()
        somewhat_similar = ((sim_matrix > 0.85) & (sim_matrix <= 0.95)).float()
        different = (sim_matrix <= 0.85).float()
        
        sim_matrix = (very_similar * sim_matrix * 1.2 +
                     somewhat_similar * sim_matrix * 0.8 +
                     different * sim_matrix * 0.5)
        
        return sim_matrix
    
    def semantic_loss(self, embeddings, labels, base_margin=1.0):
        """
        Compute semantic loss with dynamic margins based on gesture similarity
        """
        embeddings = self.semantic_projection(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        dists = torch.cdist(embeddings, embeddings, p=2)
        loss = 0
        batch_size = embeddings.size(0)
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                sem_sim = self.semantic_sims[labels[i], labels[j]]
                
                # Dynamic margin based on semantic similarity
                margin = base_margin * (2.0 if sem_sim > 0.8 else 
                                      1.5 if sem_sim > 0.5 else 1.0)
                
                # Weight loss based on similarity
                weight = torch.pow(sem_sim, 2) + 0.1
                loss += weight * torch.max(torch.tensor(0.0).to(self.device),
                                         margin - dists[i, j])
                    
        return loss / (batch_size * (batch_size - 1))
    
    def contrastive_loss(self, embeddings, labels):
        """
        Compute NT-Xent contrastive loss
        """
        features = self.contrastive_projection(embeddings)
        features = F.normalize(features, dim=1)
        
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(mask.shape[0]).view(-1, 1).to(self.device),
            0
        )
        
        mask = mask * logits_mask
        
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
        
        return -mean_log_prob_pos.mean()
    
    def forward(self, embeddings, labels, epoch=0):
        """
        Combine semantic and contrastive losses with dynamic weighting
        """
        sem_loss = self.semantic_loss(embeddings, labels)
        cont_loss = self.contrastive_loss(embeddings, labels)
        
        semantic_weight = min(0.3, (epoch / 10) * 0.3)
        contrastive_weight = min(0.5, (epoch / 20) * 0.5)
        
        total_loss = sem_loss * semantic_weight + cont_loss * contrastive_weight
        
        return total_loss, {
            'semantic_loss': sem_loss.item(),
            'contrastive_loss': cont_loss.item(),
            'total_loss': total_loss.item()
        }
    
def classify_embeddings(args, data, labels, label_index, training_rate, label_rate, balance=False, method=None):
    # contrastive + semantic learning
    try:
        train_cfg, model_cfg, dataset_cfg = load_classifier_config(args)
        label_names, label_num = load_dataset_label_names(dataset_cfg, label_index)
        device = get_device(args.gpu)
        
        print(f"Number of classes: {label_num}")
        print(f"Label names: {label_names}")
        
        # Calculate hidden dimension if not in config
        # Using 128 as default hidden dim - this is a common choice for gesture recognition
        hidden_dim = getattr(model_cfg, 'hidden_dim', 128)
        
        # Prepare data
        data_train, label_train, data_vali, label_vali, data_test, label_test = \
            prepare_classifier_dataset(data, labels, label_index=label_index, training_rate=training_rate,
                                     label_rate=label_rate, merge=model_cfg.seq_len, seed=train_cfg.seed,
                                     balance=balance)
        
        # Create datasets with augmentation only for training
        augmenter = GestureAugmenter()
        data_set_train = IMUDataset(data_train, label_train, pipeline=[augmenter.augment])
        data_set_vali = IMUDataset(data_vali, label_vali)  # No augmentation for validation
        data_set_test = IMUDataset(data_test, label_test)  # No augmentation for testing
        
        # Create dataloaders
        data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
        data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size)
        data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)

        # Initialize model with the calculated hidden dimension
        model = ContrastiveGRUClassifier(
            input_dim=data_train.shape[-1],  # 72 from your data
            hidden_dim=hidden_dim,           # Using our default or config value
            num_classes=label_num
        ).to(device)
        
        print(f"Model architecture initialized with hidden dimension: {hidden_dim}")
        
        # Initialize combined loss (classification + semantic + contrastive)
        criterion = ContrastiveCombinedLoss(
            label_names=label_names, 
            device=device,
            hidden_dim=hidden_dim  # Pass the hidden dimension
        )
        
        # Setup optimizer and trainer
        optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
        print(f"Optimizer initialized with args: {train_cfg}")
        trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, device)

        def func_loss(model, batch, current_epoch=0):
            inputs, label = batch
            
            # Get all outputs from model
            logits, features, projected = model(inputs, True)
            
            # Pass all necessary components to criterion
            total_loss, loss_dict = criterion(
                logits=logits,
                features=features,
                projected=projected,
                labels=label,
                epoch=current_epoch
            )
            
            print(f"Epoch {current_epoch} - Loss Components:")
            for loss_name, loss_value in loss_dict.items():
                print(f"  {loss_name}: {loss_value:.4f}")
            
            return total_loss, loss_dict

        def func_forward(model, batch):
            inputs, label = batch
            # For inference, we only need classification output
            logits = model(inputs, False)
            return logits, label

        def func_evaluate(label, predicts):
            # Compute accuracy and F1 score
            stat = stat_acc_f1(label.cpu().numpy(), predicts.cpu().numpy())
            return stat

        # Train the model
        trainer.train(func_loss, func_forward, func_evaluate,
                     data_loader_train, data_loader_test, data_loader_vali)
        
        # Final evaluation on test set
        label_estimate_test = trainer.run(func_forward, None, data_loader_test)
        
        print("Training completed successfully")
        return label_test, label_estimate_test
        
    except Exception as e:
        print(f"Error in classify_embeddings: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    try:
        training_rate = 0.8
        label_rate = 0.1
        balance = True
        
        mode = "contrastive"
        method = "gru"
        args = handle_argv('classifier_' + mode + "_" + method, 'train.json', method)
        
        embedding, labels = load_embedding_label(args.model_file, args.dataset, args.dataset_version)
        print("Data dimensions:", embedding.shape, "Label dimensions:", labels.shape)

        # Now receives additional return values
        """label_test, label_estimate_test, best_params, best_f1 = classify_embeddings(
            args, embedding, labels, args.label_index,
            training_rate, label_rate, balance=balance, method=method
        )"""

        label_test, label_estimate_test = classify_embeddings(
            args, embedding, labels, args.label_index,
            training_rate, label_rate, balance=balance, method=method
        )

        if label_test is not None:
            label_names, label_num = load_dataset_label_names(args.dataset_cfg, args.label_index)
            acc, matrix, f1 = stat_results(label_test, label_estimate_test)
            print("calculated acc, matrix, f1")
            matrix_norm = plot_matrix(matrix, label_names)

            
            """print("\nFinal Results:")
            print(f"Best Parameters: {best_params}")
            print(f"Best F1 Score: {best_f1:.4f}")"""
        else:
            print("Error: Grid search failed to find valid parameters")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()