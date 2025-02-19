import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt

class SemanticEmbedder:
    def __init__(
        self, 
        model_name='bert-base-uncased', 
        pooling_strategy='mean', 
        device=None
    ):
        # Set device
        self.device = device if device else (
            'cuda' if torch.cuda.is_available() else 
            'mps' if torch.backends.mps.is_available() else 
            'cpu'
        )
        
        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        
        # Set pooling strategy
        self.pooling_strategy = pooling_strategy
    
    def get_embeddings(self, descriptions):
        # Tokenize
        inputs = self.tokenizer(
            descriptions, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        
        # Get BERT outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
            
            # Create attention mask to ignore padding
            attention_mask = inputs['attention_mask']
            
            if self.pooling_strategy == 'cls':
                # Just CLS token
                embeddings = hidden_states[:, 0, :] #cls
            
            elif self.pooling_strategy == 'mean':
                # Mean pooling, ignoring padding
                embeddings = []
                for h, mask in zip(hidden_states, attention_mask):
                    # Select only non-padding tokens
                    token_embeds = h[mask == 1]
                    embedding = token_embeds.mean(dim=0)
                    embeddings.append(embedding)
                embeddings = torch.stack(embeddings)
            
            elif self.pooling_strategy == 'max':
                # Max pooling, ignoring padding
                embeddings = []
                for h, mask in zip(hidden_states, attention_mask):
                    # Select only non-padding tokens
                    token_embeds = h[mask == 1]
                    embedding = token_embeds.max(dim=0)[0]
                    embeddings.append(embedding)
                embeddings = torch.stack(embeddings)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def compute_similarity_matrix(self, descriptions):
        # Get embeddings
        embeddings = self.get_embeddings(descriptions)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.t())
        
        # Normalize to [0, 1]
        sim_matrix = (sim_matrix - sim_matrix.min()) / (sim_matrix.max() - sim_matrix.min())
        
        # Non-linear transformation to enhance differences
        sim_matrix = torch.pow(sim_matrix, 3)
        
        # Enhance similarities
        very_similar = (sim_matrix > 0.95).float()
        somewhat_similar = ((sim_matrix > 0.85) & (sim_matrix <= 0.95)).float()
        different = (sim_matrix <= 0.85).float()
        
        enhanced_sim_matrix = (
            very_similar * sim_matrix * 1.2 +
            somewhat_similar * sim_matrix * 0.8 +
            different * sim_matrix * 0.5
        )
        
        return enhanced_sim_matrix

# Example usage
if __name__ == "__main__":
    # Example descriptions

    """descriptions = [
            
            "a Vertical upward motion gesture with properties: primary type: directional, direction: up, complexity: simple",
            "a Vertical downward motion gesture with properties: primary type: directional, direction: down, complexity: simple",
            "a Horizontal lateral left motion gesture with properties: primary type: directional, direction: left, complexity: simple",
            "a Horizontal lateral right motion gesture with properties: primary type: directional, direction: right, complexity: simple",
            
            "a Clockwise wrist rotation gesture with properties: primary type: rotational, direction: clockwise, complexity: complex",
            "a Anticlockwise wrist rotation gesture with properties: primary type: rotational, direction: anticlockwise, complexity: complex",
            
            
            "a Sharp upward jerking gesture with properties: primary type: complex, direction: up, complexity: complex",
            "a Sharp downward jerking gesture with properties: primary type: complex, direction: down, complexity: complex",
            "a Sharp leftward jerking gesture with properties: primary type: complex, direction: left, complexity: complex",
            "a Sharp rightward jerking gesture with properties: primary type: complex, direction: right, complexity: complex",
            
           
            "a Square tracing gesture with properties: primary type: shape, direction: cyclic, complexity: complex",
            "a Circle tracing gesture with properties: primary type: shape, direction: cyclic, complexity: complex",
            "a Triangle tracing gesture with properties: primary type: shape, direction: cyclic, complexity: complex",
            "a Question mark tracing gesture with properties: primary type: complex, direction: mixed, complexity: complex",
            "a Figure eight tracing gesture with properties: primary type: shape, direction: cyclic, complexity: complex"
        ]"""
    
    descriptions = [
    # Basic tap gestures
    f"a Single tap gesture with properties: primary type: tap, direction: direct, complexity: simple",
    f"a Double tap gesture with properties: primary type: tap, direction: direct, complexity: complex",
    
    # Swipe gestures
    f"a Upward swipe gesture with properties: primary type: directional, direction: up, complexity: simple",
    f"a Downward swipe gesture with properties: primary type: directional, direction: down, complexity: simple",
    
    # Press gesture
    f"a Long press gesture with properties: primary type: stationary, direction: direct, complexity: simple",
    
    # Rotational gesture
    f"a Finger rotation gesture with properties: primary type: rotational, direction: circular, complexity: complex",
    
    # Location-specific taps
    f"a Lower earbud tap gesture with properties: primary type: tap, direction: bottom, complexity: simple",
    f"a Upper earbud tap gesture with properties: primary type: tap, direction: top, complexity: simple",
    f"a Ear top tap gesture with properties: primary type: tap, direction: cartilage, complexity: simple",
    f"a Earlobe tap gesture with properties: primary type: tap, direction: lobe, complexity: simple",
    f"a Jaw tap gesture with properties: primary type: tap, direction: jaw, complexity: simple"
]

    """descriptions = ["move forearm upwards", "move forearm down", "move forearm left", "move forearm right", "rotate wrist and then move forearm right", "rotate wrist and then move forearm left", "flick wrist and then move forearm up", "flick wrist and then move forearm down", "flick wrist andthen move forearm  left", "flick wrist and then move forearm right", "draw square in the air", "draw circle in the air", "draw triangle in the air", "draw question mark in the air", "draw infinity in the air"]
    """


    
    # Create embedder with different strategies
    strategies = ['cls', 'mean', 'max']
    
    for strategy in strategies:
        print(f"\n--- {strategy.upper()} Pooling Strategy ---")
        embedder = SemanticEmbedder(pooling_strategy=strategy)
        
        # Get embeddings
        embeddings = embedder.get_embeddings(descriptions)
        print("Embeddings shape:", embeddings.shape)
        
        # Compute similarity matrix
        sim_matrix = embedder.compute_similarity_matrix(descriptions)
        
        # plot similarity matrix as heatmap
        fig, ax = plt.subplots(figsize = (15,15))
        sim_matrix = sim_matrix.cpu().detach().numpy()
        cax = ax.matshow(sim_matrix, cmap='Blues')
        # 
        fig.colorbar(cax)
        for i in range(len(sim_matrix)):
            for j in range(len(sim_matrix)):
                ax.text(j, i, round(sim_matrix[i][j], 4), ha='center', va='center', color='black')
        plt.title(f"Similarity Matrix using {strategy.upper()} Pooling Strategy")
        plt.savefig(f"similarity_matrix_{strategy}.png")