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
    ]"""
    descriptions = [
        "Quick single touch and release on earbud surface with consistent moderate pressure",
        "Two rapid successive taps with brief interval, maintaining uniform pressure and timing",
        "Smooth vertical sliding motion from bottom to top of earbud surface with steady contact",
        "Continuous downward sliding gesture from top to bottom with maintained finger contact",
        "Sustained finger contact on earbud surface with steady pressure for extended duration",
        "Circular finger movement on earbud surface maintaining continuous contact and even pressure",
        "Precise tap action on bottom portion of earbud with focused point of contact",
        "Targeted tap gesture on upper section of earbud with controlled impact force",
        "Gentle tapping motion on ear's upper cartilage area while wearing earbud",
        "Light tapping contact on soft earlobe tissue near earbud position",
        "Firm tap on jawline below ear, utilizing bone conduction properties"
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