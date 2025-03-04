# python mobile_model.py v1 blind_user 20_120     
# -f saved/pretrain_base_blind_user_20_120/limu_v1     
# -s saved/classifier_contrastive_gru_blind_user_20_120/limu_gru_v1     
# -l 0

import torch
import numpy as np
from models import LIMUBertModel4Pretrain
from config import load_model_config, load_dataset_stats
from utils import handle_argv, get_device
from features import compute_energy, detect_nucleus

class ProjectorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(128, 256)
        self.layer2 = torch.nn.Linear(256, 128)

class ContrastiveGRU(torch.nn.Module):
    def __init__(self, input_dim=72, hidden_dim=128, num_classes=15):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)
        self.projector = ProjectorModule()
        
    def forward(self, x, training=False):
        h, _ = self.gru(x)
        h = h[:, -1, :]  # Take last timestep
        output = self.classifier(h)
        return output

class MobileIMUPredictor(torch.nn.Module):
    def __init__(self, bert_model, classifier_model):
        super().__init__()
        self.bert = bert_model
        self.classifier = classifier_model
        
    def forward(self, x, nucleus_mask, sig_axis_mask):
        x = x.to(next(self.parameters()).device)
        nucleus_mask = nucleus_mask.to(x.device)
        sig_axis_mask = sig_axis_mask.to(x.device)
        
        with torch.no_grad():
            embeddings = self.bert(x, nucleus_mask=nucleus_mask, sig_axis_mask=sig_axis_mask)
        output = self.classifier(embeddings, training=False)
        return torch.softmax(output, dim=-1)

def prepare_models_for_mobile(args, example_seq_len=20):
    # Load configurations
    bert_cfg = load_model_config('pretrain_base', 'base', args.model_version)
    classifier_cfg = load_model_config('classifier_gru', 'gru', args.model_version)
    dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
    
    if bert_cfg is None or classifier_cfg is None:
        raise ValueError("Failed to load model configurations")
    
    bert_path = f"{args.model_file}.pt"
    classifier_path = f"{args.save_model}.pt"
        
    print(f"Loading BERT model from: {bert_path}")
    print(f"Loading Classifier model from: {classifier_path}")
        
    bert_state = torch.load(bert_path, map_location='cpu', weights_only=True)
    classifier_state = torch.load(classifier_path, map_location='cpu', weights_only=True)
    
    print("\nBERT model keys:")
    for key in bert_state.keys():
        print(f"{key}: {bert_state[key].shape}")
    
    print("\nClassifier model keys:")
    for key in classifier_state.keys():
        print(f"{key}: {classifier_state[key].shape}")
    
    bert_model = LIMUBertModel4Pretrain(bert_cfg, output_embed=True)
    classifier_model = ContrastiveGRU(input_dim=72, hidden_dim=128, num_classes=15)
    
    bert_model.load_state_dict(bert_state)
    classifier_model.load_state_dict(classifier_state)
    
    print("\nLoaded models successfully!")
    
    mobile_model = MobileIMUPredictor(bert_model, classifier_model)
    mobile_model.eval()
    
    example_input = torch.randn(1, example_seq_len, 6)
    example_nucleus = torch.zeros(1, example_seq_len, dtype=torch.long)
    example_sig_axis = torch.zeros(1, example_seq_len, dtype=torch.long)
    
    print("\nCreating traced model...")
    
    traced_model = torch.jit.trace(
        mobile_model,
        (example_input, example_nucleus, example_sig_axis)
    )
    
    mobile_save_path = "mobile_imu_model.pt"
    traced_model.save(mobile_save_path)
    print(f"Mobile model saved to: {mobile_save_path}")
    
    return traced_model

def compute_mobile_masks(imu_data):
    if not isinstance(imu_data, torch.Tensor):
        imu_data = torch.tensor(imu_data, dtype=torch.float32)
    
    if len(imu_data.shape) == 2:
        imu_data = imu_data.unsqueeze(0)
    
    energy = compute_energy(imu_data)
    batch_nucleus_points = detect_nucleus(energy)
    
    seq_len = imu_data.size(1)
    nucleus_mask = torch.zeros((1, seq_len), dtype=torch.long)
    for i, points in enumerate(batch_nucleus_points):
        if len(points) == 2:
            start, end = points
            nucleus_mask[i, start:end] = 1
            
    abs_rotations = torch.abs(imu_data[:, :, 3:6])
    sig_axis = abs_rotations.mean(dim=1).argmax(dim=-1)
    sig_axis_mask = (abs_rotations.argmax(dim=-1) == sig_axis[:, None]).long()
    
    return nucleus_mask, sig_axis_mask

if __name__ == "__main__":
    args = handle_argv('pretrain_base', 'pretrain.json', 'base')
    
    try:
        traced_model = prepare_models_for_mobile(args)
        print("Model converted successfully!")
        
        example_data = torch.randn(1, 20, 6)
        nucleus_mask, sig_axis_mask = compute_mobile_masks(example_data)
        
        with torch.no_grad():
            predictions = traced_model(example_data, nucleus_mask, sig_axis_mask)
        print("Example inference completed successfully!")
        print("Predicted class:", torch.argmax(predictions).item())
        
    except Exception as e:
        print(f"Error during model conversion: {str(e)}")
        import traceback
        traceback.print_exc()