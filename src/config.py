import torch

class Config:
    RAW_DATA_DIR = "data/raw/Kvasir-SEG"
    IMAGE_DIR = "data/raw/Kvasir-SEG/images"
    MASK_DIR = "data/raw/Kvasir-SEG/masks"
    CHECKPOINT_DIR = "output/checkpoints"
    SPLIT_JSON = "data/splits/dataset_split.json"
    
    MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
    IMG_SIZE = 512
    NUM_CLASSES = 2  # background + polyp (nhãn mask 0/1)
    
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    OLLAMA_MODEL = "llama3" 
    OLLAMA_API_URL = "http://localhost:11434"