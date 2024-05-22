import torch
from torch import nn, optim
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import os

# Dummy data
file_path = "maywell/ko_wikidata_QA"
cache_dir = "/data/llm/"
cache_dir = "D:\\huggingface\\cache"

class ContrastiveDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=128):
        self.anchor = []
        self.positive = []
        self.negative = []

        for i, item in enumerate(data):
            if i + 1 == len(data):
                anchor_sen = item['instruction']
                positive_sen = item['output']
                negative_sen = data[0]['output']
            else:
                anchor_sen = item['instruction']
                positive_sen = item['output']
                negative_sen = data[i + 1]['output']

            anchor = tokenizer(anchor_sen, truncation=True, return_tensors="pt", max_length=max_length,
                               padding='max_length')
            positive = tokenizer(positive_sen, truncation=True, return_tensors="pt", max_length=max_length,
                                 padding='max_length')
            negative = tokenizer(negative_sen, truncation=True, return_tensors="pt", max_length=max_length,
                                 padding='max_length')

            self.anchor.append(anchor)
            self.positive.append(positive)
            self.negative.append(negative)

    def __getitem__(self, idx):
        return {
            'anchor_input_ids': self.anchor[idx]['input_ids'][0],
            'anchor_attention_mask': self.anchor[idx]['attention_mask'][0],
            'positive_input_ids': self.positive[idx]['input_ids'][0],
            'positive_attention_mask': self.positive[idx]['attention_mask'][0],
            'negative_input_ids': self.negative[idx]['input_ids'][0],
            'negative_attention_mask': self.negative[idx]['attention_mask'][0],
        }

    def __len__(self):
        return len(self.anchor)


class RobertaContrastive(nn.Module):
    def __init__(self, model_name='roberta-base', device='cuda'):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
        self.device = device

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0]


def contrastive_loss(a, p, n, temperature=0.05):
    cos = nn.CosineSimilarity(dim=1)
    cross_entropy_loss = nn.CrossEntropyLoss()

    positive_similarity = cos(a, p) / temperature
    negative_similarity = cos(a, n) / temperature
    cosine_similarity = torch.cat([positive_similarity, negative_similarity], dim=1).to(device)

    labels = torch.arange(cosine_similarity.size(0)).long().to(device)

    loss = cross_entropy_loss(cosine_similarity, labels)

    return loss


# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize tokenizer, model, and dataset
tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta-multitask',  cache_dir=cache_dir)
model = RobertaContrastive('BM-K/KoSimCSE-roberta-multitask', device=device)


raw_datasets = load_dataset(file_path, cache_dir=cache_dir)

train_testsplit = raw_datasets['train'].train_test_split(test_size=0.1, seed=42)
fix_datasets = DatasetDict({
    # 'train': train_testsplit['train'].select(range(100)),
    'train': train_testsplit['train'],
    'validation': train_testsplit['test'],
    # 'validation': train_testsplit['test'].select(range(10))
})

train_dataset = ContrastiveDataset(tokenizer, fix_datasets['train'])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataset = ContrastiveDataset(tokenizer, fix_datasets['validation'])
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=1e-5)  # Adjust learning rate as needed
num_training_steps = len(train_loader) * 3  # Assuming 3 epochs of training
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Training loop
for epoch in range(3):  # Number of epochs
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        anchor_embeddings = model(batch['anchor_input_ids'], batch['anchor_attention_mask'])
        positive_embeddings = model(batch['positive_input_ids'], batch['positive_attention_mask'])
        negative_embeddings = model(batch['negative_input_ids'], batch['negative_attention_mask'])

        loss = contrastive_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    print(f"Epoch {epoch + 1} - Training Loss: {train_loss / len(train_loader):.4f}")

    # 저장할 디렉토리 지정
    save_directory = 'path_to_save'  # 여기를 적절한 경로로 변경하세요.
    checkpoint_path = os.path.join(save_directory, f'roberta_contrastive_checkpoint_epoch_{epoch + 1}.pth')

    # 디렉토리가 존재하지 않는 경우 생성
    if not os.path.exists(save_directory):
        os.makedirs(save_directory, exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
        'scheduler_state_dict': lr_scheduler.state_dict()
    }, checkpoint_path)


    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            anchor_embeddings = model(batch['anchor_input_ids'], batch['anchor_attention_mask'])
            positive_embeddings = model(batch['positive_input_ids'], batch['positive_attention_mask'])
            negative_embeddings = model(batch['negative_input_ids'], batch['negative_attention_mask'])
            loss = contrastive_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            val_loss += loss.item()
    print(f"Epoch {epoch + 1} - Validation Loss: {val_loss / len(val_loader):.4f}")