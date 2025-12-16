# -------------------------------
# 1️⃣ Imports
# -------------------------------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
import numpy as np
# -------------------------------
# 0️⃣ Load Dataset
# -------------------------------
import pandas as pd

# Make sure the CSV file path is correct
df = pd.read_csv("fake_news_dataset.csv")

# Keep necessary columns
df = df[['text', 'label']]  # You can add 'category' if needed
df.dropna(subset=['text', 'label'], inplace=True)


# -------------------------------
# 2️⃣ Prepare Dataset
# -------------------------------
class NewsDataset(Dataset):
    def _init_(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def _len_(self):
        return len(self.texts)
    
    def _getitem_(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# -------------------------------
# 3️⃣ Load Pretrained Tokenizer (BERT)
# -------------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN = 128  # maximum number of tokens per article

# -------------------------------
# 4️⃣ Encode Labels
# -------------------------------
# If labels are strings: 'fake' / 'real'
if df['label'].dtype == object:
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

# -------------------------------
# 5️⃣ Split Data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

train_dataset = NewsDataset(X_train.tolist(), y_train.tolist(), tokenizer, MAX_LEN)
test_dataset = NewsDataset(X_test.tolist(), y_test.tolist(), tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# -------------------------------
# 6️⃣ Define Model
# -------------------------------
class FakeNewsBERT(nn.Module):
    def _init_(self):
        super(FakeNewsBERT, self)._init_()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        x = self.dropout(cls_output)
        x = self.fc(x)
        return x

# -------------------------------
# 7️⃣ Train Model
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FakeNewsBERT().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

EPOCHS = 3  # Can increase to 4-5 for better results

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device).unsqueeze(1)
        
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# -------------------------------
# 8️⃣ Evaluate
# -------------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask)
        preds = torch.sigmoid(outputs).cpu().numpy()
        all_preds.extend((preds >= 0.5).astype(int))
        all_labels.extend(labels.cpu().numpy())

from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(all_labels, all_preds))
print(classification_report(all_labels, all_preds))