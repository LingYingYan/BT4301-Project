import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import mlflow
import mlflow.pytorch


pd.set_option('display.max_columns', None)
df_X = pd.read_csv('X_train.csv')
df_y = pd.read_csv('y_train.csv')
df = pd.concat([df_X, df_y], axis=1)
df['label'] = df['label'].map({'OR': 1, 'CG': 0})

# Boolean columns change to 0/1
boolean_cols = [
    'has_html_tags', 'has_non_ascii', 'has_links', 'sentiment_match', 'has_repeated_punctuation', 'has_numbers', 'is_uppercase', 'has_weird_spacing', 
    'has_emojis'
]
for col in boolean_cols:
    df[col] = df[col].astype(int)

# Categorical columns change to one-hot encoding
df = pd.get_dummies(df, columns=['review_readability'], drop_first=True)
df = pd.get_dummies(df, columns=['sentiment_label'], drop_first=True)

# numertic columns
numeric_cols = [
    'repeated_words', 'sentiment_score', 'sentiment_discrepancy', 'rating', 'flesch_reading_ease', 'avg_word_len', 'punct_density',
     'ttr', 'avg_sentence_len'
]
scalar = StandardScaler()
df[numeric_cols] = scalar.fit_transform(df[numeric_cols])
df = df.drop(columns=['text_', 'sentiment_based_on_rating', 'category'])

# final structured feature matrix
structured_cols = boolean_cols + numeric_cols + \
    [col for col in df.columns if 'review_readability_' in col or 'sentiment_label_' in col]
structured = df[structured_cols].astype(np.float32).values


# prepare for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_data = df['text_cleaned'].astype(str).tolist()

labels = df['label'].tolist()
# encodings = tokenizer(text_data, truncation=True, padding=True, max_length=512)

#debug
# print("text_data",text_data[:5])
# print("structured",structured[:5])
# print("labels",labels[:5])

class ReviewHybridDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, structured, labels):
        self.encodings = encodings
        self.structured = structured
        self.labels = labels
        # print("encoding types: ",type(encodings['input_ids']))
        # print("encoding[0] length: ",len(self.encodings['input_ids'][0]))

    def __getitem__(self, idx):
        # print("idx",idx)
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['structured'] = torch.tensor(self.structured[idx], dtype=torch.float32)
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        # print("structured shape: ", self.structured[idx].shape)
        # print("end of item",idx)
        return item

    def __len__(self):
        return len(self.labels)
    
class HybridModel(nn.Module):
    def __init__(self, struct_feat_dim):
        super(HybridModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(768 + struct_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, struct_feats):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = bert_out.last_hidden_state[:, 0, :]  # [CLS]
        combined = torch.cat((cls_token, struct_feats), dim=1)
        return self.classifier(combined)

# Training and Validation Splitting
text_train, text_val, struct_train, struct_val, label_train, label_val = train_test_split(text_data, structured, labels, test_size=0.2, random_state=42)

train_encodings = tokenizer(text_train, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(text_val, truncation=True, padding=True, max_length=512)

train_dataset = ReviewHybridDataset(train_encodings, struct_train, label_train)
val_dataset = ReviewHybridDataset(val_encodings, struct_val, label_val)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Model Initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HybridModel(struct_feat_dim=struct_train.shape[1])
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2.0]).to(device))

num_epochs = 3


# DEBUGGING
for i, batch in enumerate(train_loader):
    print("Loaded batch", i)
    if i > 5:
        break

with mlflow.start_run():
    mlflow.log_param("model_type", "Hybrid")
    mlflow.log_param("epochs", num_epochs)
    mlflow.log_param("batch_size", 16)
    mlflow.log_param("learning_rate", 2e-5)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        print(f"\nEpoch {epoch+1}")
        loop = tqdm(train_loader, leave=True)

        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            struct_feats = batch['structured'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, struct_feats=struct_feats)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} | Training Loss: {total_loss / len(train_loader):.4f}")

# Validation Loop
        model.eval()
        preds = []
        true_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                struct_feats = batch['structured'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, struct_feats=struct_feats)
                predictions = torch.argmax(outputs, dim=1)

                preds.extend(predictions.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

    # Metrics
        print("\n🔍 Validation Metrics:")
        precision = precision_score(true_labels, preds, average=None)
        recall = recall_score(true_labels, preds, average=None)
        f1 = f1_score(true_labels, preds, average=None)

        print(f"Precision (per class): {precision}")
        print(f"Recall    (per class): {recall}")
        print(f"F1-score  (per class): {f1}")


    torch.save(model.state_dict(), "Stage1Model.pt")
    print("✅ Model saved as Stage1Model.pt")


    # Log metrics
    mlflow.log_metric("val_precision", precision)
    mlflow.log_metric("val_recall", recall)
    mlflow.log_metric("val_f1", f1)
    # Log the model
    mlflow.pytorch.log_model(model, "Stage1Model")
