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

mlflow.set_tracking_uri("http://localhost:5000")

df_test_x = pd.read_csv('X_test.csv')
df_test_y = pd.read_csv('y_test.csv')
df_test = pd.concat([df_test_x, df_test_y], axis=1)
df_test['label'] = df_test['label'].map({'OR': 1, 'CG': 0})

# Boolean columns change to 0/1
boolean_cols = [
    'has_html_tags', 'has_non_ascii', 'has_links', 'sentiment_match', 'has_repeated_punctuation', 'has_numbers', 'is_uppercase', 'has_weird_spacing', 
    'has_emojis'
]
for col in boolean_cols:
    df_test[col] = df_test[col].astype(int)

# Categorical columns change to one-hot encoding
df_test = pd.get_dummies(df_test, columns=['review_readability'], drop_first=True)
df_test = pd.get_dummies(df_test, columns=['sentiment_label'], drop_first=True)

# numertic columns
numeric_cols = [
    'repeated_words', 'sentiment_score', 'sentiment_discrepancy', 'rating', 'flesch_reading_ease', 'avg_word_len', 'punct_density',
     'ttr', 'avg_sentence_len'
]
scalar = StandardScaler()
df_test[numeric_cols] = scalar.fit_transform(df_test[numeric_cols])
df_test = df_test.drop(columns=['text_', 'sentiment_based_on_rating', 'category'])

# final structured feature matrix
structured_cols = boolean_cols + numeric_cols + \
    [col for col in df_test.columns if 'review_readability_' in col or 'sentiment_label_' in col]
structured = df_test[structured_cols].astype(np.float32).values

# prepare for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_data = df_test['text_cleaned'].astype(str).tolist()
encodings = tokenizer(text_data, truncation=True, padding=True, max_length=512)

labels = df_test['label'].tolist()


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
            nn.Linear(128, 2)  # Output: fake/genuine
        )

    def forward(self, input_ids, attention_mask, struct_feats):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = bert_out.last_hidden_state[:, 0, :]  # [CLS]
        combined = torch.cat((cls_token, struct_feats), dim=1)
        return self.classifier(combined)


test_dataset = ReviewHybridDataset(encodings, structured, labels)
test_loader = DataLoader(test_dataset, batch_size=16)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridModel(struct_feat_dim=structured.shape[1])
model.load_state_dict(torch.load("Stage1Model.pt"))
model.eval()
model.to(device)

# Run inference
predictions = []
true_labels = []

with mlflow.start_run(run_name="Testing"):
    with torch.no_grad():
        test_loop = tqdm(test_loader, desc="Testing", leave=True)
        for batch in test_loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            struct_feats = batch['structured'].to(device)
            labels_batch = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, struct_feats=struct_feats)
            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels_batch.cpu().numpy())
        
        print("length of preds: ",len(predictions))
        print("length of true_labels: ",len(true_labels))
    # ✅ Log metrics to MLflow
    precision = precision_score(true_labels, predictions, average=None)
    recall = recall_score(true_labels, predictions, average=None)
    f1 = f1_score(true_labels, predictions, average=None)

    print("\n📊 Test Metrics:")
    print(f"Precision: {precision}")
    print(f"Recall   : {recall}")
    print(f"F1-score : {f1}")

    mlflow.log_metric("test_precision_Fake", float(precision[0]))
    mlflow.log_metric("test_precision_Genuine", float(precision[1]))
    mlflow.log_metric("test_recall_Fake", float(recall[0]))
    mlflow.log_metric("test_recall_Genuine", float(recall[1]))
    mlflow.log_metric("test_f1_Fake", float(f1[0]))
    mlflow.log_metric("test_f1_Genuine", float(f1[1]))

# predict and output to csv if needed
# df_test['predicted_label'] = predictions
# df_test.to_csv("test_results.csv", index=False)