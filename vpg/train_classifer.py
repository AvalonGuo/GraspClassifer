import h5py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sentence_transformers import SentenceTransformer
from vpg.algorithmn.vilt import ViLT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "vpg/grasp_expert_data/collected_data.h5"
ENCODER_PATH = "vpg/grasp_expert_data/sbert/all-MiniLM-L6-v2"
BATCH_SIZE = 10
EPOCHS = 10
LR = 1e-4  
VAL_SPLIT = 0.2  # 验证集比例

class MultimodalDataset(Dataset):
    def __init__(self, h5_path, encoder_path):
        self.h5_path = h5_path
        
        # 加载数据
        with h5py.File(h5_path, "r") as f:
            self.imgs = f["rgb"][:]          # (N, 224, 224, 3), uint8
            texts = [t.decode("utf-8") for t in f["text"][:]]
            self.labels = f["label"][:]      # (N,)

        # 预计算所有文本嵌入（在 CPU 上）
        print("🔍 Encoding all texts with SentenceTransformer...")
        text_encoder = SentenceTransformer(encoder_path, device='cpu')
        self.text_embs = torch.stack([
            text_encoder.encode(t, convert_to_tensor=True, device='cpu')
            for t in texts
        ])  # Shape: [N, 384]
        print(f"✅ Encoded {len(self.text_embs)} text embeddings (shape: {self.text_embs.shape})")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (3, 224, 224)
        img = torch.from_numpy(img).float()

        text_emb = self.text_embs[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, text_emb, label


def main():
    dataset = MultimodalDataset(
        h5_path=DATA_PATH,
        encoder_path=ENCODER_PATH
    )

    # 划分训练集和验证集
    total_size = len(dataset)
    val_size = int(total_size * VAL_SPLIT)
    train_size = total_size - val_size

    if train_size == 0 or val_size == 0:
        raise ValueError(f"Dataset too small: {total_size} samples. Cannot split into train/val.")

    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 可复现划分
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ViLT(
        image_size=224,
        patch_size=16,
        num_classes=2,
        dim=768,
        depth=6,
        heads=12,
        mlp_dim=3072,
        lang_dim=384,
        dropout=0.1,
        emb_dropout=0.1
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    MODEL_SAVE_PATH = "vpg/grasp_expert_data/vilt/vilt_grasp_final.pth"

    print(f"🚀 Starting training on {DEVICE}")
    print(f"   Total samples: {total_size} | Train: {train_size} | Val: {val_size}")
    print(f"   Batch size: {BATCH_SIZE} | Epochs: {EPOCHS}")

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # ---------- Training ----------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for imgs, text_embs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            text_embs = text_embs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(img=imgs, lang_embed=text_embs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += preds.eq(labels).sum().item()
            train_total += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, text_embs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                text_embs = text_embs.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(img=imgs, lang_embed=text_embs)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"| Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% "
              f"| Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%")

        # 保存最佳验证准确率的模型（可选）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH.replace(".pth", "_best.pth"))
            print(f"   📦 New best model saved (Val Acc: {val_acc:.2f}%)")

    print("✅ Training completed.")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 Final model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()