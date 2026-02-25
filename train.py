import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    cohen_kappa_score,
    roc_curve
)

from preprocessing_pipeline import load_bonn_csv
from bonn_dataset import BonnDataset
from model import MavenNet


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    print("Loading dataset...")
    data, labels = load_bonn_csv("Epileptic Seizure Recognition.csv")

    print("Loading CWT representation...")
    cwt_data = np.load("cwt_data.npy")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    epochs = 40
    batch_size = 256
    lr = 0.001
    patience = 7  # Early stopping patience

    all_accuracies = []
    all_sensitivities = []
    all_specificities = []
    all_precisions = []
    all_f1s = []
    all_aucs = []
    all_kappas = []

    all_preds_global = []
    all_labels_global = []

    fold_number = 1

    for train_index, test_index in skf.split(cwt_data, labels):

        print(f"\n========== Fold {fold_number} ==========")

        X_train_full, X_test = cwt_data[train_index], cwt_data[test_index]
        y_train_full, y_test = labels[train_index], labels[test_index]

        # Validation split (10% of training fold)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.1,
            stratify=y_train_full,
            random_state=42
        )

        train_dataset = BonnDataset(X_train, y_train)
        val_dataset = BonnDataset(X_val, y_val)
        test_dataset = BonnDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        model = MavenNet().to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=3,
            factor=0.5,
            verbose=True
        )

        # Proper class weights
        class_counts = np.bincount(y_train)
        class_weights = torch.tensor(
            [len(y_train) / class_counts[0],
             len(y_train) / class_counts[1]],
            dtype=torch.float32
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_val_loss = float("inf")
        early_stop_counter = 0

        # -------------------------
        # Training
        # -------------------------
        for epoch in range(epochs):

            model.train()
            train_loss = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    val_loss += loss.item()

            scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{epochs} "
                  f"Train Loss: {train_loss:.4f} "
                  f"Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"best_model_fold_{fold_number}.pth")
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

        # Load best model
        model.load_state_dict(torch.load(f"best_model_fold_{fold_number}.pth"))

        # -------------------------
        # Evaluation
        # -------------------------
        model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                outputs = model(x)
                probs = torch.softmax(outputs, dim=1)[:, 1]

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.numpy())

        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        preds = (np.array(all_probs) > optimal_threshold).astype(int)

        all_preds_global.extend(preds)
        all_labels_global.extend(all_labels)

        accuracy = accuracy_score(all_labels, preds)
        recall = recall_score(all_labels, preds)
        precision = precision_score(all_labels, preds)
        f1 = f1_score(all_labels, preds)

        tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
        specificity = tn / (tn + fp)

        auc = roc_auc_score(all_labels, all_probs)
        kappa = cohen_kappa_score(all_labels, preds)

        print(f"\nFold {fold_number} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Sensitivity: {recall:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Kappa: {kappa:.4f}")

        all_accuracies.append(accuracy)
        all_sensitivities.append(recall)
        all_specificities.append(specificity)
        all_precisions.append(precision)
        all_f1s.append(f1)
        all_aucs.append(auc)
        all_kappas.append(kappa)

        fold_number += 1

    # -------------------------
    # Final Report
    # -------------------------
    print("\n========== 5-Fold Cross-Validation Results ==========")

    def report(metric_list, name):
        print(f"{name}: {np.mean(metric_list):.4f} ± {np.std(metric_list):.4f}")

    report(all_accuracies, "Accuracy")
    report(all_sensitivities, "Sensitivity")
    report(all_specificities, "Specificity")
    report(all_precisions, "Precision")
    report(all_f1s, "F1-score")
    report(all_aucs, "AUC")
    report(all_kappas, "Kappa")

    cm = confusion_matrix(all_labels_global, all_preds_global)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Seizure"],
                yticklabels=["Normal", "Seizure"])

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (5-Fold CV)")
    plt.tight_layout()
    plt.show()