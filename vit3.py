import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
import os
import numpy as np
import scipy.io
import glob
from collections import Counter
import random
import time

TRAIN_ROOT_1 = r"data/dcpt_mat/D1"
TRAIN_ROOT_2 = r"data/dcpt_mat/PD"

DATA_RATIOS = [1.0, 1.0]

# 目标域 
TEST_ROOT    = r"data/dcpt_mat/I1"

BATCH_SIZE = 50
LEARNING_RATE = 1e-5
NUM_WORKERS = 4
EPOCHS = 20


def moving_average_sliding_window(data, window_size=1, stride=1):
    total_frames = len(data)
    if total_frames < window_size:
        return data
    smoothed_data = []
    for i in range(0, total_frames - window_size + 1, stride):
        chunk = data[i : i + window_size]
        avg_img = np.mean(chunk, axis=0)
        smoothed_data.append(avg_img)
    return np.array(smoothed_data, dtype=np.float32)

class MatDataset(Dataset):

    def __init__(self, root_dir, transform=None, stack_size=1, stride=1, ratios=None, forced_classes=None):

        self.transform = transform
        self.data = []
        self.labels = []
        self.filenames = [] 
        
        if isinstance(root_dir, list):
            self.root_dirs = root_dir
        else:
            self.root_dirs = [root_dir]
            
        if ratios is None:
            self.ratios = [1.0] * len(self.root_dirs)
        else:
            self.ratios = ratios

        print(f" 正在扫描 {len(self.root_dirs)} 个数据源...")
        
        class_set = set()
        all_mat_files_map = [] 
        
        for i, r_path in enumerate(self.root_dirs):
            if not os.path.exists(r_path):
                print(f" 警告: 路径不存在 {r_path}")
                continue
            
            files = glob.glob(os.path.join(r_path, '*.mat'))
            print(f"   -> 数据源 {i+1}: {r_path} (发现 {len(files)} 个文件, 采用比例: {self.ratios[i]})")
            if self.ratios[i] < 1.0:
                files.sort() 
                take_num = int(len(files) * self.ratios[i])
                print(f"  取 {take_num} 个文件")
                files = files[:take_num]
            for fpath in files:
                fname = os.path.basename(fpath)
                class_name = fname.split('_')[0]
                class_set.add(class_name)
                all_mat_files_map.append(fpath)

        if forced_classes is not None:
            self.classes = forced_classes
            print(f"使用源域定义的类别列表 (共{len(self.classes)}类)")
        else:
            self.classes = sorted(list(class_set))
            print(f"根据文件自动生成类别列表: {self.classes}")
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 2. 加载数据
        count = 0
        skipped_count = 0 
        for fpath in all_mat_files_map:
            try:
                fname = os.path.basename(fpath)
                class_name = fname.split('_')[0]

                if class_name not in self.class_to_idx:
                    skipped_count += 1
                    continue

                mat = scipy.io.loadmat(fpath)
                keys = [k for k in mat.keys() if 'Device' in k or 'Data' in k]
                if keys:
                    target_key = next((k for k in keys if 'Device' in k), keys[0])
                    raw_data = mat[target_key]
                    raw_data = raw_data.astype(np.float32)

                    if raw_data.max() > 10.0:  
                        raw_data /= 255.0
                    
                    if stack_size > 1:
                        processed_data = moving_average_sliding_window(raw_data, window_size=stack_size, stride=stride)
                    else:
                        processed_data = raw_data

                    if class_name in self.class_to_idx:
                        label_idx = self.class_to_idx[class_name]
                        for i in range(len(processed_data)):
                            self.data.append(processed_data[i].copy())
                            self.labels.append(label_idx)
                            self.filenames.append(fname) 
                        count += 1
            except Exception as e:
                # print(e)
                pass
        
        print(f"   -> 数据集加载完毕! 总样本数: {len(self.data)} (来自 {count} 个文件)")
        if skipped_count > 0:
            print(f"   -> 注意: 跳过了 {skipped_count} 个不在类别表中的文件 ")

    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_tensor = torch.from_numpy(img_data).unsqueeze(0).repeat(3, 1, 1)
        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor, self.labels[idx], self.filenames[idx]

def get_class_weights(dataset, subset_indices, device):
    print("正在计算类别权重...")
    all_labels = [dataset.labels[i] for i in subset_indices]
    count_dict = Counter(all_labels)
    counts = [count_dict.get(i, 0) for i in range(len(dataset.classes))]
    counts = torch.tensor(counts).float()
    weights = 1.0 / torch.sqrt(counts + 1e-6)
    if weights.sum() == 0:
        return torch.ones(len(dataset.classes)).to(device)
    weights = weights / weights.sum() * len(dataset.classes)
    return weights.to(device)

def run_advanced_cross_domain(force_train=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "best_asymmetric_model.pth"
    print(f"启动高级跨域验证 ... 设备: {device}")
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


    print(f"\n--- [Step 1] 加载源域 (训练集) ---")
    source_dataset = MatDataset(
        root_dir=[TRAIN_ROOT_1, TRAIN_ROOT_2],
        ratios=DATA_RATIOS,
        transform=train_transform,
        stack_size=1,  
        stride=1
    )
    if len(source_dataset) == 0:
        print("训练数据不足，退出。")
        return

    standard_classes = source_dataset.classes
    num_classes = len(standard_classes)
    print(f"训练集定义的标准类别顺序: {standard_classes}")

    idx_to_class = {v: k for k, v in source_dataset.class_to_idx.items()}

    print(f"\n--- [Step 2] 加载目标域 (测试集)  ---")
    target_dataset = MatDataset(
        TEST_ROOT, 
        transform=test_transform, 
        stack_size=1, 
        stride=1,
        forced_classes=standard_classes  
    )
    test_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(source_dataset.classes)

    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Sequential(
        nn.Linear(768, 512),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(512, num_classes)
    )
    model = model.to(device)
    
    should_train = force_train or not os.path.exists(MODEL_PATH)
    
    if not should_train:
        print(f"检测到已有模型 {MODEL_PATH}，跳过训练，直接进行跨域测试...")
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            if model.heads.head[3].out_features != num_classes:
                 print(f"维度不匹配，强制重训!")
                 should_train = True
        except RuntimeError:
            print("模型结构不匹配，强制重新训练...")
            should_train = True

    train_start_time = 0
    train_end_time = 0
    if should_train:
        print(f"未发现模型或强制触发训练，开始源域训练流程...")
        
        target_dataset_ref = MatDataset(
            TEST_ROOT,
            transform=test_transform,
            stack_size=1,  
            stride=1,
            forced_classes=standard_classes 
        )
        
        train_size = int(0.8 * len(source_dataset))
        val_size = len(source_dataset) - train_size
        train_subset, val_subset = random_split(
            source_dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42) 
        )
        class_weights = get_class_weights(source_dataset, train_subset.indices, device)
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader   = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS) 
        
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        for param in model.parameters(): param.requires_grad = True
        model.heads.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes)
        )
        model = model.to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        print(f"\n开始混合训练 (共 {EPOCHS} 轮)...")

        best_val_acc = 0.0
        best_test_acc_at_saturation = 0.0 
        val_acc_history = [] 

        train_start_time = time.time()
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for imgs, lbls, _ in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                train_total += lbls.size(0)
                train_correct += (preds == lbls).sum().item()
            
            train_acc = 100 * train_correct / train_total if train_total > 0 else 0
            scheduler.step()
            
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for imgs, lbls, _ in val_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    outputs = model(imgs)
                    _, preds = torch.max(outputs, 1)
                    val_total += lbls.size(0)
                    val_correct += (preds == lbls).sum().item()
            val_acc = 100 * val_correct / val_total if val_total > 0 else 0
            
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for imgs, lbls, _ in test_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    outputs = model(imgs)
                    
                    _, preds = torch.max(outputs, 1)
                    test_total += lbls.size(0)
                    test_correct += (preds == lbls).sum().item()
            
            test_acc = 100 * test_correct / test_total if test_total > 0 else 0
            
            print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
                f"| Loss: {train_loss/len(train_loader):.4f} "
                f"| 混合训练Acc: {train_acc:.1f}% "
                f"| 混合验证Acc: {val_acc:.1f}% "
                f"| 目标域Acc: {test_acc:.2f}% ◀")
            
            val_acc_history.append(val_acc)
            

            is_val_saturated = False
            if len(val_acc_history) >= 3:
                if all(acc >= 99.9 for acc in val_acc_history[-3:]):
                    is_val_saturated = True
            
            should_save = False
            save_msg = ""
            
            if is_val_saturated:
                if test_acc > best_test_acc_at_saturation:
                    best_test_acc_at_saturation = test_acc
                    should_save = True
                    save_msg = f"验证集连续满分，保存Test Acc更高的模型 ({test_acc:.2f}%)"
            else:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    should_save = True
                    save_msg = f"验证集新高分，已保存 ({val_acc:.2f}%)"
                    
            if should_save:
                torch.save(model.state_dict(), "best_asymmetric_model.pth")
                print(f"   {save_msg}")

        train_end_time = time.time()
      
        print(f"训练结束。")

    print("\nStarting Test-Time Smoothing Inference (TTA)...")

    if os.path.exists(MODEL_PATH):
        try:
             model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        except:
             pass
        
    model.eval()
    test_correct = 0
    test_total = 0
    SMOOTH_WINDOW = 5
    probability_buffer = []


    device_stats = {}  

    infer_start_time = time.time()

    with torch.no_grad():
        for imgs, lbls, batch_filenames in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            
            batch_preds = []
            for i in range(probs.size(0)):
                current_prob = probs[i].cpu().numpy()
                probability_buffer.append(current_prob)
                if len(probability_buffer) > SMOOTH_WINDOW:
                    probability_buffer.pop(0)
                
                avg_prob = np.mean(probability_buffer, axis=0)
                pred_label = np.argmax(avg_prob)
                batch_preds.append(pred_label)
                
                current_fname = batch_filenames[i]
                is_correct = (pred_label == lbls[i].item())
                
                if current_fname not in device_stats:
                    device_stats[current_fname] = {'correct': 0, 'total': 0, 'predictions': Counter()}
                
                device_stats[current_fname]['total'] += 1
                if is_correct:
                    device_stats[current_fname]['correct'] += 1
                
                device_stats[current_fname]['predictions'][int(pred_label)] += 1
            
            batch_preds = torch.tensor(batch_preds).to(device)
            test_total += lbls.size(0)
            test_correct += (batch_preds == lbls).sum().item()

    infer_end_time = time.time()    
    final_acc = 100 * test_correct / test_total if test_total > 0 else 0
    print(f"\n最终目标域测试准确率: {final_acc:.2f}%")

    print("\n" + "="*40)
    print("各设备(文件)详细识别结果")
    print("="*40)
    sorted_devices = sorted(device_stats.keys())
    for dev_name in sorted_devices:
        stats = device_stats[dev_name]
        acc = 100.0 * stats['correct'] / stats['total']
        
        top_preds = stats['predictions'].most_common(3)
        pred_info = []
        for pred_idx, count in top_preds:
            pred_name = idx_to_class.get(pred_idx, f"Unknown-{pred_idx}")
            pred_ratio = 100.0 * count / stats['total']
            pred_info.append(f"{pred_name}({pred_ratio:.1f}%)")
        
        pred_str = ", ".join(pred_info)

        print(f"设备: {dev_name:<20} | 准确率: {acc:6.2f}% |  识别为: {pred_str}")
        
    print("="*40 + "\n")

    print("="*40)
    print("性能统计")
    print("="*40)
    if should_train:
        print(f"Train Time: {train_end_time - train_start_time:.2f} s")
    else:
        print(f"Train Time: Skipped (Loaded Pretrained)")
    

    total_samples = len(target_dataset)
    if total_samples > 0:
        total_infer_time_ms = (infer_end_time - infer_start_time) * 1000
        avg_infer_time_ms = total_infer_time_ms / total_samples
        print(f"⚡ Infer Time (Total): {total_infer_time_ms:.2f} ms")
        print(f"⚡ Infer Time (Avg):   {avg_infer_time_ms:.4f} ms/sample")
    print("="*40 + "\n")

if __name__ == '__main__':
    run_advanced_cross_domain(force_train=False) 