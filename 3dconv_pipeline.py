# Data importing, and integrity checking
import os
import pandas as pd
# for easy Path definitions
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

# for visualizing shadowy images
import matplotlib.pyplot as plt
import numpy as np
import random

# optional (I just like the summary output; install if you want with either pip or conda the package "torchsummary")
from torchsummary import summary

# Imports for confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix

# setting seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
# setting worker seed ids, also for reproducibility (in the DataLoader objects)
def worker_init_fn(worker_id):
    worker_seed = master_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

# master_seed = random.randint(0,100) # For debugging purposes, select a random seed
master_seed = 42
set_seed(master_seed)
# defining generator
g = torch.Generator()
g.manual_seed(master_seed)

# ----------------------------

# Change this string to match the profile key below
CURRENT_USER = "ALEX"  # Options: "ALEX", "NICO"
USE_SMALL_DATA = True

# Defining profiles

# 'root': The base folder for the project
# 'csv_dir': Where CSVs live (relative to root)
# 'video_dir': Where the video folders live (relative to root)
PROFILES = {
    "ALEX": {
        "root": Path("/vol/home/s4316061/CV_assignments/3_assign"),  # common path, from system root, to project folder
        "csv_dir": "",
        # Empty string means "directly in root", so like: "CV_assignments/3_assign/jester-v1-validation.csv" for me
        "video_dir": "downloaded_data_small",
        "cached_images": "downloaded_data_small",
        # I added this later, to make the training process faster (it is the folder I use to save the mean images,
        # so the script does not calculate the mean every time it wants to load it to the model)
    },
    "NICO": {
        "root": Path("data"),
        # Relative path likely works best, you used just "data/" in the shared file, so if it doesn't work, please change as needed
        "csv_dir": "labels",  # so like: "data/labels", where labels is a directory
        "video_dir": "videos",
        "cached_images": "cache"
        # change carefully: it needs to be in the same parent directory as your directory for the small dataset, and have
        # "cached_images" as the name. It HAS to match perfectly, otherwise the training will not work
    }
}


# PATH GENERATOR FUNCTION (for all the future instances where we need Paths)
def get_project_paths(user_profile, use_small=True):
    """
    Returns (train_annotation, val_annotation, video_root) based on the user profile.
    """
    if user_profile not in PROFILES:
        raise ValueError(f"Profile '{user_profile}' not found in PROFILES dictionary.")

    config = PROFILES[user_profile]
    root = config["root"]

    # defining CONSTANT Filenames (these should be the same for both of us, just locations should be different)
    if use_small:
        train_csv_name = "jester-v1-small-train.csv"
        # Matches folder name: "small-20bn-jester-v1"
        dataset_folder_name = "small-20bn-jester-v1"  # should be what the zip file unpacked
    else:
        train_csv_name = "jester-v1-train.csv"
        # Assuming full dataset name; adjust if needed
        dataset_folder_name = "20bn-jester-v1"  # this is guessing, shouldn't be used until we need the full dataset

    val_csv_name = "jester-v1-validation.csv"

    mean_cache_folder_name = "mean_cached_images"
    diff_cache_folder_name = "diff_cached_images"
    rel_diff_cache_folder_name = "rel_diff_cached_images"
    unaltered_cache_folder_name = "unaltered_cached_images"

    # Build Full Paths using '/'' operator with logic: Root / Subfolder / Filename (Pathlib magic)
    train_csv_path = root / config["csv_dir"] / train_csv_name
    val_csv_path = root / config["csv_dir"] / val_csv_name

    # point to the FOLDER containing the numbered directories
    video_root_path = root / config["video_dir"] / dataset_folder_name

    # Mean Cache Path: .../downloaded_data_small/mean_cache_images
    mean_cache_path = root / config["cached_images"] / mean_cache_folder_name
    # Diff Cache Path: .../downloaded_data_small/diff_cache_images
    diff_cache_path = root / config["cached_images"] / diff_cache_folder_name
    # Relative Diff Cache Path: .../downloaded_data_small/rel_diff_cache_images
    rel_diff_cache_path = root / config["cached_images"] / rel_diff_cache_folder_name
    # unaltered cache path: /downloaded_data_small/unaltered_cached_images
    unaltered_cache_path = root / config["cached_images"] / unaltered_cache_folder_name

    return train_csv_path, val_csv_path, video_root_path, mean_cache_path, diff_cache_path, rel_diff_cache_path, unaltered_cache_path


# -----------------------------

class Jester3DConvDataset(Dataset):
    def __init__(self, data_root, annotation_file, transform,
                 text_label_dict=None, trim_percent=0.3, cache_dir=None, 
                 num_target_frames = 20, frame_skips=1, debug=False):
        """
        Args:
            debug (bool): is used to help debug image shape problems, i.e. how many frames
                          are left after trimming and frame skipping.
            transform (Pytorch.transform.Compose): Must be present, every instantiation
                            requires a transform object that MUST include ToTensor().
            cache_dir (str): Path to a folder where .pt files will be saved.
                             If None, caching is disabled (not recommended).
        """
        self.cache_dir = cache_dir

        # Create cache directory if it's enabled and doesn't exist
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"Dataset Caching Enabled. Saving .pt files to: {self.cache_dir}")

        self.data_root = data_root
        self.transform = transform
        self.trim_percent = trim_percent  # effectively trims the images by 2 * trim_percent. This is done to
        
        self.num_target_frames = num_target_frames # Adds padding or removes frames to always have the same number of frames
        self.frame_skips = frame_skips
        self.debug = debug

        # load CSV data
        df = pd.read_csv(annotation_file, sep=';', header=None, names=['video_id', 'label'])
        self.video_ids = df['video_id'].astype(str).tolist()
        raw_labels = df['label'].tolist()

        # id_to_label_map for future lookup of predictions (so we can see what the model predicts in language. not numbers)
        self.id_to_label_map = pd.Series(df.label.values, index=df.video_id).to_dict()

        if text_label_dict is not None:
            self.class_to_idx = text_label_dict
        else:
            # creates the gesture: numeric_label map, from the gestures in train. This will be important for Validation later
            unique_labels = sorted(list(set(raw_labels)))
            self.class_to_idx = {label: i for i, label in enumerate(unique_labels)}

        self.labels = [self.class_to_idx[l] for l in raw_labels]

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        # print(f"Attempting to load {idx}")
        video_id = self.video_ids[idx]
        label = self.labels[idx]

        # CACHE PATH: if caching is on, check if we already did the work for this video: saves us A LOT of compute
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{video_id}.pt")
            if os.path.exists(cache_path):
                # FAST PATH: Load tensor directly
                video_tensor = torch.load(cache_path)
                return video_tensor, label
        
        # SLOW PATH: Calculate from scratch
        video_dir = os.path.join(self.data_root, video_id)

        try:
            frame_names = sorted([x for x in os.listdir(video_dir) if x.endswith('.jpg')])
            # debugging: seeing how many frames there are at the beginning
            if self.debug:
                print(f"Raw Frames: {len(frame_names)}")
                print(f"First: {frame_names[0]} | Last: {frame_names[-1]}")
            
        except FileNotFoundError:
            print("missed some image")
            # I'll keep this 'wrong' format, as I want the process to crash if it reaches this: it should not be able to reach this
            return torch.zeros(1), label

        # Image trimming
        # calculate how many frames to drop from each side
        total_frames = len(frame_names)
        cut_amount = int(total_frames * self.trim_percent)
        # it keeps everything if cut is 0.0 (means there aren't enough images to cut trim_percent*2 from)
        if cut_amount > 0:
            # revert to keeping only the middle frame if we cut too much (trim_percent >= 0.5)
            if (total_frames - (2 * cut_amount)) <= 0:
                mid = total_frames // 2
                frame_names = [frame_names[mid]]
            else:
                # trim it up
                frame_names = frame_names[cut_amount : -cut_amount]

        self.frames_available = len(frame_names)
        # debugging: seeing how many images are left, from how many there were (previous print)
        if self.debug:
            print(f"Trimmed Frames (Trim %: {self.trim_percent}): {len(frame_names)}")
            print(f"New First: {frame_names[0]} | New Last: {frame_names[-1]} \n")
            print(f"Will be brought down to {self.num_target_frames}")
            

        # # Difference logic
        # first_img_path = os.path.join(video_dir, frame_names[0])  # reference frame: first frame of the trimmed batch
        # ref_tensor = self.transform(Image.open(first_img_path).convert("RGB"))  # one liner to transform the PIL image 
        # # object, and close the file too (what "with open" did)
        # tensors = []
        # for i in range(0, len(frame_names), self.frame_skips):
        #     img_path = os.path.join(video_dir, frame_names[i])
        #     curr_tensor = self.transform(Image.open(img_path).convert("RGB"))
        #     # CALCULATE DIFF: |Current - Reference|
        #     # This works because we enforced that transform returns a Tensor
        #     diff_tensor = torch.abs(curr_tensor - ref_tensor)
        #     tensors.append(diff_tensor)

        
        # Relative Diff
        first_img_path = os.path.join(video_dir, frame_names[0])
        prev_tensor = self.transform(Image.open(first_img_path).convert("RGB"))
        
        tensors = []
    
        for i in range(0, len(frame_names), self.frame_skips):
            img_path = os.path.join(video_dir, frame_names[i])
            
            curr_tensor = self.transform(Image.open(img_path).convert("RGB"))
    
            # CALC DIFF: Current - Previous
            # This captures the step-by-step motion
            raw_diff = curr_tensor - prev_tensor
            
            # SIGNED NORMALIZATION: (Diff / 2) + 0.5
            # Black (0.0) = Negative change (Object left / receded)
            # Grey  (0.5) = No change (Background)
            # White (1.0) = Positive change (Object arrived / advanced)
            diff_tensor = (raw_diff / 2.0) + 0.5
            
            tensors.append(diff_tensor)
            
            prev_tensor = curr_tensor

        
        # # Basic logic: unaltered images
        # tensors = []
        # for i in range(0, len(frame_names), self.frame_skips):
        #     img_path = os.path.join(video_dir, frame_names[i])
        #     curr_tensor = self.transform(Image.open(img_path).convert("RGB"))
        #     # raw image, straight from the video folder
        #     tensors.append(curr_tensor)


        # stack all frames. so  shape (num_frames, 3, H, W)
        stacked_video = torch.stack(tensors)

        # 3D FORMATTING (pad or crop)
        # Pad or trim to a fixed number of frames (18 frames for consistency)
        num_frames = stacked_video.shape[0]
        
        if num_frames < self.num_target_frames:
            # Pad with zeros (black frames) at the end
            padding = torch.zeros(self.num_target_frames - num_frames, *stacked_video.shape[1:])
            stacked_video = torch.cat([stacked_video, padding], dim=0)
        elif num_frames > self.num_target_frames:
            # Sample evenly to fit target
            indices = torch.linspace(0, num_frames - 1, self.num_target_frames).long()
            stacked_video = stacked_video[indices]
            
        # standard format is (Channels, Depth, Height, Width) -> (C, D, H, W)
        # current is (D, C, H, W) to make a batch of shape (num_frames, channels, height, width), e.g. (16, 3, 100, 150)
        stacked_video = stacked_video.permute(1, 0, 2, 3)

        # CACHE SAVE: saving the result so we never have to calculate it again
        if self.cache_dir:
            torch.save(stacked_video, cache_path)  # saves a .pt file, which is much fater for torch to load in the CACHE PATH
        
        return stacked_video, label


# --------------------------------------

def check_data_availability(csv_path, video_root_path):
    """
    Verifies that the CSV exists and that the first 5 videos
    listed in it can be found in the video_root_path.
    """
    print(f"\n--- Sanity check ---")
    print(f"1. Checking CSV:   {csv_path}")
    print(f"2. Checking Videos: {video_root_path}")
    print(f"------------------------\n")

    # TEST 1: Check CSV Existence
    if not os.path.exists(csv_path):
        print(f"CRITICAL FAILURE: CSV file not found!")
        print(f"Path checked: {csv_path}")
        return
    else:
        print(f"--- CSV found. ---")

    # TEST 2: Check video_dir content (First 5 videos)
    try:
        # another sanity check, just for safety
        df = pd.read_csv(csv_path, sep=';', header=None, names=['video_id', 'label'])
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"CSV readable. Checking first 5 entries against video root...")

    failures = 0
    # Check the first 5 rows
    for idx, row in df.head(5).iterrows():
        video_id = str(row['video_id'])

        target_folder = os.path.join(video_root_path, video_id)

        # Check A: Folder exists?
        if not os.path.exists(target_folder):
            print(f"--- MISSING FOLDER: {target_folder} ---")
            failures += 1
            continue

            # Check B: Contains images?
        images = [x for x in os.listdir(target_folder) if x.endswith('.jpg')]
        if len(images) == 0:
            print(f"--- EMPTY FOLDER: {target_folder} exists but has 0 images. ---")
            failures += 1
        else:
            print(f"--- Found Video {video_id} ({len(images)} frames) ---")

    print(f"\n------------------------")
    if failures == 0:
        print("SUCCESS: File system structure matches the configuration.")
        print("Proceed with training.\n")
    else:
        print(f"FAILURE: {failures}/5 checks failed.")
        print("Your paths are configured, but the OS cannot find the specific folders.")
        print("Check that 'video_root' points to the folder CONTAINING the numbered directories.\n")

# -----------------------------------------

def show_video_grid(dataset, idx=None):
    """
    Visualizes a 3D video tensor as a grid of frames.
    Handles (C, D, H, W) format -> (D, H, W, C) for plotting.
    """
    # sample an image
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)
    
    # Load tensor (of shape: C, D, H, W) and Label
    video_tensor, label_idx = dataset[idx]
    
    # label from the dataset
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    label_text = idx_to_class.get(label_idx, "Unknown")
    
    print(f"Viewing Video ID: {dataset.video_ids[idx]}")
    print(f"Label: {label_text} ({label_idx})")
    print(f"Tensor Shape: {video_tensor.shape}") # expected (3, 18, H, W)

    # permuting for Matplotlib
    # Start: (C, D, H, W) -> Goal: (D, H, W, C) <=> PyTorch: (0, 1, 2, 3) -> (1, 2, 3, 0)
    video_tensor = video_tensor.permute(1, 2, 3, 0)
    
    # convert to Numpy and Clip
    images = video_tensor.numpy()
    images = np.clip(images, 0, 1) # we clip them to [0, 1] to avoid weird matplotlib artifacts.

    # make the Grid Size
    num_frames = images.shape[0]
    cols = 4
    rows = math.ceil(num_frames / cols)

    # plot
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    fig.suptitle(f"Label: {label_text}", fontsize=16)

    for i in range(rows * cols):
        ax = axes.flat[i]
        if i < num_frames:
            ax.imshow(images[i])
            ax.set_title(f"Frame {i}")
            ax.axis('off')
        else:
            # hide empty subplots if frames < grid spots
            ax.axis('off')
            
    plt.tight_layout()
    plt.show()

# -------------------------------------------------

class BasicBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels, stride, downsample, expansion_rate):
        super().__init__()
        self.expansion_rate = expansion_rate
        # Define the first convolutional layer
        self.conv1 = self.factorized_conv(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Define the second convolutional layer (no stride)
        self.conv2 = self.factorized_conv(out_channels, out_channels * self.expansion_rate)
        self.bn2 = nn.BatchNorm3d(out_channels * self.expansion_rate)

        # Skip Connection (identity mapping or 1x1x1 conv for dimensionality match)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # Store the input for the skip connection

        # --- Residual Path ---
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # --- Skip Connection ---
        if self.downsample is not None:
            identity = self.downsample(x)  # Apply 1x1x1 conv if needed

        # Add the identity (skip connection) to the residual path result
        out += identity
        out = self.relu(out)

        return out
    
    @staticmethod
    def factorized_conv(in_channels, out_planes, stride=(1, 1, 1)):
        """
        Factorized 3D convolution (Kx1x1 followed by 1xKxK)
        This is commonly used in I3D/P3D to save parameters.
        """
        spatial_conv = nn.Conv3d(
            in_channels,
            out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride[1], stride[2]),  # Only stride spatially
            padding=(0, 1, 1),
            bias=False
        )

        temporal_conv = nn.Conv3d(
            out_planes,
            out_planes,
            kernel_size=(3, 1, 1),
            stride=(stride[0], 1, 1),  # Only stride temporally
            padding=(1, 0, 0),
            bias=False
        )

        # Combine them sequentially
        return nn.Sequential(spatial_conv, temporal_conv)
class Resnet3DConvModel(nn.Module):
    def __init__(self, num_blocks, expansion_rate, img_channels=3, num_classes=27):
        super().__init__()
        self.expansion_rate = expansion_rate
        self.in_channels = 64
        # 1. Initial 3D Convolution Layer
        # Kernel size is typically large spatially (7x7) and small temporally (3 or 5)
        # Stride of (1, 2, 2) reduces H and W by half, but keeps the frame count (D)
        self.initial_conv = nn.Sequential(
            nn.Conv3d(
                img_channels, self.in_channels, kernel_size=(5, 7, 7),
                stride=(1, 2, 2), padding=(2, 3, 3), bias=False
            ),
            nn.BatchNorm3d(self.in_channels),
            nn.ReLU(inplace=True),
            # Pool to further reduce spatial size, but keep frame count (D)
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        # 2. Sequential Layers of Residual Blocks
        # The 'layers' list (e.g., [2, 2, 2, 2]) defines how many blocks are in each stage.
        self.layer1 = self._make_layer(64, num_blocks[0], stride=(1, 1, 1))
        self.layer2 = self._make_layer(128, num_blocks[1], stride=(2, 2, 2))  # Spatially and temporally downsample
        self.layer3 = self._make_layer(256, num_blocks[2], stride=(2, 2, 2))  # Spatially and temporally downsample
        self.layer4 = self._make_layer(512, num_blocks[3], stride=(2, 2, 2))  # Spatially and temporally downsample

        # 3. Final Classification Layers
        # Global Average Pooling over D, H, and W dimensions
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        final_channels = 512 * self.expansion_rate

        self.fc = nn.Linear(final_channels, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, inter_channels, num_blocks, stride):
        layers = []

        # Determine if a downsampling layer is needed
        downsample = None
        channel_requirement = inter_channels * self.expansion_rate
        if stride != (1, 1, 1) or self.in_channels != channel_requirement:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels,
                    channel_requirement,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm3d(channel_requirement)
            )

        # First block in the layer handles downsampling/channel change
        layers.append(BasicBlock3D(self.in_channels, inter_channels, stride=stride, downsample=downsample, expansion_rate=self.expansion_rate))
        self.in_channels = inter_channels * self.expansion_rate

        # Remaining blocks maintain dimension
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock3D(self.in_channels, inter_channels, stride=(1, 1, 1), downsample=None, expansion_rate=self.expansion_rate))

        return nn.Sequential(*layers)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (N, C, D, H, W) -> (Batch, 3, Frames, Height, Width)

        x = self.initial_conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)  # Flatten (N, C, 1, 1, 1) to (N, C)

        x = self.fc(x)

        return x

# ------------------------------------

def calculate_test_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    # pre-allocate tensors on GPU to avoid repeated transfers
    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            # data passed to gpu every batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(imgs)
            _, preds = torch.max(outputs, dim=1)

            # calculating accuracy on GPU
            correct += (preds == labels).sum()
            total += labels.size(0)

    # moving final scalar to CPU
    accuracy = (correct.float() / total) * 100
    return accuracy.item()


def train_model(model, train_loader, test_loader, device, save_path, num_epochs=20, lr=0.001):
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses = []
    val_accuracies = []

    best_accuracy = 0.0

    # use torch.amp for mixed precision training
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        # training loop with progress bar
        for videos, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            # move to gpu once per batch
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # zero gradients
            optimizer.zero_grad(set_to_none=True)  # More efficient than zeroing

            # mixed precision forward pass
            if scaler:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(videos)
                    loss = loss_fn(outputs, labels)

                # Mixed precision backward pass and optimization
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                # standard precision, just in case
                outputs = model(videos)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * videos.size(0)

        scheduler.step()
        # get the epoch numbers
        avg_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # evaluate on the validation set
        val_accuracy = calculate_test_accuracy(model, test_loader, device)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch:02d}: Train loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # model checkpoint addition
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), # extra data, could be taken out in cases (would require same optimizer to work)
                'scheduler_state_dict': scheduler.state_dict(), # extra data, could be taken out in cases (would require same scheduler to work)
                'accuracy': val_accuracy,
                'loss': avg_train_loss
            }, save_path)
            
            print(f'New best model with Accuracy: {val_accuracy:.2f}% saved to {save_path}')
            
    return train_losses, val_accuracies

# -----------------------------------------------

def load_model(checkpoint_path, model):
    print(f"\nLoading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # and we don't exactly need these for confusion matrices
    # scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    # if scaler: scaler.load_state_dict(checkpoint["scaler"])  # I don't have a scheduler, but my partner does

# ------------------------------------------------

def generate_confusion_matrix(model, loader, device, idx_to_class, model_type, load_from=None):
    if load_from:  # If load from is not None, load the model from a checkpoint in order to keep training from there
        load_model(load_from, model)

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    # pre-allocate tensors on GPU to avoid repeated transfers
    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            # data passed to gpu every batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(imgs)
            _, preds = torch.max(outputs, dim=1)

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    cm = confusion_matrix(y_true, y_pred)  # from scikit-learn
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    num_classes = len(idx_to_class)
    class_names = [idx_to_class[i] for i in range(num_classes)]

    cm_df = pd.DataFrame(
        cm_norm,
        index=class_names,
        columns=class_names
    )

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_df,
        cmap="Blues",
        annot=False,  # set True if you want numbers (can get cluttered)
        fmt=".2f",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(f"Confusion matrix for 3D conv model using {model_type}", fontsize=18, pad=20)
    plt.xlabel("Predicted label", fontsize=14)
    plt.ylabel("True label", fontsize=14)

    # Rotated x-ticks and increased tick label size
    plt.xticks(rotation=45, ha='right', fontsize=10) 
    plt.yticks(fontsize=10)
    
    plt.tight_layout()

    filename = f"confusion_matrix_{model_type.replace(' ', '_').lower()}.png"
    plt.savefig(filename)
    print(f"Confusion matrix saved to {filename}")

    plt.show()
    plt.close()

# -----------------------------------------------

if __name__ == "__main__":
    # to be ran once for the whole project
    train_annotation, val_annotation, video_root, mean_cache_root, diff_cache_root, rel_diff_cache_root, unaltered_cache_root = get_project_paths(
        CURRENT_USER, USE_SMALL_DATA)

    # DEBUG CHECK
    print(f"\nUser: {CURRENT_USER}")
    print(f"Train CSV:  {train_annotation}")
    print(f"Video Root: {video_root}")
    print(f"Mean cache Root: {mean_cache_root}")
    print(f"Diff cache Root: {diff_cache_root}")
    print(
        f"Exists?     {train_annotation.exists()} (CSV), {video_root.exists()} (Video Dir), {rel_diff_cache_root.exists()} (rel_diff_cache_root Dir)\n")
    # .exists is a Pathlib method
    
    train_annotation, val_annotation, video_root, mean_cache_root, diff_cache_root, rel_diff_cache_root, unaltered_cache_root = str(
        train_annotation), str(val_annotation), str(video_root), str(mean_cache_root), str(diff_cache_root), str(
        rel_diff_cache_root), str(unaltered_cache_root)

    # TRAINING SPECIFICS
    trim_percent = 0.2  # found empirically to yield the best outputs (clearest shadows and images)
    num_target_frames = 18
    frame_skips = 1

    # list of models I'd say we should have
    best_models = {
        "best_2d_mean_pixel_images.ckpt": {"done": True, "saved": False, "final_acc_%": 44},  # accuracy is old, has not been retrained
        "best_2d_difference_images.ckpt": {"done": True, "saved": False, "final_acc_%": 0},  # unknown_by_Alex
        "best_3d_unaltered_images.ckpt": {"done": True, "saved": True, "final_acc_%": 80.96}, 
        "best_3d_diff_images.ckpt": {"done": True, "saved": True, "final_acc_%": 83.9},
        "best_3d_rel_diff_images.ckpt": {"done": False, "saved": False, "final_acc_%": 0}
    }

    # variable to easily switch between caches and other (model, path) specific variables (Nico's idea, and a great one!)
    cache_type = "unaltered"  
    match cache_type:
        case "diff":
            cache_dir = diff_cache_root
            save_path = "best_3d_diff_images.ckpt"
        case "rel_diff":
            cache_dir = rel_diff_cache_root  # for me it's None, for now
            save_path = "best_3d_rel_diff_images.ckpt"
        case "unaltered":
            cache_dir = unaltered_cache_root
            save_path = "best_3d_unaltered_images.ckpt"

    transform = transforms.Compose([
        transforms.Resize((100, 150)),
        transforms.ToTensor()
    ])

    train_3d_data = Jester3DConvDataset(
        data_root=video_root,
        annotation_file=train_annotation,
        transform=transform,
        trim_percent=trim_percent,
        num_target_frames=num_target_frames,
        frame_skips=frame_skips,
        cache_dir=cache_dir,
        debug=False
    )

    # label map learned (generated) from the train videos: e.g. "Stop sign" is 1, and so on
    label_map = train_3d_data.class_to_idx

    valid_3d_data = Jester3DConvDataset(
        data_root=video_root,
        annotation_file=val_annotation,
        transform=transform,
        text_label_dict=label_map,
        # so the Validation loader does not generate new ones and turn everything on its head
        trim_percent=trim_percent,
        num_target_frames=num_target_frames,
        frame_skips=frame_skips,
        cache_dir=cache_dir,
        debug=False
    )
    
    # CHECKS AND DEBUG MESSAGES
    
    # if train_annotation works, then val_annotation works too. This has to return SUCCESS, otherwise the class cannot access the data locations
    check_data_availability(train_annotation, video_root)

    # This line is for debugging, to check if the video frames make sense
    # show_video_grid(train_3d_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Resnet3DConvModel([2,2,2,2], 2, img_channels=3, num_classes=27).to(device) # [3,4,6,3] is the number of Res-blocks of ResNet50

    summary(model, input_size=(3, num_target_frames, 100, 150))

    batch_size = 64
    if device == "cpu":
        num_workers = min(12, os.cpu_count() or 2)  # dynamic core loading; swap the hard limit (12) depending on the amount of ram available (<16)
        persistent = False # standard logic for cpu
    else:
        num_workers = 12
        persistent = True
    # some computers can handle 12 core usage, but (with the assumption that we're calculating for video processing) we might run into OOM
    # "Out of Memory" errors on the RAM side, not the VRAM side. Note that this is foe Data Loading only! inspect machine_limit.py file for more info
    print(f"\nDevice available: {device}, with num_workers={num_workers} and persistent memory={persistent}")
    epochs = 30

    # DATA LOADERS AND TRAINING FUNCTION
    train_loader = DataLoader(
        train_3d_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=g,
        persistent_workers=persistent, # keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None,  # forcing workers to buffer 2 batches ahead (workers gotta work harder)
        pin_memory=True
    )

    val_loader = DataLoader(
        valid_3d_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent, 
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True
    )

    print("\nDiagnostics complete.")
    # input("Press Enter to start training (or Ctrl+C to stop)...")
    # train_losses, test_accuracies = train_model(
    #     model=model,
    #     train_loader=train_loader,
    #     test_loader=val_loader,
    #     device=device,
    #     save_path=save_path,
    #     num_epochs=epochs,
    #     lr=0.001
    # )

    # print(f"Finished with \nTrain_losses: {train_losses} \nTest_accuracies: {test_accuracies}")

    print("Confusion matrix generation commencing. Attributes:")
    print(f"Cache path: {cache_dir}")
    print(f"Loading Model from: {save_path}")
    print(f"idx_to_class map available: {True if label_map else False}")
    input("Press Enter to start training (or Ctrl+C to stop)...")

    idx_to_class = {v: k for k, v in valid_3d_data.class_to_idx.items()}
    generate_confusion_matrix(model, val_loader, device, idx_to_class, "Unmodified frames",
                              load_from=save_path)
