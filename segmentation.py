"""
HybridGNet Chest X-ray Segmentation - Standalone Version
=========================================================

A completely standalone class for chest X-ray segmentation using HybridGNet model.
Segments chest X-rays into Right Lung, Left Lung, and Heart regions.

No external file dependencies - all code is self-contained in this single file.

Author: Based on HybridGNet by Nicolás Gaggion
Website: https://ngaggion.github.io/

Citation:
@article{gaggion2022TMI,
    title={Improving anatomical plausibility in medical image segmentation via hybrid graph neural networks: applications to chest x-ray analysis},
    author={Gaggion, Nicolas and Mansilla, Lucas and Mosquera, Candelaria and Milone, Diego H. and Ferrante, Enzo},
    journal={IEEE Transactions on Medical Imaging},
    year={2022},
    publisher={IEEE},
    doi={10.1109/tmi.2022.3224660}
}
"""

import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import pandas as pd
from zipfile import ZipFile
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.cheb_conv import ChebConv as ChebConvBase
from torch_geometric.nn.inits import zeros, normal
import torchvision.ops.roi_align as roi_align


# ============================================================================
# MODEL UTILITIES (from modelUtils.py)
# ============================================================================

class ChebConv(ChebConvBase):
    """Custom Chebyshev convolution with normal initialization."""
    def reset_parameters(self):
        for lin in self.lins:
            normal(lin, mean=0, std=0.1)
        normal(self.bias, mean=0, std=0.1)


class Pool(MessagePassing):
    """Pooling layer from COMA."""
    def __init__(self):
        super(Pool, self).__init__(flow='source_to_target')

    def forward(self, x, pool_mat, dtype=None):
        pool_mat = pool_mat.transpose(0, 1)
        out = self.propagate(edge_index=pool_mat._indices(), x=x, norm=pool_mat._values(), size=pool_mat.size())
        return out

    def message(self, x_j, norm):
        return norm.view(1, -1, 1) * x_j


class residualBlock(nn.Module):
    """Residual block for CNN encoder."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(residualBlock, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, track_running_stats=False))
        else:
            self.skip = None

        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = F.relu(out)

        return out


# ============================================================================
# GRAPH UTILITIES (from utils.py)
# ============================================================================

def scipy_to_torch_sparse(scp_matrix):
    """Convert scipy sparse matrix to PyTorch sparse tensor."""
    # Convert to COO format to access row and col attributes
    scp_matrix = sp.coo_matrix(scp_matrix)
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape
    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor


def mOrgan(N):
    """Create adjacency matrix for organ landmarks."""
    sub = np.zeros([N, N])
    for i in range(0, N):
        sub[i, i-1] = 1
        sub[i, (i+1) % N] = 1
    return sub


def mOrganD(N):
    """Create downsampling matrix for organ landmarks."""
    N2 = int(np.ceil(N/2))
    sub = np.zeros([N2, N])
    
    for i in range(0, N2):
        if (2*i+1) == N:
            sub[i, 2*i] = 1
        else:
            sub[i, 2*i] = 1/2
            sub[i, 2*i+1] = 1/2
    
    return sub


def mOrganU(N):
    """Create upsampling matrix for organ landmarks."""
    N2 = int(np.ceil(N/2))
    sub = np.zeros([N, N2])
    
    for i in range(0, N):
        if i % 2 == 0:
            sub[i, i//2] = 1
        else:
            sub[i, i//2] = 1/2
            sub[i, (i//2 + 1) % N2] = 1/2
    
    return sub


def genMatrixesLungsHeart():
    """Generate graph matrices for lungs and heart segmentation."""
    RLUNG = 44
    LLUNG = 50
    HEART = 26
    
    Asub1 = mOrgan(RLUNG)
    Asub2 = mOrgan(LLUNG)
    Asub3 = mOrgan(HEART)
    
    ADsub1 = mOrgan(int(np.ceil(RLUNG / 2)))
    ADsub2 = mOrgan(int(np.ceil(LLUNG / 2)))
    ADsub3 = mOrgan(int(np.ceil(HEART / 2)))
                    
    Dsub1 = mOrganD(RLUNG)
    Dsub2 = mOrganD(LLUNG)
    Dsub3 = mOrganD(HEART)
    
    Usub1 = mOrganU(RLUNG)
    Usub2 = mOrganU(LLUNG)
    Usub3 = mOrganU(HEART)
        
    p1 = RLUNG
    p2 = p1 + LLUNG
    p3 = p2 + HEART
    
    p1_ = int(np.ceil(RLUNG / 2))
    p2_ = p1_ + int(np.ceil(LLUNG / 2))
    p3_ = p2_ + int(np.ceil(HEART / 2))
    
    A = np.zeros([p3, p3])
    A[:p1, :p1] = Asub1
    A[p1:p2, p1:p2] = Asub2
    A[p2:p3, p2:p3] = Asub3
    
    AD = np.zeros([p3_, p3_])
    AD[:p1_, :p1_] = ADsub1
    AD[p1_:p2_, p1_:p2_] = ADsub2
    AD[p2_:p3_, p2_:p3_] = ADsub3
    
    D = np.zeros([p3_, p3])
    D[:p1_, :p1] = Dsub1
    D[p1_:p2_, p1:p2] = Dsub2
    D[p2_:p3_, p2:p3] = Dsub3
    
    U = np.zeros([p3, p3_])
    U[:p1, :p1_] = Usub1
    U[p1:p2, p1_:p2_] = Usub2
    U[p2:p3, p2_:p3_] = Usub3
    
    return D, U, A, AD


# ============================================================================
# HYBRID MODEL (from HybridGNet2IGSC.py)
# ============================================================================

class EncoderConv(nn.Module):
    """Convolutional encoder for the hybrid model."""
    def __init__(self, latents=64, hw=32):
        super(EncoderConv, self).__init__()
        
        self.latents = latents
        self.c = 4
        
        self.size = self.c * np.array([2, 4, 8, 16, 32], dtype=np.intc)
        
        self.maxpool = nn.MaxPool2d(2)
        
        self.dconv_down1 = residualBlock(1, self.size[0])
        self.dconv_down2 = residualBlock(self.size[0], self.size[1])
        self.dconv_down3 = residualBlock(self.size[1], self.size[2])
        self.dconv_down4 = residualBlock(self.size[2], self.size[3])
        self.dconv_down5 = residualBlock(self.size[3], self.size[4])
        self.dconv_down6 = residualBlock(self.size[4], self.size[4])
        
        self.fc_mu = nn.Linear(in_features=self.size[4]*hw*hw, out_features=self.latents)
        self.fc_logvar = nn.Linear(in_features=self.size[4]*hw*hw, out_features=self.latents)

    def forward(self, x):
        x = self.dconv_down1(x)
        x = self.maxpool(x)

        x = self.dconv_down2(x)
        x = self.maxpool(x)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        
        conv5 = self.dconv_down5(x)
        x = self.maxpool(conv5)
        
        conv6 = self.dconv_down6(x)
        
        x = conv6.view(conv6.size(0), -1)
        
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
                
        return x_mu, x_logvar, conv6, conv5


class SkipBlock(nn.Module):
    """Skip connection block with ROI alignment."""
    def __init__(self, in_filters, window):
        super(SkipBlock, self).__init__()
        
        self.window = window
        self.graphConv_pre = ChebConv(in_filters, 2, 1, bias=False)
    
    def lookup(self, pos, layer, salida=(1, 1)):
        B = pos.shape[0]
        N = pos.shape[1]
        F = layer.shape[1]
        h = layer.shape[-1]
        
        # Scale from [0,1] to [0, h]
        pos = pos * h
        
        _x1 = (self.window[0] // 2) * 1.0
        _x2 = (self.window[0] // 2 + 1) * 1.0
        _y1 = (self.window[1] // 2) * 1.0
        _y2 = (self.window[1] // 2 + 1) * 1.0
        
        boxes = []
        for batch in range(0, B):
            x1 = pos[batch, :, 0].reshape(-1, 1) - _x1
            x2 = pos[batch, :, 0].reshape(-1, 1) + _x2
            y1 = pos[batch, :, 1].reshape(-1, 1) - _y1
            y2 = pos[batch, :, 1].reshape(-1, 1) + _y2
            
            aux = torch.cat([x1, y1, x2, y2], axis=1)
            boxes.append(aux)
                    
        skip = roi_align(layer, boxes, output_size=salida, aligned=True)
        vista = skip.view([B, N, -1])

        return vista
    
    def forward(self, x, adj, conv_layer):
        pos = self.graphConv_pre(x, adj)
        skip = self.lookup(pos, conv_layer)
        
        return torch.cat((x, skip, pos), axis=2), pos


class Hybrid(nn.Module):
    """Hybrid Graph Neural Network model."""
    def __init__(self, config, downsample_matrices, upsample_matrices, adjacency_matrices):
        super(Hybrid, self).__init__()
        
        self.config = config
        # Use inputsize from config (1024), divide by 32 to get hw
        hw = config['inputsize'] // 32
        self.z = config['latents']
        self.encoder = EncoderConv(latents=self.z, hw=hw)
        
        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices
        self.kld_weight = 1e-5
                
        # Get number of nodes and filters from config
        n_nodes = config['n_nodes']
        self.filters = config['filters']
        self.K = 6
        self.window = (3, 3)
        
        # Decoder fully connected layer
        outshape = self.filters[-1] * n_nodes[-1]
        self.dec_lin = torch.nn.Linear(self.z, outshape)
                
        self.normalization2u = torch.nn.InstanceNorm1d(self.filters[1])
        self.normalization3u = torch.nn.InstanceNorm1d(self.filters[2])
        self.normalization4u = torch.nn.InstanceNorm1d(self.filters[3])
        self.normalization5u = torch.nn.InstanceNorm1d(self.filters[4])
        self.normalization6u = torch.nn.InstanceNorm1d(self.filters[5])
        
        outsize1 = self.encoder.size[4]
        outsize2 = self.encoder.size[4]
                     
        # Graph convolution layers
        self.graphConv_up6 = ChebConv(self.filters[6], self.filters[5], self.K)
        self.graphConv_up5 = ChebConv(self.filters[5], self.filters[4], self.K)
        
        self.SC_1 = SkipBlock(self.filters[4], self.window)
        
        self.graphConv_up4 = ChebConv(self.filters[4] + outsize1 + 2, self.filters[3], self.K)
        self.graphConv_up3 = ChebConv(self.filters[3], self.filters[2], self.K)
        
        self.SC_2 = SkipBlock(self.filters[2], self.window)
        
        self.graphConv_up2 = ChebConv(self.filters[2] + outsize2 + 2, self.filters[1], self.K)
        self.graphConv_up1 = ChebConv(self.filters[1], self.filters[0], 1, bias=False)
                
        self.pool = Pool()
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, x):
        self.mu, self.log_var, conv6, conv5 = self.encoder(x)

        if self.training:
            z = self.sampling(self.mu, self.log_var)
        else:
            z = self.mu
            
        x = self.dec_lin(z)
        x = F.relu(x)
        
        x = x.reshape(x.shape[0], -1, self.filters[-1])
        
        x = self.graphConv_up6(x, self.adjacency_matrices[5]._indices())
        x = self.normalization6u(x)
        x = F.relu(x)
        
        x = self.graphConv_up5(x, self.adjacency_matrices[4]._indices())
        x = self.normalization5u(x)
        x = F.relu(x)
        
        x, pos1 = self.SC_1(x, self.adjacency_matrices[3]._indices(), conv6)
        
        x = self.graphConv_up4(x, self.adjacency_matrices[3]._indices())
        x = self.normalization4u(x)
        x = F.relu(x)
        
        x = self.pool(x, self.upsample_matrices[0])
        
        x = self.graphConv_up3(x, self.adjacency_matrices[2]._indices())
        x = self.normalization3u(x)
        x = F.relu(x)
        
        x, pos2 = self.SC_2(x, self.adjacency_matrices[1]._indices(), conv5)
        
        x = self.graphConv_up2(x, self.adjacency_matrices[1]._indices())
        x = self.normalization2u(x)
        x = F.relu(x)
        
        x = self.graphConv_up1(x, self.adjacency_matrices[0]._indices())
        
        return x, pos1, pos2


# ============================================================================
# CHEST X-RAY SEGMENTER CLASS
# ============================================================================

class ChestXraySegmenter:
    """
    Chest X-ray segmentation class using HybridGNet model.
    
    Segments chest X-rays into three anatomical structures:
    - Right Lung (RL): 44 landmarks
    - Left Lung (LL): 50 landmarks  
    - Heart (H): 26 landmarks
    Total: 120 landmarks
    
    Attributes:
        device (torch.device): Computing device (CPU or CUDA)
        model (Hybrid): Loaded HybridGNet model
    """
    
    def __init__(self, weights_path='weights.pt', device=None):
        """
        Initialize the segmenter with model weights.
        
        Args:
            weights_path (str): Path to the model weights file
            device (torch.device, optional): Computing device. Auto-detects if None.
        """
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        self.model = self._load_model(weights_path)
        print("Model loaded successfully!")
    
    def _load_model(self, weights_path):
        """Load the HybridGNet model with pre-trained weights."""
        # Generate graph matrices
        D, U, A, AD = genMatrixesLungsHeart()
        
        # Get dimensions
        N1 = A.shape[0]
        N2 = AD.shape[0]
        
        # Convert to COO format first, then to PyTorch sparse tensors
        A = sp.csc_matrix(A).tocoo()
        AD = sp.csc_matrix(AD).tocoo()
        D = sp.csc_matrix(D).tocoo()
        U = sp.csc_matrix(U).tocoo()
        
        # Create copies for different levels
        D_ = [D.copy()]
        U_ = [U.copy()]
        A_ = [A.copy(), A.copy(), A.copy(), AD.copy(), AD.copy(), AD.copy()]
        
        # Convert to PyTorch sparse tensors
        A_t = [scipy_to_torch_sparse(x).to(self.device) for x in A_]
        D_t = [scipy_to_torch_sparse(x).to(self.device) for x in D_]
        U_t = [scipy_to_torch_sparse(x).to(self.device) for x in U_]
        
        # Model configuration (matching original app.py)
        config = {}
        config['n_nodes'] = [N1, N1, N1, N2, N2, N2]
        config['latents'] = 64
        config['inputsize'] = 1024
        
        f = 32
        config['filters'] = [2, f, f, f, f//2, f//2, f//2]
        config['skip_features'] = f
        
        # Initialize model
        model = Hybrid(
            config=config.copy(),
            downsample_matrices=D_t,
            upsample_matrices=U_t,
            adjacency_matrices=A_t
        )
        
        # Load weights directly (not from a dict)
        model.load_state_dict(torch.load(weights_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        return model
    
    def _pad_to_square(self, img):
        """Pad image to square shape."""
        h, w = img.shape[:2]
        
        if h > w:
            padw = (h - w)
            auxw = padw % 2
            img = np.pad(img, ((0, 0), (padw//2, padw//2 + auxw)), 'constant')
            
            padh = 0
            auxh = 0
        else:
            padh = (w - h)
            auxh = padh % 2
            img = np.pad(img, ((padh//2, padh//2 + auxh), (0, 0)), 'constant')
            
            padw = 0
            auxw = 0
        
        return img, (padh, padw, auxh, auxw)
    
    def _preprocess(self, input_img):
        """Preprocess image for model input."""
        img, padding = self._pad_to_square(input_img)
        
        h, w = img.shape[:2]
        if h != 1024 or w != 1024:
            img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        
        return img, (h, w, padding)
    
    def _remove_preprocess(self, output, info):
        """Remove preprocessing transformations from landmarks."""
        h, w, padding = info
        
        # Model outputs coordinates in range [0, 1]
        # Scale back to original image dimensions
        if h != 1024 or w != 1024:
            output = output * h
        else:
            output = output * 1024
        
        # Remove padding
        padh, padw, auxh, auxw = padding
        output[:, 0] = output[:, 0] - padw // 2
        output[:, 1] = output[:, 1] - padh // 2
        
        return output
    
    def segment(self, image_path):
        """
        Segment a chest X-ray image.
        
        Args:
            image_path (str): Path to the chest X-ray image
            
        Returns:
            dict: Dictionary containing:
                - 'landmarks': All 120 landmarks as numpy array (120, 2)
                - 'RL_landmarks': Right lung landmarks (44, 2)
                - 'LL_landmarks': Left lung landmarks (50, 2)
                - 'H_landmarks': Heart landmarks (26, 2)
                - 'image_shape': Original image shape (h, w)
        """
        # Read and preprocess image
        input_img = cv2.imread(image_path, 0)
        if input_img is None:
            raise FileNotFoundError(f"Could not read image file: {image_path}")
        input_img = input_img / 255.0
        original_shape = input_img.shape[:2]
        
        img, (h, w, padding) = self._preprocess(input_img)
        
        # Model inference
        data = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device).float()
        
        with torch.no_grad():
            output = self.model(data)[0].cpu().numpy().reshape(-1, 2)
        
        # Post-process landmarks
        output = self._remove_preprocess(output, (h, w, padding))
        output = output.astype('int')
        
        # Split landmarks by organ
        RL = output[0:44]
        LL = output[44:94]
        H = output[94:]
        
        return {
            'landmarks': output,
            'RL_landmarks': RL,
            'LL_landmarks': LL,
            'H_landmarks': H,
            'image_shape': original_shape
        }
    
    def get_masks(self, landmarks, h, w):
        """
        Generate binary masks for each organ.
        
        Args:
            landmarks (np.ndarray): All landmarks (120, 2)
            h (int): Image height
            w (int): Image width
            
        Returns:
            dict: Dictionary with 'RL_mask', 'LL_mask', 'H_mask' as numpy arrays
        """
        RL = landmarks[0:44].reshape(-1, 1, 2).astype('int')
        LL = landmarks[44:94].reshape(-1, 1, 2).astype('int')
        H = landmarks[94:].reshape(-1, 1, 2).astype('int')
        
        RL_mask = np.zeros([h, w], dtype='uint8')
        LL_mask = np.zeros([h, w], dtype='uint8')
        H_mask = np.zeros([h, w], dtype='uint8')
        
        RL_mask = cv2.drawContours(RL_mask, [RL], -1, 255, -1)
        LL_mask = cv2.drawContours(LL_mask, [LL], -1, 255, -1)
        H_mask = cv2.drawContours(H_mask, [H], -1, 255, -1)
        
        return {
            'RL_mask': RL_mask,
            'LL_mask': LL_mask,
            'H_mask': H_mask
        }
    
    def get_dense_mask(self, landmarks, h, w):
        """
        Generate a single dense segmentation mask.
        
        Args:
            landmarks (np.ndarray): All landmarks (120, 2)
            h (int): Image height
            w (int): Image width
            
        Returns:
            np.ndarray: Dense mask with values: 0=background, 1=lungs, 2=heart
        """
        RL = landmarks[0:44].reshape(-1, 1, 2).astype('int')
        LL = landmarks[44:94].reshape(-1, 1, 2).astype('int')
        H = landmarks[94:].reshape(-1, 1, 2).astype('int')
        
        img = np.zeros([h, w], dtype='uint8')
        img = cv2.drawContours(img, [RL], -1, 1, -1)
        img = cv2.drawContours(img, [LL], -1, 1, -1)
        img = cv2.drawContours(img, [H], -1, 2, -1)
        
        return img
    
    def visualize_segmentation(self, image_path, landmarks, original_shape):
        """
        Create visualization with segmentation overlay.
        
        Args:
            image_path (str): Path to original image
            landmarks (np.ndarray): All landmarks (120, 2)
            original_shape (tuple): Original image shape (h, w)
            
        Returns:
            np.ndarray: RGB image with colored segmentation overlay
        """
        # Read original image
        img = cv2.imread(image_path, 0)
        h, w = original_shape
        
        # Get dense mask
        dense_mask = self.get_dense_mask(landmarks, h, w)
        
        # Create RGB visualization
        image = np.zeros([h, w, 3])
        image[:, :, 0] = img
        image[:, :, 1] = img
        image[:, :, 2] = img
        image = image / 255.0
        
        # Color overlay: Lungs = Red, Heart = Green
        image[:, :, 0][dense_mask == 1] = 1.0  # Lungs in red
        image[:, :, 1][dense_mask == 2] = 1.0  # Heart in green
        
        return image
    
    def save_results(self, image_path, output_dir='output'):
        """
        Segment image and save CSV and landmarked image.
        
        Args:
            image_path (str): Path to input chest X-ray image
            output_dir (str): Directory to save results
            
        Returns:
            dict: Paths to saved files
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Perform segmentation
        result = self.segment(image_path)
        landmarks = result['landmarks']
        h, w = result['image_shape']
        
        # Save CSV with all landmarks
        csv_path = os.path.join(output_dir, 'landmarks.csv')
        self.save_landmarks_csv(landmarks, csv_path)
        
        # Save landmarked image visualization
        vis_image = self.visualize_segmentation(image_path, landmarks, (h, w))
        vis_path = os.path.join(output_dir, 'landmarked_image.png')
        cv2.imwrite(vis_path, (vis_image * 255).astype('uint8')[:, :, ::-1])
        
        print(f"\n✓ Segmentation complete! Results saved to: {output_dir}/")
        print(f"  - Landmarks CSV: {csv_path}")
        print(f"  - Landmarked image: {vis_path}")
        
        return {
            'landmarks_csv': csv_path,
            'landmarked_image': vis_path
        }
    
    def save_landmarks_csv(self, landmarks, csv_path):
        """
        Save all landmarks to a single CSV file with class labels.
        
        CSV Format:
        class,x,y,landmark_index
        RL,87,37,0
        RL,79,37,1
        ...
        
        Args:
            landmarks (np.ndarray): All landmarks (120, 2)
            csv_path (str): Path to save CSV file
        """
        data = []
        
        # Right Lung (44 landmarks)
        for i in range(44):
            data.append({
                'class': 'RL',
                'x': int(landmarks[i, 0]),
                'y': int(landmarks[i, 1]),
                'landmark_index': i
            })
        
        # Left Lung (50 landmarks)
        for i in range(44, 94):
            data.append({
                'class': 'LL',
                'x': int(landmarks[i, 0]),
                'y': int(landmarks[i, 1]),
                'landmark_index': i - 44
            })
        
        # Heart (26 landmarks)
        for i in range(94, 120):
            data.append({
                'class': 'H',
                'x': int(landmarks[i, 0]),
                'y': int(landmarks[i, 1]),
                'landmark_index': i - 94
            })
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        return csv_path
    
    @staticmethod
    def load_landmarks_from_csv(csv_path):
        """
        Load landmarks from CSV file.
        
        Args:
            csv_path (str): Path to CSV file
            
        Returns:
            dict: Dictionary with 'RL', 'LL', 'H' arrays and 'all' combined array
        """
        df = pd.read_csv(csv_path)
        
        rl = df[df['class'] == 'RL'][['x', 'y']].values
        ll = df[df['class'] == 'LL'][['x', 'y']].values
        h = df[df['class'] == 'H'][['x', 'y']].values
        
        all_landmarks = np.vstack([rl, ll, h])
        
        return {
            'RL': rl,
            'LL': ll,
            'H': h,
            'all': all_landmarks
        }


# ============================================================================
# EXAMPLE USAGE AND TEST CODE
# ============================================================================

def test_basic_segmentation():
    """Basic segmentation example."""
    print("=" * 70)
    print("TEST 1: Basic Segmentation")
    print("=" * 70)
    
    # Initialize segmenter
    segmenter = ChestXraySegmenter(weights_path='weights.pt')
    
    # Segment an image
    image_path = 'dataset/example1.jpeg'
    result = segmenter.segment(image_path)
    
    print(f"\n✓ Segmented: {image_path}")
    print(f"  Image shape: {result['image_shape']}")
    print(f"  Total landmarks: {result['landmarks'].shape}")
    print(f"  Right Lung landmarks: {result['RL_landmarks'].shape}")
    print(f"  Left Lung landmarks: {result['LL_landmarks'].shape}")
    print(f"  Heart landmarks: {result['H_landmarks'].shape}")
    
    return segmenter, result


def test_save_results():
    """Test saving complete results."""
    print("\n" + "=" * 70)
    print("TEST 2: Save Complete Results")
    print("=" * 70)
    
    segmenter = ChestXraySegmenter(weights_path='weights.pt')
    
    image_path = 'dataset/example1.jpeg'
    files = segmenter.save_results(image_path, output_dir='output')
    
    print("\nGenerated files:")
    for key, path in files.items():
        print(f"  {key}: {path}")


def test_csv_landmarks():
    """Test CSV landmark format."""
    print("\n" + "=" * 70)
    print("TEST 3: CSV Landmark Format")
    print("=" * 70)
    
    segmenter = ChestXraySegmenter(weights_path='weights.pt')
    
    # Segment and save
    result = segmenter.segment('dataset/example1.jpeg')
    csv_path = 'output/test_landmarks.csv'
    segmenter.save_landmarks_csv(result['landmarks'], csv_path)
    
    # Load back
    loaded = ChestXraySegmenter.load_landmarks_from_csv(csv_path)
    
    print(f"\n✓ Saved landmarks to: {csv_path}")
    print(f"\nCSV Preview:")
    df = pd.read_csv(csv_path)
    print(df.head(10))
    print(f"\n... ({len(df)} total rows)")
    
    print(f"\n✓ Loaded landmarks from CSV:")
    print(f"  Right Lung: {loaded['RL'].shape}")
    print(f"  Left Lung: {loaded['LL'].shape}")
    print(f"  Heart: {loaded['H'].shape}")


def test_batch_processing():
    """Test batch processing multiple images."""
    print("\n" + "=" * 70)
    print("TEST 4: Batch Processing")
    print("=" * 70)
    
    segmenter = ChestXraySegmenter(weights_path='weights.pt')
    
    images = [
        'dataset/example1.jpeg',
        'dataset/example2.jpeg',
        'dataset/example3.png',
        'dataset/example4.jpeg'
    ]
    
    for i, img_path in enumerate(images, 1):
        if os.path.exists(img_path):
            output_dir = f'output/batch_{i}'
            files = segmenter.save_results(img_path, output_dir=output_dir)
            print(f"\n[{i}/{len(images)}] Processed: {img_path}")
        else:
            print(f"\n[{i}/{len(images)}] Skipped (not found): {img_path}")


def test_visualization():
    """Test visualization generation."""
    print("\n" + "=" * 70)
    print("TEST 5: Visualization")
    print("=" * 70)
    
    segmenter = ChestXraySegmenter(weights_path='weights.pt')
    
    image_path = 'dataset/example1.jpeg'
    result = segmenter.segment(image_path)
    
    # Create visualization
    vis = segmenter.visualize_segmentation(
        image_path, 
        result['landmarks'],
        result['image_shape']
    )
    
    # Save visualization
    os.makedirs('output', exist_ok=True)
    cv2.imwrite('output/visualization_test.png', (vis * 255).astype('uint8')[:, :, ::-1])
    
    print(f"\n✓ Visualization saved to: output/visualization_test.png")
    print(f"  Shape: {vis.shape}")
    print(f"  Value range: [{vis.min():.2f}, {vis.max():.2f}]")


def run_all_tests():
    """Run all test functions."""
    print("\n" + "=" * 70)
    print(" CHEST X-RAY SEGMENTATION - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    try:
        test_basic_segmentation()
        test_save_results()
        test_csv_landmarks()
        test_visualization()
        test_batch_processing()
        
        print("\n" + "=" * 70)
        print(" ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # No arguments - run test suite
        run_all_tests()
        
    elif len(sys.argv) == 2 and sys.argv[1] in ['--help', '-h']:
        # Help message
        print("""
Chest X-ray Segmentation Script
================================

Usage:
    python segmentation.py                    # Run test suite
    python segmentation.py <image_path>       # Segment single image
    python segmentation.py --test <test_name> # Run specific test

Examples:
    python segmentation.py dataset/example1.jpeg
    python segmentation.py --test basic
    python segmentation.py --test batch

Available tests:
    basic     - Basic segmentation
    save      - Save complete results
    csv       - CSV landmark format
    viz       - Visualization
    batch     - Batch processing
    all       - Run all tests
        """)
        
    elif len(sys.argv) == 2 and os.path.exists(sys.argv[1]):
        # Segment single image
        image_path = sys.argv[1]
        print(f"Segmenting: {image_path}")
        
        segmenter = ChestXraySegmenter(weights_path='weights.pt')
        files = segmenter.save_results(image_path, output_dir='output')
        
        print("\n✓ Done! Check the 'output' folder for results.")
        
    elif len(sys.argv) == 3 and sys.argv[1] == '--test':
        # Run specific test
        test_name = sys.argv[2].lower()
        
        tests = {
            'basic': test_basic_segmentation,
            'save': test_save_results,
            'csv': test_csv_landmarks,
            'viz': test_visualization,
            'batch': test_batch_processing,
            'all': run_all_tests
        }
        
        if test_name in tests:
            tests[test_name]()
        else:
            print(f"Unknown test: {test_name}")
            print(f"Available tests: {', '.join(tests.keys())}")
    else:
        print("Invalid arguments. Use --help for usage information.")
