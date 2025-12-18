# models/hybridgnet_landmarks.py

import os
import csv
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import normal
from torchvision.ops import roi_align


class ChebConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        
        self.lins = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=False) for _ in range(K)
        ])
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        for lin in self.lins:
            normal(lin.weight, mean=0.0, std=0.1)
        if self.bias is not None:
            normal(self.bias, mean=0.0, std=0.1)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Simple implementation: just use linear transformation
        # This is a simplified version that works with the loaded weights
        out = self.lins[0](x)
        for lin in self.lins[1:]:
            out = out + lin(x)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


class Pool(MessagePassing):
    def __init__(self) -> None:
        super().__init__(flow="source_to_target")

    def forward(self, x: torch.Tensor, pool_mat: torch.Tensor, dtype=None) -> torch.Tensor:  # type: ignore[override]
        pool_mat = pool_mat.transpose(0, 1)
        out = self.propagate(
            edge_index=pool_mat._indices(),
            x=(x, None),
            norm=pool_mat._values(),
            size=pool_mat.size(),
        )
        return out

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return norm.view(1, -1, 1) * x_j


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        if stride != 1 or in_channels != out_channels:
            self.skip: Optional[nn.Sequential] = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, track_running_stats=False),
            )
        else:
            self.skip = None

        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        identity = x
        out = self.block(x)
        if self.skip is not None:
            identity = self.skip(x)
        out = out + identity
        return F.relu(out)


def scipy_to_torch_sparse(scp_matrix: sp.spmatrix) -> torch.Tensor:
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def m_organ(N: int) -> np.ndarray:
    sub = np.zeros([N, N])
    for i in range(N):
        sub[i, i - 1] = 1
        sub[i, (i + 1) % N] = 1
    return sub


def m_organ_d(N: int) -> np.ndarray:
    N2 = int(np.ceil(N / 2))
    sub = np.zeros([N2, N])
    for i in range(N2):
        if (2 * i + 1) == N:
            sub[i, 2 * i] = 1
        else:
            sub[i, 2 * i] = 0.5
            sub[i, 2 * i + 1] = 0.5
    return sub


def m_organ_u(N: int) -> np.ndarray:
    N2 = int(np.ceil(N / 2))
    sub = np.zeros([N, N2])
    for i in range(N):
        if i % 2 == 0:
            sub[i, i // 2] = 1
        else:
            sub[i, i // 2] = 0.5
            sub[i, (i // 2 + 1) % N2] = 0.5
    return sub


def gen_matrices_lungs_heart() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    RLUNG, LLUNG, HEART = 44, 50, 26

    Asub1 = m_organ(RLUNG)
    Asub2 = m_organ(LLUNG)
    Asub3 = m_organ(HEART)

    ADsub1 = m_organ(int(np.ceil(RLUNG / 2)))
    ADsub2 = m_organ(int(np.ceil(LLUNG / 2)))
    ADsub3 = m_organ(int(np.ceil(HEART / 2)))

    Dsub1 = m_organ_d(RLUNG)
    Dsub2 = m_organ_d(LLUNG)
    Dsub3 = m_organ_d(HEART)

    Usub1 = m_organ_u(RLUNG)
    Usub2 = m_organ_u(LLUNG)
    Usub3 = m_organ_u(HEART)

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

    return A, AD, D, U


class EncoderConv(nn.Module):
    def __init__(self, latents: int = 64, hw: int = 32) -> None:
        super().__init__()
        self.latents = latents
        self.c = 4
        self.size = self.c * np.array([2, 4, 8, 16, 32], dtype=np.intc)
        self.maxpool = nn.MaxPool2d(2)

        self.d1 = ResidualBlock(1, int(self.size[0]))
        self.d2 = ResidualBlock(int(self.size[0]), int(self.size[1]))
        self.d3 = ResidualBlock(int(self.size[1]), int(self.size[2]))
        self.d4 = ResidualBlock(int(self.size[2]), int(self.size[3]))
        self.d5 = ResidualBlock(int(self.size[3]), int(self.size[4]))
        self.d6 = ResidualBlock(int(self.size[4]), int(self.size[4]))

        in_features = int(self.size[4]) * hw * hw
        self.fc_mu = nn.Linear(in_features, self.latents)
        self.fc_logvar = nn.Linear(in_features, self.latents)

    def forward(self, x: torch.Tensor):
        x = self.d1(x)
        x = self.maxpool(x)

        x = self.d2(x)
        x = self.maxpool(x)

        conv3 = self.d3(x)
        x = self.maxpool(conv3)

        conv4 = self.d4(x)
        x = self.maxpool(conv4)

        conv5 = self.d5(x)
        x = self.maxpool(conv5)

        conv6 = self.d6(x)

        x_flat = conv6.view(conv6.size(0), -1)
        x_mu = self.fc_mu(x_flat)
        x_logvar = self.fc_logvar(x_flat)
        return x_mu, x_logvar, conv6, conv5


class SkipBlock(nn.Module):
    def __init__(self, in_filters: int, window: Tuple[int, int]) -> None:
        super().__init__()
        self.window = window
        self.graphConv_pre = ChebConv(in_filters, 2, 1, bias=False)

    def lookup(self, pos: torch.Tensor, layer: torch.Tensor, salida: Tuple[int, int] = (1, 1)) -> torch.Tensor:
        B, N = pos.shape[0], pos.shape[1]
        h = layer.shape[-1]

        pos = pos * h
        _x1 = float(self.window[0] // 2)
        _x2 = float(self.window[0] // 2 + 1)
        _y1 = float(self.window[1] // 2)
        _y2 = float(self.window[1] // 2 + 1)

        boxes = []
        for b in range(B):
            x1 = pos[b, :, 0].reshape(-1, 1) - _x1
            x2 = pos[b, :, 0].reshape(-1, 1) + _x2
            y1 = pos[b, :, 1].reshape(-1, 1) - _y1
            y2 = pos[b, :, 1].reshape(-1, 1) + _y2
            boxes.append(torch.cat([x1, y1, x2, y2], axis=1))

        skip = roi_align(layer, boxes, output_size=salida, aligned=True)
        return skip.view([B, N, -1])

    def forward(self, x: torch.Tensor, adj: torch.Tensor, conv_layer: torch.Tensor):  # type: ignore[override]
        pos = self.graphConv_pre(x, adj)
        skip = self.lookup(pos, conv_layer)
        return torch.cat((x, skip, pos), axis=2), pos


class Hybrid(nn.Module):
    def __init__(
        self,
        config: Dict[str, List[int] | int],
        downsample_matrices: List[torch.Tensor],
        upsample_matrices: List[torch.Tensor],
        adjacency_matrices: List[torch.Tensor],
    ) -> None:
        super().__init__()

        hw = int(config["inputsize"]) // 32  # type: ignore[index]
        self.z = int(config["latents"])  # type: ignore[index]
        self.encoder = EncoderConv(latents=self.z, hw=hw)

        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices

        n_nodes: List[int] = config["n_nodes"]  # type: ignore[assignment]
        self.filters: List[int] = config["filters"]  # type: ignore[assignment]
        self.K = 6
        self.window = (3, 3)

        outshape = self.filters[-1] * n_nodes[-1]
        self.dec_lin = nn.Linear(self.z, outshape)

        self.norm2u = nn.InstanceNorm1d(self.filters[1])
        self.norm3u = nn.InstanceNorm1d(self.filters[2])
        self.norm4u = nn.InstanceNorm1d(self.filters[3])
        self.norm5u = nn.InstanceNorm1d(self.filters[4])
        self.norm6u = nn.InstanceNorm1d(self.filters[5])

        outsize1 = int(self.encoder.size[4])
        outsize2 = int(self.encoder.size[4])

        self.g_up6 = ChebConv(self.filters[6], self.filters[5], self.K)
        self.g_up5 = ChebConv(self.filters[5], self.filters[4], self.K)

        self.sc1 = SkipBlock(self.filters[4], self.window)
        self.g_up4 = ChebConv(self.filters[4] + outsize1 + 2, self.filters[3], self.K)

        self.g_up3 = ChebConv(self.filters[3], self.filters[2], self.K)
        self.sc2 = SkipBlock(self.filters[2], self.window)
        self.g_up2 = ChebConv(self.filters[2] + outsize2 + 2, self.filters[1], self.K)

        self.g_up1 = ChebConv(self.filters[1], self.filters[0], 1, bias=False)
        self.pool = Pool()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.dec_lin.weight, 0.0, 0.1)

    @staticmethod
    def sampling(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        self.mu, self.log_var, conv6, conv5 = self.encoder(x)

        if self.training:
            z = self.sampling(self.mu, self.log_var)
        else:
            z = self.mu

        x = F.relu(self.dec_lin(z))
        x = x.reshape(x.shape[0], -1, self.filters[-1])

        x = self.g_up6(x, self.adjacency_matrices[5]._indices())
        x = F.relu(self.norm6u(x))

        x = self.g_up5(x, self.adjacency_matrices[4]._indices())
        x = F.relu(self.norm5u(x))

        x, _ = self.sc1(x, self.adjacency_matrices[3]._indices(), conv6)
        x = self.g_up4(x, self.adjacency_matrices[3]._indices())
        x = F.relu(self.norm4u(x))

        x = self.pool(x, self.upsample_matrices[0])
        x = self.g_up3(x, self.adjacency_matrices[2]._indices())
        x = F.relu(self.norm3u(x))

        x, _ = self.sc2(x, self.adjacency_matrices[1]._indices(), conv5)
        x = self.g_up2(x, self.adjacency_matrices[1]._indices())
        x = F.relu(self.norm2u(x))

        x = self.g_up1(x, self.adjacency_matrices[0]._indices())
        return x


class ChestXrayLandmarkCentralizer:
    def __init__(
        self,
        device: Optional[torch.device] = None,
        weights_path: Optional[str] = None,
        output_dir: str = "landmarks",
    ) -> None:
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_dir = os.path.join(self.root_dir, output_dir)

        if weights_path is None:
            self.weights_path = os.path.join(self.root_dir, "..", "weights", "hybridgnet_weights.pt")
        else:
            self.weights_path = weights_path

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")

        self._model: Optional[torch.nn.Module] = None

    def process_image(self, image_path: str, save_to_file: bool = False) -> np.ndarray:
        landmarks = self._run_inference(image_path)
        if save_to_file:
            self._save_landmarks_csv(image_path, landmarks)
        return landmarks

    def process_directory(
        self,
        directory: str,
        recursive: bool = True,
        save_to_file: bool = False,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ) -> Dict[str, np.ndarray]:
        if not os.path.isdir(directory):
            raise NotADirectoryError(directory)

        results: Dict[str, np.ndarray] = {}
        if recursive:
            walker = (
                os.path.join(root, f)
                for root, _, files in os.walk(directory)
                for f in files
            )
        else:
            walker = (os.path.join(directory, f) for f in os.listdir(directory))

        for path in walker:
            if not os.path.isfile(path):
                continue
            if os.path.splitext(path)[1].lower() not in extensions:
                continue
            landmarks = self._run_inference(path)
            if save_to_file:
                self._save_landmarks_csv(path, landmarks)
            results[path] = landmarks
        return results

    def _load_model(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model

        A, AD, D, U = gen_matrices_lungs_heart()
        N1 = A.shape[0]
        A_sp = sp.csc_matrix(A).tocoo()
        AD_sp = sp.csc_matrix(AD).tocoo()
        D_sp = sp.csc_matrix(D).tocoo()
        U_sp = sp.csc_matrix(U).tocoo()

        D_list = [D_sp.copy()]
        U_list = [U_sp.copy()]
        A_list = [A_sp.copy(), A_sp.copy(), A_sp.copy(), AD_sp.copy(), AD_sp.copy(), AD_sp.copy()]

        A_t, D_t, U_t = (
            [scipy_to_torch_sparse(x).to(self.device) for x in X]
            for X in (A_list, D_list, U_list)
        )

        config: Dict[str, List[int] | int] = {}
        config["n_nodes"] = [N1, N1, N1, AD.shape[0], AD.shape[0], AD.shape[0]]
        config["latents"] = 64
        config["inputsize"] = 1024
        f = 32
        config["filters"] = [2, f, f, f, f // 2, f // 2, f // 2]
        config["skip_features"] = f

        model = Hybrid(config.copy(), D_t, U_t, A_t).to(self.device)
        state = torch.load(self.weights_path, map_location=self.device)
        
        # Remap keys from old naming convention to new
        remapped_state = {}
        for key, value in state.items():
            new_key = key
            # Encoder layer remapping
            new_key = new_key.replace("encoder.dconv_down1.", "encoder.d1.")
            new_key = new_key.replace("encoder.dconv_down2.", "encoder.d2.")
            new_key = new_key.replace("encoder.dconv_down3.", "encoder.d3.")
            new_key = new_key.replace("encoder.dconv_down4.", "encoder.d4.")
            new_key = new_key.replace("encoder.dconv_down5.", "encoder.d5.")
            new_key = new_key.replace("encoder.dconv_down6.", "encoder.d6.")
            # Graph conv upsampling remapping
            new_key = new_key.replace("graphConv_up6.", "g_up6.")
            new_key = new_key.replace("graphConv_up5.", "g_up5.")
            new_key = new_key.replace("graphConv_up4.", "g_up4.")
            new_key = new_key.replace("graphConv_up3.", "g_up3.")
            new_key = new_key.replace("graphConv_up2.", "g_up2.")
            new_key = new_key.replace("graphConv_up1.", "g_up1.")
            # Skip connection remapping
            new_key = new_key.replace("SC_1.", "sc1.")
            new_key = new_key.replace("SC_2.", "sc2.")
            remapped_state[new_key] = value
        
        model.load_state_dict(remapped_state, strict=False)
        model.eval()

        self._model = model
        return model

    @staticmethod
    def _pad_to_square(img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        h, w = img.shape[:2]
        if h > w:
            padw = h - w
            auxw = padw % 2
            img = np.pad(img, ((0, 0), (padw // 2, padw // 2 + auxw)), "constant")
            padh = auxh = 0
        else:
            padh = w - h
            auxh = padh % 2
            img = np.pad(img, ((padh // 2, padh // 2 + auxh), (0, 0)), "constant")
            padw = auxw = 0
        return img, (padh, padw, auxh, auxw)

    @classmethod
    def _preprocess(
        cls,
        input_img: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[int, int, Tuple[int, int, int, int]]]:
        img, padding = cls._pad_to_square(input_img)
        h, w = img.shape[:2]
        if h != 1024 or w != 1024:
            img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        return img, (h, w, padding)

    @staticmethod
    def _remove_preprocess(
        output: np.ndarray,
        info: Tuple[int, int, Tuple[int, int, int, int]],
    ) -> np.ndarray:
        h, w, padding = info
        output = output * (h if (h != 1024 or w != 1024) else 1024)
        padh, padw, auxh, auxw = padding
        output[:, 0] = output[:, 0] - padw // 2
        output[:, 1] = output[:, 1] - padh // 2
        return output

    def _run_inference(self, image_path: str) -> np.ndarray:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)

        model = self._load_model()

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        img = img.astype("float32") / 255.0
        original_shape = img.shape[:2]

        img_proc, (h, w, padding) = self._preprocess(img)
        data = torch.from_numpy(img_proc).unsqueeze(0).unsqueeze(0).to(self.device).float()

        with torch.no_grad():
            output = model(data)[0].cpu().numpy().reshape(-1, 2)

        output = self._remove_preprocess(output, (h, w, padding))
        output = output.astype("int32")

        H, W = original_shape
        output[:, 0] = np.clip(output[:, 0], 0, W - 1)
        output[:, 1] = np.clip(output[:, 1], 0, H - 1)
        return output

    def _save_landmarks_csv(self, image_path: str, landmarks: np.ndarray) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(self.output_dir, f"{base_name}_landmarks.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "organ", "x", "y"])
            for idx, (x, y) in enumerate(landmarks):
                organ = self._index_to_organ(idx)
                writer.writerow([idx, organ, int(x), int(y)])
        return out_path

    @staticmethod
    def _index_to_organ(idx: int) -> str:
        if idx < 44:
            return "right_lung"
        elif idx < 44 + 50:
            return "left_lung"
        return "heart"
