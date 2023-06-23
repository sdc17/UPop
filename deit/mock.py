from torchvision.datasets.folder import (
    Any, Callable, cast, Dict, List, Optional, Tuple, VisionDataset, default_loader, IMG_EXTENSIONS, 
    DatasetFolder, VisionDataset, io, os, Image, make_dataset, find_classes)


#############################################################################################################
####################### Mocking torchvision.datasets.folder in torchvision 0.12.0 ###########################
#############################################################################################################

class DatasetFolder(VisionDataset):
    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        client: Optional[Any] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.client=client
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return find_classes(directory)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        if self.client is not None:
            with io.BytesIO(self.client.get(os.path.join('s3://sdcBucket', 'datasets/vision/imagenet_prep', '/'.join(path.split('/')[-3:])))) as f:
                sample = Image.open(f).convert('RGB')   ######
        else:
            sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

import torchvision.datasets.folder
torchvision.datasets.folder.DatasetFolder = DatasetFolder

class ImageFolder(DatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        client: Optional[Any] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            client=client,
        )
        self.imgs = self.samples

import torchvision.datasets
torchvision.datasets.ImageFolder = ImageFolder


#############################################################################################################
############################# Mocking timm.models.layers.mlp in timm 0.4.12 #################################
#############################################################################################################


from timm.models.layers import Mlp, nn


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 search=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        if search:
            self.alpha = nn.Parameter(torch.ones(1, 1, hidden_features))

    def forward(self, x):
        x = self.fc1(x)

        if hasattr(self, 'alpha'):
            x = x * self.alpha

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

import timm.models.layers
timm.models.layers.Mlp = Mlp


#############################################################################################################
########################## Mocking timm.models.vision_transformer in timm 0.4.12 ############################
#############################################################################################################


from timm.models.vision_transformer import VisionTransformer,partial, nn, PatchEmbed, torch, Block, OrderedDict, DropPath


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 search=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if search:
            self.alpha = nn.Parameter(torch.ones(1, 1, 1, 1, head_dim))

    def forward(self, x):
        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
 
        if hasattr(self, 'alpha'):
            qkv = qkv * self.alpha

        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 search=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                              search=search)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       search=search)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


import timm.models.vision_transformer
timm.models.vision_transformer.Attention = Attention
timm.models.vision_transformer.Block = Block



def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', 
                 search=False):
    super(VisionTransformer, self).__init__()
    self.layers = depth ###
    self.num_classes = num_classes
    self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
    self.num_tokens = 2 if distilled else 1
    norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
    act_layer = act_layer or nn.GELU

    self.patch_embed = embed_layer(
        img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
    num_patches = self.patch_embed.num_patches

    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
    self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
    self.pos_drop = nn.Dropout(p=drop_rate)

    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
    self.blocks = nn.Sequential(*[
        Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
            search=search)
        for i in range(depth)])
    self.norm = norm_layer(embed_dim)

    # Representation layer
    if representation_size and not distilled:
        self.num_features = representation_size
        self.pre_logits = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(embed_dim, representation_size)),
            ('act', nn.Tanh())
        ]))
    else:
        self.pre_logits = nn.Identity()

    # Classifier head(s)
    self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
    self.head_dist = None
    if distilled:
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    self.init_weights(weight_init)

VisionTransformer.__init__ = __init__