from numpy import ones
import torch
from torch import nn, einsum
from utils.drop_path import DropPath
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .SPT import ShiftedPatchTokenization
import torch.nn.functional as F
from .losses import EMDLoss

# helpers
 
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
class PreNorm(nn.Module):
    def __init__(self, num_tokens, dim, fn):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), ** kwargs)
 
class FeedForward(nn.Module):
    def __init__(self, dim, num_patches, hidden_dim, dropout = 0.):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )            
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., is_LSA=False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.num_patches = num_patches
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.inner_dim = inner_dim
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(self.dim, self.inner_dim * 3, bias = False)
        init_weights(self.to_qkv)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
            
        if is_LSA:
            self.scale = nn.Parameter(self.scale*torch.ones(heads))    
            self.mask = torch.eye(self.num_patches+1, self.num_patches+1)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        else:
            self.mask = None

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        if self.mask is None:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        else:
            scale = self.scale
            dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
            dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -987654321

        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v) 
            
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
    def flops(self):
        flops = 0
        if not self.is_coord:
            flops += self.dim * self.inner_dim * 3 * (self.num_patches+1)
        else:
            flops += (self.dim+2) * self.inner_dim * 3 * self.num_patches  
            flops += self.dim * self.inner_dim * 3  


class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout = 0., stochastic_depth=0., is_LSA=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.scale = {}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(num_patches, dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout, is_LSA=is_LSA)),
                PreNorm(num_patches, dim, FeedForward(dim, num_patches, dim * mlp_dim_ratio, dropout = dropout))
            ]))            
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    
    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):       
            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x            
            self.scale[str(i)] = attn.fn.scale
        return x

class ViT(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio, channels = 3, 
                 dim_head = 16, dropout = 0., emb_dropout = 0., stochastic_depth=0.,is_LSA=False, is_SPT=False,is_MABS=False,is_PAM=False,DT=False,t=1,pamt =0.5):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width
        self.dim = dim
        self.num_classes = num_classes
        self.SPT = is_SPT
        self.LSA = is_LSA
        self.is_MABS = is_MABS
        self.is_PAM = is_PAM
        self.DT = DT
        self.t =t
        self.pamt =pamt

        if not is_SPT:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                nn.Linear(self.patch_dim, self.dim)
            )
            
        else:
            self.Rearrange= (
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width))
            self.conv_gamma = nn.Sequential(nn.Linear(self.patch_dim, self.patch_dim)
            ,nn.Sigmoid(),nn.Linear(self.patch_dim, 1))
            self.conv_beta= nn.Sequential(nn.Linear(self.patch_dim, self.patch_dim)
            ,nn.Sigmoid(),nn.Linear(self.patch_dim, 1))
            self.to_patch_embedding = nn.Linear(self.patch_dim, self.dim)
         
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))
            
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(self.dim, self.num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout, 
                                       stochastic_depth, is_LSA=is_LSA)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes)
        )
        
        self.apply(init_weights)

    def forward(self, img):
        # patch embedding

        if self.SPT:
            patch_img = self.Rearrange(img)
            gamma_shift = self.conv_gamma(patch_img)
            beta_shift = self.conv_beta(patch_img)
            x_gt = patch_img
            x_shift = patch_img*gamma_shift+beta_shift
            if self.is_MABS:
                # dist = nn.KLDivLoss(reduction='none')
                # x_shift_ = F.softmax(x_shift,dim=-1)
                # x_gt_ = F.softmax(x_gt,dim=-1)
                dist = EMDLoss()
                pam_dist = torch.abs(dist(x_shift,x_gt))
                # pam_dist = torch.abs(dist(x_shift,x_gt).sum(-1))
                a = torch.topk(input =pam_dist,k=int(pam_dist.shape[1]*self.pamt),dim=-1)
                # pam_zero = torch.zeros(pam_dist.shape).cuda()
                # pam_one = torch.ones(a[1].shape).cuda()
                # pam_zero.scatter_(1,a[1],pam_one)
                x_pam = a[1].unsqueeze(-1).repeat(1,1,x_shift.shape[-1])
                if self.LSA:
                    x_shift = x_gt+x_shift
                x_shift.scatter_(1,x_pam,torch.zeros(x_pam.shape).cuda())
            # if self.LSA:
            #     x_shift = x_gt+x_shift

            mean_shift = torch.abs(x_shift.flatten(1).mean(-1)-x_shift.flatten(0).mean(-1)).mean(-1)
            std_shift = torch.abs(x_shift.flatten(1).std(-1).log()-x_shift.flatten(0).std(-1).log()).mean(-1)
            # else:
            #     mean_shift = (x_shift.flatten(1).mean(-1)-x_shift.flatten(0).mean(-1)).mean(-1)
            #     std_shift = (x_shift.flatten(1).std(-1).log()-x_shift.flatten(0).std(-1).log()).mean(-1)
            if self.is_PAM:
                if self.t >60:
                    dist = nn.KLDivLoss(reduction='none')
                    x_shift_ = F.softmax(x_shift,dim=-1)
                    x_gt_ = F.softmax(x_gt,dim=-1)
                    pam_dist = torch.abs(dist(x_shift_.log(),x_gt_).sum(-1))
                    pam = (F.softmax(1./(pam_dist*self.pamt),dim=1)*pam_dist.shape[1]*10).round()/10
                    x_shift = x_shift*pam.unsqueeze(-1).detach()
                else:
                    pam = torch.ones_like(x_shift)
            if self.DT:
                x = self.to_patch_embedding(x_shift.detach())
            else:
                x = self.to_patch_embedding(x_shift)
        else:
            x = self.to_patch_embedding(img)

        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
      
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)      

        if self.SPT:
            return self.mlp_head(x[:, 0]),x_gt,x_shift,mean_shift,std_shift
        else:
            return self.mlp_head(x[:, 0])

