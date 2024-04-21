import torch
import torch.nn as nn
from .DWT_IDWT.DWT_IDWT_layer import *


# --------- basic layer --------- # 
class Downsample_dwt(nn.Module):
    def __init__(self, wavename = 'haar'):  
        super(Downsample_dwt, self).__init__()
        self.dwt = DWT_2D(wavename = wavename)
        # wavename check
        # In WaveCNet codes : 

    def forward(self, input):
        ll, lh, hl, hh = self.dwt(input)
        subband = []
        subband.append(ll)
        subband.append(lh)
        subband.append(hl)
        subband.append(hh)
        return subband
        # return ll, lh, hl, hh

'''
DWT_2D : input : 2D data
         output : LL, LH, HL, HH
DWT_2D_tiny : input : 2D data
              output : LL (Low frequency)
IDWT_2D : input : LL, LH, HL, HH
          output : 2D data
'''

def bili_resize(factor):
    return nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=False)

class denselayer(nn.Module):
    def __init__(self, in_size, out_size):
        super(denselayer, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU()
    def forward(self, x):
        return torch.cat([x, self.act(self.conv(x))], 1)
    
class DenseBlock(nn.Module):
    def __init__(self, in_size, rate, num):
        super(DenseBlock, self).__init__()
        self.layers = nn.Sequential(*[denselayer(in_size+rate*i, rate) for i in range(num)])
        self.conv1 = nn.Conv2d(in_size+rate*num, rate, 1)

    def forward(self, x):
        out = self.conv1(self.layers(x))
        return out
        
class ResiDenseBlock(nn.Module):
    def __init__(self, in_size, rate, num):
        super(ResiDenseBlock, self).__init__()
        self.layers = nn.Sequential(*[denselayer(in_size+rate*i, rate) for i in range(num)])
        self.conv1 = nn.Conv2d(in_size+rate*num, in_size, 1)
    def forward(self, x):
        return x + self.conv1(self.layers(x))



# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# --------- RDAB (residual daul attention block) --------- # 

# CCA in RDAN
# contrast-aware channel attention module (imdn)  # git 
def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

# SA in RDAB
class SALayer(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, 1, keepdim=True)
        channel_pool = torch.cat([max_pool, avg_pool], dim=1)  
        y = self.conv_du(channel_pool)

        return x * y
   

# --------- feature refinemnet --------- # 
# for DFEB, MFEB
class FRP(nn.Module):
    def __init__(self, in_channels, out_channels, ca2, wf):
        super(FRP, self).__init__()
        self.ca2 = ca2
        self.ca1 = CALayer(in_channels, reduction=16, bias=False)
        
        if ca2 :
            self.ca2 = CALayer(out_channels, reduction=16, bias=False)
        
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.LeakyReLU()
        self.RDB = ResiDenseBlock(out_channels, wf, 3)

    def forward(self, x):
        # out = self.ca1(x)
        # out = x
        # out = self.act(self.conv3x3(out))
        # out = self.RDB(out)
        
        if self.ca2 :  # defb
            
            out = self.ca1(x)
            out = self.act(self.conv3x3(out))
            out = self.RDB(out)
            out = self.ca2(out)
            return out

        else :
            out = x
            out = self.act(self.conv3x3(out))
            out = self.RDB(out)
            return out


# --------- DFEB (Dual Feature Extraction Block) --------- # 

class DFEB(nn.Module):
    def __init__(self, n_feat, o_feat, bias, wf):  # n_feat=3
        super(DFEB, self).__init__()
        # feature extraction part
        # -- wavelet feature extraction part
        self.dwt = Downsample_dwt(wavename = 'haar')

        self.block = nn.ModuleList()
        for i in range(4):
            self.block.append(nn.Conv2d(n_feat, o_feat//2, 3, 1, 1, bias=bias))
        self.act_l = nn.LeakyReLU()
        self.act = nn.ReLU()
        
        self.idwt = IDWT_2D(wavename = 'haar')


        # -- identity feature extraction part
        self.conv = DenseBlock(n_feat, o_feat//2, 3)  # 96//2 (96=wf)

        # feature refinement part
        self.FR =  FRP(o_feat, o_feat, ca2=True, wf=wf)

# 40hr
    def forward(self, x):   

        # feature extraction part
        # ----- wavelet feature extraction
        subband = self.dwt(x)

        sub_tr = []
        for i, block in enumerate(self.block):  
            tr = block(subband[i])
            if i == 0 :
                tr = self.act_l(tr)  # for low frequency componet LL
            else : 
                tr = self.act(tr)
            sub_tr.append(tr)
        
        wavelet_feat = self.idwt(sub_tr[0], sub_tr[1], sub_tr[2], sub_tr[3])

        # ----- identity feature extraction
        # feat_iden = self.act(self.conv(x))
        feat_iden = self.conv(x)
        
        out = torch.cat([wavelet_feat, feat_iden], dim=1)
        # feature refinement part
        out = self.FR(out)

        # out = torch.cat([wavelet_feat, feat_iden], dim=1)
        # out = self.CA(out)
        # out = self.activate(self.conv3x3(out))
        
        return out


class MFEB(nn.Module):
    def __init__(self, n_feat, mid_feat, o_feat, bias, wf):
        super(MFEB, self).__init__()  
        # multi-scale feature extraction part
        self.d1 = nn.Conv2d(n_feat, mid_feat, kernel_size=3, stride=1, padding=1, dilation=1, bias=bias)
        self.d2 = nn.Conv2d(n_feat, mid_feat, kernel_size=3, stride=1, padding=2, dilation=2, bias=bias)
        self.d4 = nn.Conv2d(n_feat, mid_feat, kernel_size=3, stride=1, padding=4, dilation=4, bias=bias)
        self.d8 = nn.Conv2d(n_feat, mid_feat, kernel_size=3, stride=1, padding=8, dilation=8, bias=bias)
        # self.conv3x3 = nn.Conv2d(o_feat + n_feat, o_feat, 3, 1, 1, bias=bias)
        self.act = nn.LeakyReLU()
        # bias = FALSE add
        
        # feature refinement part
        self.FR = FRP(o_feat + n_feat, o_feat, ca2=False, wf=wf)
        # self.act = nn.LeakyReLU()
        # self.CA = CALayer(o_feat + n_feat, reduction=16, bias=bias)
        # 256 

    def forward(self, x):
        # multi-scale feature extraction part
        d1 = self.act(self.d1(x))  
        d2 = self.act(self.d2(x))
        d4 = self.act(self.d4(x))
        d8 = self.act(self.d8(x))
        
        feat = torch.cat([x, d1, d2, d4, d8], dim=1)

        # feature refinement part
        out = self.FR(feat)

        return out
 
# RDAB
class RDAB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, padding = 1, stride = 1, bias=False, act=nn.PReLU()):
        super(RDAB, self).__init__()
        
        modules_body = [nn.Conv2d(n_feat, n_feat, kernel_size, stride, padding, bias=bias), act, nn.Conv2d(n_feat, n_feat, kernel_size, stride, padding, bias=bias)]
        
        self.body = nn.Sequential(*modules_body)
        
        ## Spatial Attention
        self.SA = SALayer()  

        ## Contrast-aware Channel Attention   
        self.CCA = CCALayer(n_feat, reduction) 

        self.conv1x1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        cca_branch = self.CCA(res)

        res = sa_branch + cca_branch
        res = self.conv1x1(res)
        res += x
        return res
    

class Encoder(nn.Module):
    def __init__(self, in_size, out_size, dfeb, downsample, wf):
        super(Encoder, self).__init__()
        self.downsample = downsample        
        
        if dfeb :  
            self.body = DFEB(n_feat=3, o_feat=in_size, bias=False, wf=wf)
            
        else :  
            self.body = MFEB(n_feat=in_size, mid_feat=in_size//2, o_feat=out_size, bias=False, wf=wf)
            # HWBC_di(n_feat=in_size, o_feat=out_size, reduction = in_size//2, bias=False, act=nn.LeakyReLU()
        

        if downsample:
            self.downsample = nn.MaxPool2d(2)
        

    def forward(self, x):
        
        out = self.body(x)

        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class Decoder(nn.Module):
    def __init__(self, in_size, out_size, i, wf):
        super(Decoder, self).__init__()
        # self.up = PS_up(in_size, out_size, upscale=2) # 
        upscale=2
        self.up_conv = nn.Conv2d(in_size, in_size*upscale, 1, 1, 0, bias=False)
        # bias = False add
        self.PS = nn.PixelShuffle(upscale)

        self.conv3x3 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=False)
        self.act = nn.LeakyReLU()
        # self.conv_block = UNetConvBlock(in_size, out_size, downsample=False, wf=wf)

        self.att = RDAB(out_size)

        self.bili = bili_resize(2 ** i)
        self.conv1x1 = nn.Conv2d((2 ** i) * wf, wf, 3, 1, 1, bias=False)
        # bias = False add


    def forward(self, x, bridge):

        up = self.up_conv(x)
        up = self.PS(up)
        out = torch.cat([up, bridge], dim=1)
        out = self.act(self.conv3x3(out))
        out = self.att(out)

        out_up = self.bili(out)
        out_up = self.conv1x1(out_up)

        return out, out_up


class SKFF_fcl(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF_fcl, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fcl = nn.Sequential(
            nn.Linear(in_channels, d, bias=False),   # out_c : in_c // reduction in selayer
            nn.LeakyReLU()                               # relu in selayer
        )
        
        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Linear(d, in_channels, bias=False))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size, n_feats, H, W = inp_feats[1].shape

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U).view(batch_size,n_feats)
        # feats_Z = self.conv_du(feats_S)
        feats_Z = self.fcl(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]

        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)

        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V


class MRLENet(nn.Module):
    def __init__(self, in_channels=3, wf=32, depth=4):  # wf(32)
        super(MRLENet, self).__init__()
        self.depth = depth


        # encoder part 
        # input --> DFEB
        # self.in_conv = DFEB(n_feat=3, o_feat=wf, bias=False)

        self.down_path = nn.ModuleList()
     
      
        # encoder of UNet-64
        prev_channels = wf
        for i in range(depth):  # 0,1,2,3
            downsample = True if (i + 1) < depth else False
            dfeb = True if i == 0 else False
            self.down_path.append(Encoder(prev_channels, (2 ** i) * wf, dfeb, downsample, wf))

            prev_channels = (2 ** i) * wf
        
        self.att = RDAB(prev_channels)
        # DAU(prev_channels)
        
        self.skip_conv = nn.ModuleList()
        self.skip_act = nn.LeakyReLU()

        # decoder part
        self.bottom_conv = nn.Conv2d(prev_channels, wf, 3, 1, 1)
        self.bottom_up = bili_resize(2 ** (depth-1))

        self.up_path = nn.ModuleList()
        # self.conv_up = nn.ModuleList()
        

        for i in reversed(range(depth - 1)):
            self.up_path.append(Decoder(prev_channels, (2 ** i) * wf, i, wf))
            self.skip_conv.append(nn.Conv2d((2 ** i) * wf, (2 ** i) * wf, 3, 1, 1))
            # self.conv_up.append(nn.Sequential(*[bili_resize(2 ** i), nn.Conv2d((2 ** i) * wf, wf, 3, 1, 1)]))
            prev_channels = (2 ** i) * wf

        
        self.skff_0 = SKFF_fcl(in_channels=wf, height=2)
        self.skff_1 = SKFF_fcl(in_channels=wf, height=2)
        self.skff_2 = SKFF_fcl(in_channels=wf, height=2)
        # self.last = conv3x3(prev_channels, in_chn, bias=True)
        self.last = nn.Conv2d(prev_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        img = x
        # scale_img = img

        ##### shallow conv #####
        # x1 = self.in_conv(img)

        x1 = img
        encs = []
        ######## UNet-64 ########
        # Down-path (Encoder)
        for i, down in enumerate(self.down_path):
            if i == 0:
                x1, x1_up = down(x1)
                encs.append(x1_up)
            elif (i + 1) < self.depth:
                x1, x1_up = down(x1)
                encs.append(x1_up)
            else:
                x1 = down(x1)
                x1 = self.att(x1)

        # Up-path (Decoder)
        ms_result = [self.bottom_up(self.bottom_conv(x1))]
        for i, up in enumerate(self.up_path):
            x1, x1_up = up(x1, self.skip_act(self.skip_conv[i](encs[-i - 1])))
            # ms_result.append(self.conv_up[i](x1))
            ms_result.append(x1_up)
        # Multi-scale selective feature fusion
        msff_result_0 = self.skff_0([ms_result[0], ms_result[1]])  # wf
        msff_result_1 = self.skff_1([msff_result_0, ms_result[2]]) 
        msff_result_2 = self.skff_2([msff_result_1, ms_result[3]]) 

        ##### Reconstruct #####
        out_1 = self.last(msff_result_2) + img 

        return out_1
    

if __name__ == "__main__":
    from thop import profile
    input = torch.ones(1, 3, 256, 256, dtype=torch.float, requires_grad=False).cuda()

    model = MRLENet(in_channels=3, wf=96, depth=4).cuda()
    print(model)
    out = model(input)
    flops, params = profile(model, inputs=(input,))

    print('input shape:', input.shape)
    print('parameters:', params/1e6)
    print('flops', flops/1e9)
    print('output shape', out.shape)