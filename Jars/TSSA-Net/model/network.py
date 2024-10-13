import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import np
import math
from einops import rearrange
import numpy as np


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=opt.lr_decay_gamma)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                   factor=opt.lr_decay_gamma,
                                                   patience=opt.lr_decay_patience)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'mean_space':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1 / (height * weight))
            elif init_type == 'mean_channel':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1 / (channel))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)



def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


'''class l1_regularization(nn.Module):
    def __init__(self):
        super(l1_regularization, self).__init__()
        self.loss = nn.L1Loss(size_average=False)


    def __call__(self,input1,input2):
     loss=self.loss(input1,input2)
     loss+=0.01*loss

     return loss'''

class SumToOneLoss(nn.Module):
    def __init__(self):
        super(SumToOneLoss, self).__init__()
        self.register_buffer('one', torch.tensor(1, dtype=torch.float))

        self.loss = nn.L1Loss(size_average=False)

    def get_target_tensor(self, input):
        target_tensor = self.one
        return target_tensor.expand_as(input)

    def __call__(self, input):
        input = torch.sum(input, 1)
        target_tensor = self.get_target_tensor(input)
        # print(input[0,:,:])
        loss = self.loss(input, target_tensor)
        # loss = torch.sum(torch.abs(target_tensor - input))
        return loss


def kl_divergence(p, q):
    p = F.softmax(p,dim=1)
    q = F.softmax(q,dim=1)

    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2


class SparseKLloss(nn.Module):
    def __init__(self):
        super(SparseKLloss, self).__init__()
        self.register_buffer('zero', torch.tensor(0.01, dtype=torch.float))

    def __call__(self, input):
        input = torch.sum(input, 0, keepdim=True)
        target_zero = self.zero.expand_as(input)
        loss = kl_divergence(target_zero, input)
        return loss




# 定义EMSR模块
class MRS_Block(nn.Module):
    def __init__(self, useSoftmax=True):
        super(MRS_Block, self).__init__()

        channel = 64
        #self.batch_norm_layers = nn.BatchNorm2d(num_features=64)
        self.conv0 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,bias=False)
        self.conv_3_1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,bias=False)
        #self.conv_3_2 = nn.Conv2d(in_channels=channel * 1, out_channels=channel * 1, kernel_size=3, stride=1, padding=1,bias=False)
        self.conv_5_1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=5, stride=1, padding=2,bias=False)
        #self.conv_5_2 = nn.Conv2d(in_channels=channel * 1, out_channels=channel * 1, kernel_size=5, stride=1, padding=2,bias=False)
        self.confusion = nn.Conv2d(in_channels=channel * 2, out_channels=channel, kernel_size=1, stride=1, padding=0,bias=False)
        self.conv_final = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0,bias=False)
        self.usesoftmax = useSoftmax
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        #out = self.conv0(x)
        output_3_1 = self.conv_3_1(x)
        output_3_1 = self.relu(output_3_1)
        output_5_1 = self.conv_5_1(x)
        output_5_1 = self.relu(output_5_1)
        input = torch.cat([output_3_1, output_5_1],1)
        input =self.confusion(input)
        input = self.relu(input)
        #output = torch.add(input, out)
        #output= self.conv_final(output)
        #output = self.relu(output)
        return input



class SpeMultiHead(torch.nn.Module):
    def __init__(self,l_h,n_heads):   # [1966,4]
        super(SpeMultiHead, self).__init__()
        self.input_dim=l_h*l_h
        self.d_k = self.d_v = self.input_dim // n_heads  # 491.5
        self.n_heads = n_heads
        self.W_Q = torch.nn.Linear(self.input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(self.input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(self.input_dim, self.d_v * self.n_heads, bias=False)

        self.relu = nn.LeakyReLU(0.2, True)
        self.d_k = self.d_v = self.input_dim
        self.output_dim = self.input_dim

    def forward(self,z):
        b, c, h, w = z.shape

        a = rearrange(z,'b c h w   ->  b c (h w) ')
        B ,C,S =a.shape
        Q = self.W_Q(a).reshape(self.n_heads*B,self.d_k // self.n_heads, C )
        K = self.W_K(a).reshape(self.n_heads*B,self.d_k // self.n_heads, C )
        V = self.W_V(a).reshape(self.n_heads*B,self.d_k // self.n_heads,C )
        scores = torch.matmul(Q.transpose(-2,-1), K) / np.sqrt(self.d_k )
        attn = torch.nn.Softmax(dim=-1)(scores)
        print("scoresscores:",scores.shape)
        print("VV:", V.shape)

        context = self.relu(torch.matmul(V,attn))
        context = context.transpose(-2,-1)
        output = context.reshape(b, c, h, w)
        return output



class MultiHead(torch.nn.Module):
    def __init__(self,n_heads):   # [1966,4]
        super(MultiHead, self).__init__()
        input_dim=64
        self.n_heads=n_heads
        self.d_k = self.d_v = input_dim // n_heads

        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.relu = nn.LeakyReLU(0.2, True)

        self.output_dim = input_dim

    def forward(self,z):
        b, c, h, w = z.shape

        a = rearrange(z,'b c h w   ->  (b h) w c ')
        B1 ,w ,C =a.shape
        Q1 = self.W_Q(a).reshape(self.n_heads*b*h, self.d_k , w )
        V1 = z.reshape(b*h, w , c)
        V1 = self.W_K(V1).reshape(self.n_heads*b*h, self.d_k, w )
        scores1 = torch.matmul(Q1.transpose(-2,-1), Q1) / np.sqrt(self.d_k )
        print("scores1:",scores1.shape)
        attn1 = torch.nn.Softmax(dim=-1)(scores1)
        context1 = self.relu(torch.matmul(V1,attn1))


        m = rearrange(z,'b c h w   ->  (b w) h c ')
        B2 ,h ,C1 =m.shape
        Q2 = self.W_Q(m).reshape(self.n_heads * b * w, self.d_k , h)
        V2 = z.reshape(b * w, h, c)
        V2 = self.W_K(V2).reshape(self.n_heads * b * w, self.d_k , h)
        scores2 = torch.matmul(Q2.transpose(-2, -1), Q2) / np.sqrt(self.d_k)
        print("scores2:", scores2.shape)
        attn2 = torch.nn.Softmax(dim=-1)(scores2)
        context2 = self.relu(torch.matmul(V2, attn2))
        context = torch.add(context1,context2)

        output = context.reshape(b, c, h, w)
        return output

class crossMultiHead(torch.nn.Module):
    def __init__(self,l_h,h_h,n_heads):   # [1966,4]
        super(crossMultiHead, self).__init__()
        self.input_dim=l_h*l_h
        self.input_dim_1 = 64
        self. n_heads= n_heads
        self.d_k = self.d_v = self.input_dim// n_heads
        #self.d_k = self.d_v = self.input_dim // n_heads
        self.d_k1 = self.d_v1 = self.input_dim_1// n_heads
        self.n_heads = n_heads
        self.W_Q = torch.nn.Linear(self.input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(self.input_dim, self.d_k * self.n_heads, bias=False)

        self.W_V = torch.nn.Linear(self.input_dim_1, self.d_k1* self.n_heads, bias=False)

        self.relu = nn.LeakyReLU(0.2, True)


    def forward(self,z,x):
        b, c, h, w = z.shape
        a = rearrange(z,'b c h w   ->  b c (h w) ')
        B ,C,S =a.shape

        b1, c1, h1, w1 = x.shape
        r = rearrange(x,'b c h w   ->  b  (h w) c ')
        B1,S1,C1 =r.shape

        Q = self.W_Q(a).reshape(self.n_heads*B , self.d_k , C )
        K = self.W_K(a).reshape(self.n_heads*B , self.d_k , C )
        V = self.W_V(r).reshape(self.n_heads*B1 , S1//self.n_heads , C )
        print("V:", V.shape)
        scores = torch.matmul(Q.transpose(-2,-1), K) / np.sqrt(self.d_k )
        attn = torch.nn.Softmax(dim=-1)(scores)
        print("attnattn:",attn.shape)

        context = self.relu(torch.matmul(V,attn))
        print(context.shape)
        context = context.transpose(-2,-1)
        output = context.reshape(b1, c1, h1, w1)
        return output

class SpeT(torch.nn.Module):
    def __init__(self,l_h,n_heads):
        super(SpeT, self).__init__()
        self.mulit=SpeMultiHead(l_h,n_heads)
        self.LN = torch.nn.LayerNorm(l_h,l_h)
        self.CNN = MRS_Block()
        self.BN = torch.nn.BatchNorm2d(64)


    def forward(self,x):
        out = self.mulit(x)
        out1 = self.BN(x+out)
        output = self.BN(self.CNN(out1)+out1)
        return output

class SpaT(torch.nn.Module):
    def __init__(self,h_h,n_heads):
        super(SpaT, self).__init__()
        self.mulit=MultiHead(n_heads)
        self.LN = torch.nn.LayerNorm(h_h,h_h)
        self.CNN = MRS_Block()
        self.BN = torch.nn.BatchNorm2d(64)


    def forward(self,x):
        out = self.mulit(x)
        out1 = self.BN(x+out)
        output = self.BN(self.CNN(out1)+out1)
        return output


class CrossT(torch.nn.Module):
    def __init__(self,l_h,h_h,n_heads):
        super(CrossT, self).__init__()
        self.mulit=crossMultiHead(l_h,h_h,n_heads)
        self.LN = torch.nn.LayerNorm(h_h,h_h)
        self.CNN = MRS_Block()
        self.BN = torch.nn.BatchNorm2d(64)


    def forward(self,z,x):
        out = self.mulit(z,x)
        out1 = self.BN(x+out)
        output = self.BN(self.CNN(out1)+out1)
        return output


class Msi2Delta(nn.Module):
    def __init__(self,l_h ,h_h,n_heads,input_c, output_c, out_channels_MSRB=64, ngf=64, n_res=3, useSoftmax=True):
        super(Msi2Delta, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=input_c, out_channels=64, kernel_size=3, stride=1, padding=1,bias=False)
        self.residual1 = SpaT(h_h,n_heads)
        self.residual2 = SpaT(h_h, n_heads)
        self.residual3 = SpaT(h_h, n_heads)
        self.residual4 = SpaT(h_h, n_heads)
        self.Ct = CrossT(l_h,h_h,n_heads)
        self.finalconv2d = nn.Conv2d(out_channels_MSRB , output_c, 1, 1, 0)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.LeakyReLU(0.2, True)


    def forward(self,x,z):
        out0 = self.conv_input(x)
        out1 = self.residual1(out0)
        out2 = self.residual1(out1)
        out3 = self.residual1(out2)
        out = out0+out1+out2+out3
        out = self.Ct(z,out)
        out = self.finalconv2d(out)
        out = self.relu(out)
        out = self.softmax(out)
        return out

class Lr2Delta(nn.Module):
    def __init__(self, l_h, n_heads,input_c,output_c,out_channels_MSRB=64, ngf=64, n_res=3, useSoftmax=True):
        super(Lr2Delta, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=input_c, out_channels=64, kernel_size=3, stride=1, padding=1,
                                    bias=False)
        self.residual1 = SpeT(l_h, n_heads)
        self.residual2 = SpeT(l_h, n_heads)
        self.residual3 = SpeT(l_h, n_heads)
        self.residual4 = SpeT(l_h, n_heads)

        self.finalconv2d = nn.Conv2d(out_channels_MSRB, output_c, 1, 1, 0)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.LeakyReLU(0.2, True)


    def forward(self, x):
        out0 = self.conv_input(x)
        out1 = self.residual1(out0)
        out2 = self.residual1(out1)
        out3 = self.residual1(out2)
        out_f = out0 + out1 + out2 + out3
        out = self.finalconv2d(out_f)
        out = self.relu(out)
        out = self.softmax(out)
        return out,out_f



def define_msi2s(l_h,h_h,n_heads,input_ch, output_ch, gpu_ids, init_type='kaiming', init_gain=0.02, useSoftmax=True):
    net = Msi2Delta(l_h=l_h,h_h=h_h,n_heads=n_heads,input_c=input_ch, output_c=output_ch, ngf=64 , useSoftmax=useSoftmax)

    return init_net(net, init_type, init_gain, gpu_ids)



def define_s2img(input_ch, output_ch, gpu_ids, init_type='kaiming', init_gain=0.02):
    net = S2Img(input_c=input_ch, output_c=output_ch)
    return init_net(net, init_type, init_gain, gpu_ids)


class S2Img(nn.Module):
    def __init__(self, input_c, output_c):
        super(S2Img, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_c, output_c, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.net(x)



def define_lr2s(l_h, n_heads,input_ch, output_ch, gpu_ids, init_type='kaiming', init_gain=0.02, useSoftmax=True):

    net = Lr2Delta(l_h=l_h, n_heads=n_heads,input_c=input_ch, output_c=output_ch, ngf=64, useSoftmax=useSoftmax)
    return init_net(net, init_type, init_gain, gpu_ids)




def define_L2img(input_ch, output_ch, gpu_ids, init_type='kaiming', init_gain=0.02):
    net = L2Img(input_c=input_ch, output_c=output_ch)
    return init_net(net, init_type, init_gain, gpu_ids)

class L2Img(nn.Module):
    def __init__(self, input_c, output_c):
        super(L2Img, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_c, output_c, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.net(x)





def define_psf(scale, gpu_ids, init_type='mean_space', init_gain=0.02):
    net = PSF(scale=scale)
    return init_net(net, init_type, init_gain, gpu_ids)

class PSF(nn.Module):
    def __init__(self, scale):
        super(PSF, self).__init__()
        self.net = nn.Conv2d(1, 1, scale, scale, 0, bias=False)
        self.scale = scale
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch, channel, height, weight = list(x.size())
        return torch.cat([self.net(x[:, i, :, :].view(batch, 1, height, weight)) for i in range(channel)], 1)


def define_hr2msi(args, hsi_channels, msi_channels, sp_matrix, sp_range, gpu_ids, init_type='mean_channel',
                  init_gain=0.02):
    if args.isCalSP == False:
        net = matrix_dot_hr2msi(sp_matrix)
    elif args.isCalSP == True:
        net = convolution_hr2msi(hsi_channels, msi_channels, sp_range)
    return init_net(net, init_type, init_gain, gpu_ids)


class convolution_hr2msi(nn.Module):
    def __init__(self, hsi_channels, msi_channels, sp_range):
        super(convolution_hr2msi, self).__init__()

        self.sp_range = sp_range.astype(int)
        self.length_of_each_band = self.sp_range[:, 1] - self.sp_range[:, 0] + 1
        self.length_of_each_band = self.length_of_each_band.tolist()
        # import ipdb
        # ipdb.set_trace()
        self.conv2d_list = nn.ModuleList([nn.Conv2d(x, 1, 1, 1, 0, bias=False) for x in self.length_of_each_band])
        # self.scale_factor_net = nn.Conv2d(1,1,1,1,0,bias=False)

    def forward(self, input):
        # batch,channel,height,weight = list(input.size())
        # scaled_intput = torch.cat([self.scale_factor_net(input[:,i,:,:].view(batch,1,height,weight)) for i in range(channel)], 1)
        scaled_intput = input
        cat_list = []
        for i, layer in enumerate(self.conv2d_list):
            input_slice = scaled_intput[:, self.sp_range[i, 0]:self.sp_range[i, 1] + 1, :, :]
            out = layer(input_slice).div_(layer.weight.data.sum(dim=1).view(1))
            cat_list.append(out)
        return torch.cat(cat_list, 1).clamp_(0, 1)


class matrix_dot_hr2msi(nn.Module):
    def __init__(self, spectral_response_matrix):
        super(matrix_dot_hr2msi, self).__init__()
        self.register_buffer('sp_matrix', torch.tensor(spectral_response_matrix.transpose(1, 0)).float())

    def __call__(self, x):
        batch, channel_hsi, heigth, width = list(x.size())
        channel_msi_sp, channel_hsi_sp = list(self.sp_matrix.size())
        hmsi = torch.bmm(self.sp_matrix.expand(batch, -1, -1),
                         torch.reshape(x, (batch, channel_hsi, heigth * width))).view(batch, channel_msi_sp, heigth,
                                                                                      width)
        return hmsi


class NormGANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(NormGANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.L1Loss(size_average=False)

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class NonZeroClipper(object):

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(0, 1e8)


class ZeroOneClipper(object):

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(0, 1)


class SumToOneClipper(object):

    def __call__(self, module):
        if hasattr(module, 'weight'):
            if module.in_channels != 1:
                w = module.weight.data
                w.clamp_(0, 10)
                w.div_(w.sum(dim=1, keepdim=True))
            elif module.in_channels == 1:
                w = module.weight.data
                w.clamp_(0, 5)
