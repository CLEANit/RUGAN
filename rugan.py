import torch
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np


from options.train_options import TrainOptions
from torchsummary import summary
from torchsummary import torch_summarize
import pickle

cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)
datafolder = './'


'''Channels and dimension of latent space'''
nclasses  = 1   # =1 if using continuous labels, = # of classes otherwise
latent_channels = 48+nclasses
latent_channel_dim_1 = 6  #spatial dimensions of latent channels
latent_channel_dim_2 = 6
training_dim_1 = 12  #spatial dimensions of training set
training_dim_2 = 12
norm_critic = False  #if LayerNorm should be used for critic


'''Dataset class, returns configs+dopings for training'''
class trainingData(Dataset):
    def __init__(self, opt):

        with open(datafolder+'labels_reg.p','rb') as f:
            labels = pickle.load(f)


        with open(datafolder+'configs.p','rb') as f:
            set = pickle.load(f)


        firstlen = set.shape[0]

        labels = torch.from_numpy(labels.astype(np.float32)).to(device)
        set = torch.from_numpy(set.astype(np.float32)).unsqueeze_(1).to(device)
        for k in range(1,4): #augment data set by rotating samples
            set = torch.cat((set, torch.rot90(set[:firstlen], k, [2,3])))

            labels = torch.cat((labels,labels[:firstlen]))

        set = torch.cat((set,torch.flip(set[:firstlen], [1, 3])))

        labels = torch.cat((labels,labels[:firstlen]))
        set = torch.cat((set,torch.flip(set[:firstlen], [1, 2])))
        labels = torch.cat((labels,labels[:firstlen]))

        self.labels = labels
        self.set = set
        print(self.set.shape)

    def __len__(self):
        return self.set.shape[0]

    def __getitem__(self,idx):
        return self.set[idx], self.labels[idx]

    def getObservables(self, N):
        idx = random.sample(range(0, self.labels.shape[0]), N)
        return self.labels[idx]


'''custom weights initialization'''
def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif type(m) == nn.BatchNorm2d or type(m) == nn.InstanceNorm2d :
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


'''For convolutions with PBC or zero padding'''
class PeriodicPadding(nn.Module):
    def __init__(self, kSize, mode = 'zero'):
        super(PeriodicPadding, self).__init__()
        self.kSize = kSize
        self.pad = nn.functional.pad
        self.mode = mode

    def forward(self, input):
        if self.mode == 'periodic':
            if self.kSize == 1:
                return input

            if self.kSize%2 != 0: #assume this is for dimensionality-preserving convolutions
                padsize = int((self.kSize-1)/2.)
                padding = (padsize, padsize, padsize, padsize)
                len1 = input.size(2)
                len2 = input.size(3)
                input = input.repeat(1,1,3,3)[..., (len1-int(padsize)):(2*len1 + int(padsize)), (len2-int(padsize)):(2*len2 + int(padsize))]
                # input = self.pad(input, padding, mode='circular')
                return input

            elif input.size(2) %2 == 0 and input.size(3) %2 ==0:  #assume this is for downsampling,
                return input

        elif self.mode == 'zero':

            if self.kSize == 1 or self.kSize == 2:
                return input
            else:
                padsize = int((self.kSize-1)/2.)
                padding = (padsize, padsize, padsize, padsize)
                input = self.pad(input, padding, mode='constant')
                return input


def mask(input): #creates circular configuration
    input[:, :, 0, :] = 0.
    input[:, :, 11, :] = 0.
    input[:, :, :, 0] = 0.
    input[:, :, :, 11] = 0.

    input[:, :, 1, 0:4] = 0.
    input[:, :, 10, 0:4] = 0.
    input[:, :, 1, 8:12] = 0.
    input[:, :, 10, 8:12] = 0.

    input[:, :,  0:4, 1] = 0.
    input[:, :,  0:4, 10] = 0.
    input[:, :, 8:12, 1] = 0.
    input[:, :, 8:12, 10] = 0.
    return input


class resBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kSize=3, scale=None, hw = 0, mode = 'zero'):
        super(resBlock, self).__init__()

        self.scale = scale
        self.input_channels = input_channels
        self.output_channels = output_channels

        init = []
        block = []


        if scale == 'up':
            init += [nn.Upsample(scale_factor=2)]
            init += [nn.Conv2d(input_channels, output_channels, kernel_size = 1)]

            block += [nn.BatchNorm2d(input_channels)]
            block += [nn.ReLU()]
            block += [nn.Upsample(scale_factor=2)]
            block += [PeriodicPadding(kSize, mode)]
            block += [nn.Conv2d(input_channels, output_channels, kSize, bias = False)]
            block += [nn.BatchNorm2d(output_channels)]
            block += [nn.ReLU()]
            block += [PeriodicPadding(kSize, mode)]
            block += [nn.Conv2d(output_channels, output_channels, kSize)]

        elif scale =='down':
            init += [PeriodicPadding(2, mode)]
            init += [nn.AvgPool2d(kernel_size=2, stride = 2)]
            init += [nn.Conv2d(input_channels, output_channels, kernel_size = 1)]

            if norm_critic == True:
                block += [nn.LayerNorm([input_channels,hw,hw])]
            block += [nn.ReLU()]
            block += [PeriodicPadding(kSize, mode)]
            block += [nn.Conv2d(input_channels, input_channels, kSize, bias = False)]
            if norm_critic == True:
                block += [nn.LayerNorm([input_channels,hw,hw])]
            block += [nn.ReLU()]
            block += [PeriodicPadding(kSize, mode)]
            block += [nn.Conv2d(input_channels, output_channels, kSize)]
            block += [PeriodicPadding(2, mode)]
            block += [nn.AvgPool2d(kernel_size=2,stride=2)]

        elif scale == 'same':
            init += [nn.Conv2d(input_channels, output_channels, kernel_size = 1)]

            block += [nn.BatchNorm2d(input_channels)]
            block += [nn.ReLU()]
            block += [PeriodicPadding(kSize, mode)]
            block += [nn.Conv2d(input_channels, input_channels, kSize, bias = False)]
            block += [nn.BatchNorm2d(input_channels)]
            block += [nn.ReLU()]
            block += [PeriodicPadding(kSize, mode)]
            block += [nn.Conv2d(input_channels, output_channels, kSize)]

        self.init = nn.Sequential(*init)
        self.block = nn.Sequential(*block)
    def forward(self, input):

        if self.scale == 'same' and self.input_channels == self.output_channels:
            original = input
        else:
            original = self.init(input)

        output = self.block(input)

        return original + output

'''Generator with channel dimension equal to DIM'''
DIMG = 128
class G(nn.Module):
    def __init__(self):
        super(G,self).__init__()

        model = []
        model += [resBlock(latent_channels, DIM, kSize=3, scale='same')]
        model += [resBlock(DIM,DIM, kSize = 3, scale ='up')]
        model += [resBlock(DIM, DIM, kSize=3, scale='same')]
        model += [nn.BatchNorm2d(DIM)]
        model += [nn.ReLU()]
        model += [PeriodicPadding(3)]
        model += [nn.Conv2d(DIM,1, kernel_size=3, bias=False)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        img = self.model(input)
        return img


''''Critic with channel dimension equal to DIM. Batchnorm not allowed here for GP calculation '''
DIM=128
class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()


        model = []
        model += [resBlock(1+nclasses,DIM, kSize =3, scale = 'down')] #(6,6)
        model += [nn.Dropout(0.25)]
        model += [resBlock(DIM,DIM, kSize =3, scale = 'down')] #(3,3)
        model += [nn.Dropout(0.25)]
        model += [nn.Conv2d(DIM, DIM, kernel_size=3)]
        model += [nn.ReLU()]
        self.model= nn.Sequential(*model)
        self.out_layer = nn.Sequential(nn.Linear(1*1*DIM ,1))

    def forward(self, input):
        with torch.no_grad():
            input = mask(input)
        out = self.model(input)
        out = out.view(out.shape[0], -1)
        validity = self.out_layer(out)
        return validity, out


'''concatenate sample and label for critic input'''
def cat_input_label(input, Elabel):
    labels  = torch.zeros(input.shape[0], nclasses, input.shape[2], input.shape[3], device = device)
    for i in range(input.shape[0]):
        labels[i, 0, ...] = Elabel[i]
    input = torch.cat( (input, labels), 1)
    return input

'''calculates gradient penalty for Lipschitz-continuity'''
def calc_gradient_penalty(netD, real_data, fake_data, opt):

    alpha = torch.rand(opt.bSize, 1, 1, 1, device = device)
    alpha = alpha.expand_as(real_data)
    interpolates = alpha * real_data + ((1-alpha) * fake_data)

    interpolates = autograd.Variable(interpolates, requires_grad = True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(gradients.size(0), -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return opt.gradient_penalty * ((gradients_norm - 1) ** 2).mean()

'''calculates consistensy term for Lipschitz-continuity'''
def calc_consistency_term(disc, real_data):
    d1, d_1 = disc(real_data)
    d2, d_2 = disc(real_data)
    ct = (d1-d2).norm(2, dim =1) + 0.1 * (d_1 - d_2).norm(2,dim=1)
    return 2*ct.mean()

class WGAN(object):
    def __init__(self, data, opt):
        self.data = data
        self.g = None
        self.d = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.num_epochs = opt.num_epochs
        self.bSize = opt.bSize
        self.build_model(opt)

    def build_model(self, opt):
        self.g = G().to(device)
        summary(self.g, (latent_channels,latent_channel_dim_1,latent_channel_dim_2))
        self.d =D().to(device)
        summary(self.d, (1+nclasses,training_dim_1,training_dim_2))

        self.g.apply(weights_init)
        self.d.apply(weights_init)

        self.g_optimizer = optim.Adam(self.g.parameters(), lr=opt.lr,betas=(opt.beta1, opt.beta2), weight_decay=opt.wd)
        self.d_optimizer = optim.Adam(self.d.parameters(), lr=opt.lr,betas=(opt.beta1, opt.beta2), weight_decay=opt.wd)

    def save_model(self,  filename):
        state = {
            'g_dict'  : self.g.state_dict(),
            'd_dict'  : self.d.state_dict(),

            'g_optimizer' : self.g_optimizer.state_dict(),
            'd_optimizer' : self.d_optimizer.state_dict(),
        }
        torch.save(state, filename)

    def load_eval(self, opt):
        state = torch.load(datafolder+'save'+str(opt.epoch)+'.pth')
        self.g.load_state_dict(state['g_dict'])
        self.d.load_state_dict(state['d_dict'])
        self.g.eval()
        self.d.eval()

    def load_train(self ,filename):
        state = torch.load(datafolder+'save'+str(opt.epoch)+'.pth')
        self.g.load_state_dict(state['g_dict'])
        self.d.load_state_dict(state['d_dict'])

        self.g_optimizer.load_state_dict(state['g_optimizer'])
        self.d_optimizer.load_state_dict(state['d_optimizer'])

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    '''set requires_grad=False to avoid unnecessary computation'''
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    '''generate a batch of configurations at dopings distributed like the training set'''
    def sample_generator(self, opt):
        Es = self.data.getObservables(opt.bSize)
        noise = torch.randn((opt.bSize, latent_channels-nclasses, latent_channel_dim_1, latent_channel_dim_2), device = device)
        noise = cat_input_label(noise, Es)
        latent_samples = Variable(noise)
        generated_data = self.g(latent_samples)

        return generated_data, Es

    '''generate N configurations at a target doping E'''
    def target_sample(self, E, N):
        noise = torch.randn((N, latent_channels-nclasses, latent_channel_dim_1, latent_channel_dim_2), device = device)
        noise = cat_input_label(noise, E)

        latent_samples = Variable(noise)
        generated_data = self.g(latent_samples)
        return generated_data


    def train_critic_iteration(self, batch, E_batch, opt):

        self.set_requires_grad(self.g, False)
        self.set_requires_grad(self.d, True)
        self.reset_grad()

        fake = self.target_sample(E_batch, self.bSize)

        input_real = cat_input_label(batch, E_batch)
        d_real, _ = self.d(input_real)

        input_fake = cat_input_label(fake, E_batch)
        d_fake, _ = self.d(input_fake)

        d_gradient_penalty = calc_gradient_penalty(self.d, input_real, input_fake, opt)
        d_ct = calc_consistency_term(self.d, input_real)
        d_loss = d_fake.mean() - d_real.mean() + d_gradient_penalty + d_ct
        d_loss.backward()
        self.d_optimizer.step()

        return d_loss

    def train_generator_iteration(self, opt):

        self.set_requires_grad(self.g, True)
        self.set_requires_grad(self.d, False)

        self.reset_grad()

        fake, Es = self.sample_generator(opt)

        input_fake = cat_input_label(fake, Es)
        d_fake, _ = self.d(input_fake)

        g_loss_d = -1.0 * d_fake.mean()

        g_loss = g_loss_d
        g_loss.backward()
        self.g_optimizer.step()



    def train(self,opt):
        dataset_loader = DataLoader(dataset = self.data, batch_size=self.bSize, shuffle = True, num_workers = 0, drop_last=True) # add pin_memory=True and increase num_workers if dataset is on CPU
        f= open(datafolder+'training.txt',"w+")

        for step in range(self.num_epochs):
            iter_step = 0
            for i, (samples, E_samples) in enumerate(dataset_loader):
                d_loss = self.train_critic_iteration(samples,E_samples, opt)
                if iter_step % opt.critic_iters == 0:  #only train generator after critic_iters steps, allows for better estimation of W-distance
                    self.train_generator_iteration(opt)
                    print("i:"+str(step*float(len(self.data))/self.bSize + i)+"\t" + "D loss:"+str(d_loss.item()))
                    f.write(str(step*float(len(self.data))/self.bSize + i)+"\t"+str( d_loss.item())+"\n")
                    f.flush()
                iter_step +=1


            self.save_model(datafolder+'save'+str(step+1)+'.pth')

if __name__ == "__main__":

    opt = TrainOptions().parse()
    data = trainingData(opt)
    wgan = WGAN(data, opt)

    wgan.train(opt)
