import torch
a= torch.arrange(0,4).view(1,1,2,2).float()


def dft_amp(img):
    fft_img = torch.view_as_real(torch.fft.rfftn(img, dim = (2,3), norm="backward")
    print("Pytorch FFT 1.9", fft_im)

b = dft_amp(a)
print('pytorch 1.9 amp', b)

