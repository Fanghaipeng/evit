import torch
pthfile = "/apsarapangu/disk2/fanghaipeng.fhp/CVPR/evit/output/EvitS_300_8_128_shrink_fuse_0.5_12nga_lfb_4sto/checkpoint.pth"
net = torch.load(pthfile)
print(net['model'].keys())
for i in net['model'].keys():
    if "relative" in i:
        print(i)
        print(net['model'][i][0][0][10])
    # print(net['model']['ypos'])

# print(net['model'])