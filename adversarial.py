import torch
import torchvision.transforms as transforms


def l2_projection(step, start, eps):
    dif = step - start
    dif = dif.reshape(-1, 32*32*3)
    n = torch.sum(dif ** 2, dim=1).reshape(-1, 1) ** (1/2.) # l2 norm
    out = torch.where(n <= eps, dif, (dif / n) * eps)
    return out.view(-1, 3, 32, 32)

def li_projection(step, start, eps):
    return torch.clamp(step - start, min=-eps, max=eps)


def pgd(inputs, labels, model, stepsize, eps, steps, constraint='l_2'):
    loss = torch.nn.CrossEntropyLoss(reduction='sum')
    base = inputs.clone().detach()
    for _ in range(steps):
        inputs.requires_grad = True
        model.zero_grad()
        l = loss(model(inputs), labels)
        l.backward()
        if constraint == 'l_inf':
            step = inputs + stepsize * inputs.grad.sign()
            inputs = (base + li_projection(step, base, eps)).detach()
        elif constraint == 'l_2':
            # stepsize = 2.5 * eps / 100
            step = inputs + stepsize * inputs.grad #* len(inputs)
            inputs = (base + l2_projection(step, base, eps)).detach()

    return inputs

def pgd_attack(input, label, model, stepsize, eps, steps, constraint='l_2'):
    invertTransform = transforms.Compose([
                                transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    # input = invertTransform(input).view(1, 3, 32, 32)
    transform = transforms.Compose([
                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    ])
    input = transform(input).view(1, 3, 32, 32)
    loss = torch.nn.CrossEntropyLoss(reduction='sum')
    base = input.clone().detach()
    label = torch.Tensor([label]).cpu().long()
    print(input.shape, label.shape)

    for _ in range(steps):
        input.requires_grad = True
        model.zero_grad()
        l = loss(model(input), label)
        l.backward()
        if constraint == 'l_inf':
            step = input - stepsize * input.grad.sign()
            input = (base + li_projection(step, base, eps)).detach()
        elif constraint == 'l_2':
            # stepsize = 2.5 * eps / 100
            step = input - stepsize * input.grad
            input = (base + l2_projection(step, base, eps)).detach()

    return torch.clamp(invertTransform(input.view(3, 32, 32)), min=0, max=1)

def fgsm(inputs, labels, model, eps):
    inputs.requires_grad = True
    loss = torch.nn.CrossEntropyLoss()
    out = model(inputs)
    l = loss(out, labels)
    l.backward()
    step = inputs + eps * inputs.grad.sign()
    return step.detach()
