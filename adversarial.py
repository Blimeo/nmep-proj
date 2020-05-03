import torch
import torchvision.transforms as transforms


def l2_projection(step, start, eps):
    dif = step - start

    dif = dif.view(-1, 32*32*3)
    n = torch.sum(dif ** 2, dim=1).view(-1, 1) # l2 norm
    # n = torch.norm(dif, dim=1).view(-1, 1)
    out = torch.where(n <= eps, dif, (dif / n) * eps)
    return out.view(-1, 3, 32, 32)

def li_projection(step, start, eps):
    return torch.clamp(step - start, min=-eps, max=eps)


def pgd(inputs, labels, model, stepsize, eps, steps, constraint='l_inf'):
    loss = torch.nn.CrossEntropyLoss()
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
            stepsize = 2.5 * eps / 100
            step = inputs + stepsize * inputs.grad
            inputs = (base + l2_projection(step, base, eps)).detach()

    return inputs

def pgd_attack(input, label, model, stepsize, eps, steps, constraint='l_inf'):
    invertTransform = transforms.Compose([
                                transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.2023, 1/0.1994, 1/0.2010 ]),
                                transforms.Normalize(mean = [ -0.4914, -0.4822, -0.4465 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    input = invertTransform(input).view(1, 3, 32, 32)
    loss = torch.nn.CrossEntropyLoss()
    base = input.clone().detach()
    label = torch.Tensor([label]).cuda().long()
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
            stepsize = 2.5 * eps / 100
            step = input - stepsize * input.grad
            input = (base + l2_projection(step, base, eps)).detach()

    transform = transforms.Compose([
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return transform(input.view(3, 32, 32))

def fgsm(inputs, labels, model, eps):
    inputs.requires_grad = True
    loss = torch.nn.CrossEntropyLoss()
    out = model(inputs)
    l = loss(out, labels)
    l.backward()
    step = inputs + eps * inputs.grad.sign()
    return step.detach()
