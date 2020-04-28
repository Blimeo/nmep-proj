import torch


def l2_projection(step, start, eps):
    if torch.norm(x_prime) <= eps:
        return step - start

    return ((step - start) / torch.norm(step - start)) * eps


def pgd(input, model, loss, alpha, eps, steps):

    x_prime = input.clone()

    for _ in range(steps):
        loss = model.forward(input)
        grad = loss.backward()
        x_prime += alpha * torch.sign(grad)
        x_prime = start + l2_projection(x_prime, input)

    return x_prime
