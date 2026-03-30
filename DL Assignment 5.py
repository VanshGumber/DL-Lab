import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dir = "Surface/train/images"
test_dir  = "Surface/validation/images"

transform = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train = datasets.ImageFolder(train_dir, transform=transform)
test  = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train, batch_size=32, shuffle=True)
test_loader  = DataLoader(test, batch_size=32, shuffle=False)

def run(model, name):
    print("\n", name)

    for param in model.parameters():
        param.requires_grad = False

    if name == "resnet":
        model.fc = nn.Linear(model.fc.in_features, 6)
    elif name == "efficientnet":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 6)
    elif name == "inception":
        model.fc = nn.Linear(model.fc.in_features, 6)

    model = model.to(device)

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for e in range(10):
        total_loss = 0
        total = 0
        model.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            out = model(x)

            if name == "inception" and isinstance(out, tuple):
                out, aux = out
                loss = loss_fn(out, y) + 0.4 * loss_fn(aux, y)
            else:
                loss = loss_fn(out, y)

            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

        print("epoch", e+1, "loss", total_loss / total)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)

            if isinstance(out, tuple):
                out = out[0]

            pred = out.argmax(1)
            correct += (pred.cpu() == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print("acc:", acc)
    return acc


r = models.resnet50(pretrained=True)
e = models.efficientnet_b0(pretrained=True)
i = models.inception_v3(pretrained=True)


results = {}

results["ResNet50"] = run(r, "resnet")
results["EfficientNetB0"] = run(e, "efficientnet")
results["InceptionV3"] = run(i, "inception")

print("\nFinal Results:")
for k, v in results.items():
    print(k, ":", v)

best = max(results, key=results.get)
print("\nBest Model:", best)
