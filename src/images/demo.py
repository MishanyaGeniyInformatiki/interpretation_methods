import torch
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

from images.utils.load_data import load_data
from images.utils.models import Net
from images.utils.integrated_gradients import IntegratedGradients
from images.utils.utils import draw_input_image, show_mask_my, show_mask_on_image_my, save_explanation
from images.utils.grad_cam import GradCam

args = {}
kwargs = {}
args['batch_size'] = 1000
args['test_batch_size'] = 1000
args['epochs'] = 10
args['lr'] = 0.01
args['momentum'] = 0.5

args['seed'] = 1
args['log_interval'] = 10
args['cuda'] = False


def train():
    for epoch in range(1, args['epochs'] + 1):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.data
        print(f'Train Epoch: {epoch}, Loss: {total_loss}')


def test():
    model.eval()
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().to(device).sum()

    acc = 100. * correct.item() / len(test_loader.dataset)
    print("Test accuracy: ", round(acc, 2))


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_xy, test_xy = load_data()

    train_loader = torch.utils.data.DataLoader(train_xy, batch_size=args['batch_size'], shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_xy, batch_size=args['batch_size'], shuffle=True, **kwargs)

    model = Net()
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])

    train()
    test()

    # Берем изображение из тестового набора данных
    image_tensor = test_xy[30][0]
    image_label = test_xy[30][1].item()

    with torch.no_grad():
        probs = torch.squeeze(F.softmax(model(image_tensor.to(device)), dim=1))
    pred = torch.argmax(probs).item()  # предсказание модели
    prob = probs[pred].item()  # вероятность

    plt.imshow(image_tensor.numpy()[0], cmap='gray')
    plt.show()
    print(f' Класс картинки: {image_label}\n Предсказание модели: {pred}\n Вероятность: {prob}')

    # Explanation IntegratedGradients
    integrated_gradients = IntegratedGradients(model)
    mask = integrated_gradients.get_mask(image_tensor=image_tensor.unsqueeze(0).to(device))

    explanation_name = "explanation_ig"
    explanation_path = "../images/explanation_results/" + explanation_name

    save_explanation(mask, explanation_path)

    with open(explanation_path, 'rb') as file:
        mask_ig = np.load(file, allow_pickle=True)
    file.close()

    figure, axes = plt.subplots(1, 3, figsize=(12, 12), tight_layout=True)

    draw_input_image(image_tensor.numpy()[0], title="Input image", axis=axes[0])
    show_mask_my(mask_ig, title="IntegratedGradients mask", axis=axes[1])
    show_mask_on_image_my(image_tensor, mask_ig, title="Mask on image", k=0.7, axis=axes[2])

    # HEATMAP IntegratedGradients
    # приведем диапазон значений маски к [0, 1]
    normalizedMaskIntegratedGradients = (mask_ig - np.min(mask_ig)) / (np.max(mask_ig) - np.min(mask_ig))
    roundNormalizedMaskIntegratedGradients = np.squeeze(np.around(normalizedMaskIntegratedGradients, 2), axis=2)

    plt.figure(figsize=(15, 15))
    ax = plt.axes()
    sns.heatmap(roundNormalizedMaskIntegratedGradients, annot=True, ax=ax)
    ax.set_title("Heatmap IntegratedGradients mask")
    plt.show()

    # Explanation GradCam
    grad_cam = GradCam(model)
    mask = grad_cam.get_mask(image_tensor=image_tensor.unsqueeze(0).to(device))
    grad_cam.remove_hooks()

    explanation_name = "explanation_gc"
    explanation_path = "../images/explanation_results/" + explanation_name

    save_explanation(mask, explanation_path)

    with open(explanation_path, 'rb') as file:
        mask_grad_cam = np.load(file, allow_pickle=True)
    file.close()

    figure, axes = plt.subplots(1, 3, figsize=(12, 12), tight_layout=True)

    mask_grad_cam_unsq = np.expand_dims(mask_grad_cam, axis=2)

    draw_input_image(image_tensor.numpy()[0], title="Input image", axis=axes[0])
    show_mask_my(mask_grad_cam_unsq, title="GradCAM mask", axis=axes[1])
    show_mask_on_image_my(image_tensor, mask_grad_cam_unsq, title="Mask on image", k=0.7, axis=axes[2])
    plt.show()

    # HEATMAP GradCam
    # приведем диапазон значений маски к [0, 1]
    normalizedMaskGradCAM = (mask_grad_cam - np.min(mask_grad_cam)) / (np.max(mask_grad_cam) - np.min(mask_grad_cam))
    roundNormalizedMaskGradCAM = np.around(normalizedMaskGradCAM, 2)

    plt.figure(figsize=(15, 15))
    sns.heatmap(roundNormalizedMaskGradCAM, annot=True)
    plt.show()
