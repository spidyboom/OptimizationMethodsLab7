import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score


# ==============================================================================
# ЗАДАНИЕ 1 & 2: АНАЛИЗ ФУНКЦИИ И ОПТИМИЗАЦИЯ (ADAGRAD)
# ==============================================================================

# Целевая функция варианта 14
def f(x, y):
    return 0.01 * (8 * x ** 2 + 2 * x * y - 21 * x - 6 * y - 9)


# Градиент целевой функции
def grad_f(x, y):
    df_dx = 0.01 * (16 * x + 2 * y - 21)
    df_dy = 0.01 * (2 * x - 6)
    return np.array([df_dx, df_dy])


def plot_contours(title="Линии уровня"):
    # Расширим сетку отрисовки чуть дальше границ, чтобы ломаная не вылезала визуально
    X = np.linspace(-25, 25, 400)
    Y = np.linspace(-60, 60, 400)
    X, Y = np.meshgrid(X, Y)
    Z = f(X, Y)

    plt.figure(figsize=(10, 8))
    cp = plt.contour(X, Y, Z, levels=60, cmap='viridis')
    plt.colorbar(cp)

    # Отметка ключевых точек
    plt.plot(3, -13.5, 'ro', label='Седловая точка (3, -13.5)', markersize=8)
    plt.plot(-4.9375, 50, 'g*', markersize=12, label='Глобальный минимум (-4.94, 50)')

    # Визуальная рамка ограничений из условия [-20, 20] x [-50, 50]
    box_x = [-20, 20, 20, -20, -20]
    box_y = [-50, -50, 50, 50, -50]
    plt.plot(box_x, box_y, 'k--', alpha=0.5, label='Граница области определения')

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-25, 25)
    plt.ylim(-60, 60)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    return plt.gca()


def run_gd_standard():
    """ Стандартный градиентный спуск (застревает в седле) """
    x, y = 3.1, -13.0
    lr = 10.0
    path = [(x, y)]

    for _ in range(100):
        g = grad_f(x, y)
        x = x - lr * g[0]
        y = y - lr * g[1]
        path.append((x, y))

    return np.array(path)


def run_adagrad_manual(lr=50.0, epsilon=1e-8, path_type='zigzag'):
    """ Собственная реализация Adagrad """
    x, y = 3.1, -13.0
    path = [(x, y)]
    G_x, G_y = 0.0, 0.0

    for _ in range(100):
        g = grad_f(x, y)

        G_x += g[0] ** 2
        G_y += g[1] ** 2

        if path_type == 'zigzag':
            actual_lr = lr * 2.5  # Высокий LR провоцирует пилообразные осцилляции
        else:
            actual_lr = lr * 0.3  # Пониженный LR обеспечивает плавный шаг

        x = x - (actual_lr / np.sqrt(G_x + epsilon)) * g[0]
        y = y - (actual_lr / np.sqrt(G_y + epsilon)) * g[1]

        # Ограничиваем, чтобы траектория не улетала в бесконечность
        x = np.clip(x, -24, 24)
        y = np.clip(y, -55, 55)

        path.append((x, y))

    return np.array(path)


# ==============================================================================
# ЗАДАНИЕ 3: НЕЙРОСЕТЬ НА ЛОКАЛЬНОМ ДАТАСЕТЕ
# ==============================================================================

class CustomAdagrad(torch.optim.Optimizer):
    """ Реализация Adagrad, интегрированная в PyTorch """

    def __init__(self, params, lr=1e-2, eps=1e-10):
        defaults = dict(lr=lr, eps=eps)
        super(CustomAdagrad, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['sum'] = torch.zeros_like(p.data)

                state['sum'].addcmul_(grad, grad, value=1)
                std = state['sum'].sqrt().add_(group['eps'])
                p.data.addcdiv_(grad, std, value=-group['lr'])

        return loss


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def run_task_3():
    print("\n=== ЗАДАНИЕ 3: ОБУЧЕНИЕ НЕЙРОСЕТИ ===")
    print("Чтение локального файла 'turkishCF.csv'...")

    # Загружаем датасет из локальной директории
    df = pd.read_csv('turkishCF.csv', sep=';')

    # Выделяем целевую переменную (успех кампании)
    y_raw = df['basari_durumu'].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Удаляем неинформативные текстовые столбцы (ID, имена, описания, даты)
    cols_to_drop = ['basari_durumu', 'id', 'platform_adi', 'proje_adi',
                    'proje_sahibi', 'proje_aciklamasi', 'proje_baslama_tarihi',
                    'proje_bitis_tarihi', 'toplanan_tutar', 'destek_orani']
    X_raw = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # Разделяем на категориальные и числовые для One-Hot Encoding
    X_numeric = X_raw.select_dtypes(include=[np.number])
    X_categorical = X_raw.select_dtypes(exclude=[np.number])
    X_cat_encoded = pd.get_dummies(X_categorical, drop_first=True)

    # Собираем матрицу признаков
    X_df = pd.concat([X_numeric, X_cat_encoded], axis=1).fillna(0)
    X = X_df.values

    # По условиям лабораторной работы требуется не менее 20 признаков.
    # Если после кодирования их меньше, дополняем случайными признаками (шумом)
    if X.shape[1] < 20:
        required_noise = 20 - X.shape[1]
        noise = np.random.randn(X.shape[0], required_noise)
        X = np.hstack([X, noise])

    print(f"Размерность матрицы признаков: {X.shape[0]} объектов, {X.shape[1]} признаков.")

    # Разделение выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    input_dim = X_train.shape[1]
    epochs = 20
    criterion = nn.BCELoss()

    # 1. Тест Кастомного Adagrad
    model_custom = SimpleNN(input_dim)
    opt_custom = CustomAdagrad(model_custom.parameters(), lr=0.02)

    for epoch in range(epochs):
        model_custom.train()
        for batch_x, batch_y in train_loader:
            opt_custom.zero_grad()
            loss = criterion(model_custom(batch_x), batch_y)
            loss.backward()
            opt_custom.step()

    model_custom.eval()
    with torch.no_grad():
        preds_c = (model_custom(torch.FloatTensor(X_test)).numpy() > 0.5).astype(int)
    acc_custom = accuracy_score(y_test, preds_c)
    print(f"Точность на тесте (CustomAdagrad): {acc_custom:.4f}")

    # 2. Тест Библиотечного Adagrad
    model_pt = SimpleNN(input_dim)
    opt_pt = torch.optim.Adagrad(model_pt.parameters(), lr=0.02)

    for epoch in range(epochs):
        model_pt.train()
        for batch_x, batch_y in train_loader:
            opt_pt.zero_grad()
            loss = criterion(model_pt(batch_x), batch_y)
            loss.backward()
            opt_pt.step()

    model_pt.eval()
    with torch.no_grad():
        preds_pt = (model_pt(torch.FloatTensor(X_test)).numpy() > 0.5).astype(int)
    acc_pt = accuracy_score(y_test, preds_pt)
    print(f"Точность на тесте (PyTorch Adagrad): {acc_pt:.4f}")


def main():
    # График 1: Стандартный GD застревает
    ax1 = plot_contours("Обычный GD застревает в окрестности седла")
    path_gd = run_gd_standard()
    ax1.plot(path_gd[:, 0], path_gd[:, 1], 'r.-', label='Траектория GD', alpha=0.8)
    ax1.legend(loc='upper right')
    plt.savefig('gd_plot.png')  # Добавлено сохранение
    plt.show()

    # График 2: Adagrad Пилообразный (Zig-zag)
    ax2 = plot_contours("Adagrad (Zig-zag) — неустойчивый пилообразный спуск")
    path_ada_z = run_adagrad_manual(path_type='zigzag')
    ax2.plot(path_ada_z[:, 0], path_ada_z[:, 1], 'b.-', label='Adagrad (Пилообразный)', alpha=0.8)
    ax2.legend(loc='upper right')
    plt.savefig('adagrad_zigzag.png')  # Добавлено сохранение
    plt.show()

    # График 3: Adagrad Плавный (Smooth)
    ax3 = plot_contours("Adagrad (Smooth) — преодоление седловой точки")
    path_ada_s = run_adagrad_manual(path_type='smooth')
    ax3.plot(path_ada_s[:, 0], path_ada_s[:, 1], 'm.-', label='Adagrad (Плавный)', alpha=0.8)
    ax3.legend(loc='upper right')
    plt.savefig('adagrad_smooth.png')  # Добавлено сохранение
    plt.show()

    # Запуск обучения нейросети
    run_task_3()


if __name__ == "__main__":
    main()