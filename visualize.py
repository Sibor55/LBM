import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import glob

def main():
    # Получаем список всех сохраненных файлов
    files = sorted(glob.glob('curl_data/curl_frame_*.npy'))
    
    if not files:
        print("Файлы данных не найдены! Сначала запустите simulation.py")
        return
    
    # Загружаем все данные в память
    frames = [np.load(file) for file in files]
    
    # Создаем фигуру для анимации
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(frames[0], cmap='bwr', vmin=-0.1, vmax=0.1)
    fig.colorbar(img, ax=ax, label='Завихренность')
    ax.set_title("Визуализация завихренности потока")
    ax.set_xlabel("X координата")
    ax.set_ylabel("Y координата")
    
    def update(frame):
        """Обновление кадра анимации"""
        img.set_data(frames[frame])
        ax.set_title("Завихренность потока")
        return img,
    
    # Создаем анимацию
    ani = FuncAnimation(
        fig, 
        update, 
        frames=len(frames),
        interval=100,  # Интервал между кадрами в мс
        blit=True
    )
    
    plt.tight_layout()
    plt.show()
    
    # Опционально: сохранение анимации в файл (требует ffmpeg)
    # ani.save('flow_animation.mp4', writer='ffmpeg', fps=10, dpi=200)

if __name__ == "__main__":
    main()