import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import glob
import argparse


def main():
    # Настройка парсера аргументов
    parser = argparse.ArgumentParser(description="Визуализация завихренности потока")
    parser.add_argument(
        "--input_dir", default="/content/curl_data", help="Папка с файлами .npy"
    )
    parser.add_argument(
        "--output_path",
        default="/content/flow_animation.mp4",
        help="Путь для сохранения видео",
    )
    args = parser.parse_args()

    # Увеличиваем лимит встраивания анимации до 100 МБ
    plt.rcParams["animation.embed_limit"] = 100  # В мегабайтах

    # Поиск файлов в указанной директории
    files = sorted(glob.glob(os.path.join(args.input_dir, "curl_frame_*.npy")))
    if not files:
        print(
            f"Файлы данных не найдены в {args.input_dir}! Сначала запустите simulation.py"
        )
        return

    # Оптимизированная загрузка кадров через генератор
    frames = (np.load(file) for file in files)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(
        next(frames),  # Первый кадр для инициализации
        cmap="bwr",
        vmin=-0.1,
        vmax=0.1,
        origin="lower",
    )

    fig.colorbar(img, ax=ax, label="Ротор")
    ax.set_title("Визуализация завихренности потока")
    ax.set_xlabel("X координата")
    ax.set_ylabel("Y координата")

    def update(frame):
        img.set_data(frame)
        return (img,)

    # Создаем анимацию с использованием генератора
    ani = FuncAnimation(
        fig,
        update,
        frames=frames,  # Используем генератор напрямую
        interval=100,
        blit=True,
        save_count=len(files),  # Явно указываем количество кадров
    )

    # Сохраняем анимацию в файл
    print(f"Сохранение анимации в {args.output_path}...")
    ani.save(
        args.output_path,
        writer="ffmpeg",
        fps=10,
        dpi=200,
        progress_callback=lambda i, n: print(f"\rПрогресс: {i + 1}/{n} кадров", end=""),
    )
    print("\nСохранение завершено!")

    # Показываем анимацию
    plt.show()


if __name__ == "__main__":
    main()
