import os
import argparse


def main(
    nx=800,
    ny=500,
    tau=0.53,
    Nt=13000,
    save_every=30,
    cx=None,
    cy=None,
    radius=None,
    use_gpu=True,
    output_dir="curl_data",
):
    """
    LBM (Lattice Boltzmann Method) симуляция потока жидкости с препятствием

    Параметры:
    nx, ny     -- размеры расчетной сетки
    tau        -- время релаксации (вязкость)
    Nt         -- количество итераций
    save_every -- частота сохранения завихренности
    cx, cy     -- координаты центра цилиндра
    radius     -- радиус цилиндра
    use_gpu    -- использовать GPU (CuPy) или CPU (NumPy)
    output_dir -- папка для сохранения результатов
    """

    # =========================================================================
    # Инициализация вычислительных библиотек (GPU/CPU)
    # =========================================================================
    if use_gpu:
        import cupy as xp
        from cupy import asnumpy

        print("Режим вычислений: GPU (CuPy)")
    else:
        import numpy as xp
        from numpy import asarray as asnumpy

        print("Режим вычислений: CPU (NumPy)")

    # Создаем папку для сохранения данных
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Параметры цилиндра по умолчанию
    if cx is None:
        cx = nx // 4
    if cy is None:
        cy = ny // 2
    if radius is None:
        radius = ny // 7

    # =========================================================================
    # СХЕМА D2Q9 (2D, 9 скоростей)
    # =========================================================================
    """
    D2Q9 - решетчатая схема с 9 направлениями скоростей в 2D пространстве
    
    Индексы и направления:
      6  2  5
        \ | /
      3 - 0 - 1
        / | \
      7  4  8
    
    Скорости:
      i |  cx | cy |  вес  | направление
      --|-----|----|-------|------------
      0 |  0  |  0 | 4/9   | (0, 0)    покой
      1 |  1  |  0 | 1/9   | → °
      2 |  0  |  1 | 1/9   | ↑ 
      3 | -1  |  0 | 1/9   | ← 
      4 |  0  | -1 | 1/9   | ↓ 
      5 |  1  |  1 | 1/36  | ↗ 
      6 | -1  |  1 | 1/36  | ↖ 
      7 | -1  | -1 | 1/36  | ↙ 
      8 |  1  | -1 | 1/36  | ↘ 
    
    Равновесное распределение:
      f_i^{eq} = rho * w_i * [1 + 3*(c_i · u) + 9/2*(c_i · u)^2 - 3/2*u^2]
      где:
        rho - плотность жидкости
        w_i - вес для направления i
        c_i = (c_{ix}, c_{iy}) - вектор скорости
        u = (u_x, u_y) - макроскопическая скорость
        (c_i · u) = c_{ix}*u_x + c_{iy}*u_y - скалярное произведение
    """
    NL = 9  # Количество направлений

    # Векторы скоростей для каждого направления
    cxs = xp.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
    cys = xp.array([0, 0, 1, 0, -1, 1, 1, -1, -1])

    # Весовые коэффициенты
    weights = xp.array(
        [
            4 / 9,  # i=0: центр
            1 / 9,  # i=1: вправо
            1 / 9,  # i=2: вверх
            1 / 9,  # i=3: влево
            1 / 9,  # i=4: вниз
            1 / 36,  # i=5: вверх-вправо
            1 / 36,  # i=6: вверх-влево
            1 / 36,  # i=7: вниз-влево
            1 / 36,  # i=8: вниз-вправо
        ]
    )

    # Индексы для отскока от препятствий (bounce-back)
    bounce_indices = xp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

    # =========================================================================
    # ИНИЦИАЛИЗАЦИЯ ПЕРЕМЕННЫХ
    # =========================================================================
    # Функция распределения
    F = xp.ones((ny, nx, NL)) + 0.01 * xp.random.randn(ny, nx, NL)
    F[:, :, 1] = 2.3  # Начальный поток вправо

    # Создание препятствия (цилиндра)
    X, Y = xp.meshgrid(xp.arange(nx), xp.arange(ny))
    cylinder = (X - cx) ** 2 + (Y - cy) ** 2 < radius**2

    frame_count = 0  # Счетчик кадров

    # =========================================================================
    # ГЛАВНЫЙ ЦИКЛ СИМУЛЯЦИИ
    # =========================================================================
    for it in range(Nt):
        # Вывод прогресса
        if it % 100 == 0:
            print(f"Итерация: {it}/{Nt}")

        # Граничные условия (фиксированное давление)
        F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]  # Правая граница
        F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]  # Левая граница

        # -------------------------------------
        # Фаза переноса (streaming)
        # -------------------------------------
        for i in range(NL):
            F[:, :, i] = xp.roll(F[:, :, i], shift=int(cxs[i]), axis=1)
            F[:, :, i] = xp.roll(F[:, :, i], shift=int(cys[i]), axis=0)

        # -------------------------------------
        # Расчет макроскопических величин
        # -------------------------------------
        rho = xp.sum(F, axis=2)  # Плотность
        ux = xp.sum(F * cxs, axis=2) / rho  # Скорость по x
        uy = xp.sum(F * cys, axis=2) / rho  # Скорость по y

        # Обработка столкновений с препятствием
        F[cylinder] = F[cylinder][:, bounce_indices]
        ux[cylinder] = 0
        uy[cylinder] = 0

        # -------------------------------------
        # Расчет равновесного распределения
        # -------------------------------------
        Feq = xp.zeros_like(F)
        for i in range(NL):
            # Скалярное произведение (c_i · u)
            cu = cxs[i] * ux + cys[i] * uy

            # Квадрат макроскопической скорости (u · u)
            u_sq = ux**2 + uy**2

            # Формула равновесного распределения:
            # f_i^{eq} = ρ * w_i * [1 + 3*(c_i·u) + 4.5*(c_i·u)^2 - 1.5*u^2]
            Feq[:, :, i] = rho * weights[i] * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u_sq)

        # -------------------------------------
        # Фаза столкновений (collision)
        # -------------------------------------
        F += -(1 / tau) * (F - Feq)

        # -------------------------------------
        # Сохранение завихренности
        # -------------------------------------
        if it % save_every == 0:
            # Расчет ротора: ∂u_y/∂x - ∂u_x/∂y
            dfydx = ux[2:, 1:-1] - ux[:-2, 1:-1]  # ∂u_x/∂y
            dfxdy = uy[1:-1, 2:] - uy[1:-1, :-2]  # ∂u_y/∂x
            curl = dfxdy - dfydx

            # Сохранение в файл
            import numpy as np

            curl_np = asnumpy(curl)
            np.save(f"{output_dir}/curl_frame_{frame_count:04d}.npy", curl_np)
            frame_count += 1

    print(f"Симуляция завершена. Сохранено {frame_count} кадров.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Симуляция сетки Больцмана"
    )
    parser.add_argument("--nx", type=int, default=800, help="Ширина сетки")
    parser.add_argument("--ny", type=int, default=500, help="Высота сетки")
    parser.add_argument("--tau", type=float, default=0.53, help="Время релаксации")
    parser.add_argument("--Nt", type=int, default=13000, help="Число итераций")
    parser.add_argument(
        "--save_every", type=int, default=30, help="Частота сохранения кадров"
    )
    parser.add_argument(
        "--cx", type=int, default=None, help="X-координата центра цилиндра"
    )
    parser.add_argument(
        "--cy", type=int, default=None, help="Y-координата центра цилиндра"
    )
    parser.add_argument("--radius", type=int, default=None, help="Радиус цилиндра")
    parser.add_argument(
        "--cpu", action="store_true", help="Использовать CPU вместо GPU"
    )
    parser.add_argument(
        "--output_dir", default="curl_data", help="Папка для сохранения данных"
    )

    args = parser.parse_args()

    main(
        nx=args.nx,
        ny=args.ny,
        tau=args.tau,
        Nt=args.Nt,
        save_every=args.save_every,
        cx=args.cx,
        cy=args.cy,
        radius=args.radius,
        use_gpu=not args.cpu,
        output_dir=args.output_dir,
    )
