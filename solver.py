import numpy as np
import matplotlib.pyplot as plt

# Конфигурационные параметры
PLOT_EVERY = 10  # Частота визуализации
SIMULATION_STEPS = 3000  # Количество шагов симуляции

# Параметры решетки
GRID_WIDTH = 400    # Ширина расчетной области
GRID_HEIGHT = 100   # Высота расчетной области
RELAXATION_TIME = 0.53  # Время релаксации

# Параметры цилиндра
CYLINDER_RADIUS = 13
CYLINDER_CENTER_X = GRID_WIDTH // 4
CYLINDER_CENTER_Y = GRID_HEIGHT // 2

def calculate_distance(x1, y1, x2, y2):
    """Вычисляет евклидово расстояние между двумя точками"""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def main():
    # Инициализация модели D2Q9
    num_directions = 9
    direction_x = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])  # Компоненты скорости по X
    direction_y = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])  # Компоненты скорости по Y
    weights = np.array([  # Весовые коэффициенты
        4/9,  # Нулевая скорость
        1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36  # Остальные направления
    ])
    
    # Инициализация распределений
    particle_dist = np.ones((GRID_HEIGHT, GRID_WIDTH, num_directions)) + 0.01 * np.random.randn(
        GRID_HEIGHT, GRID_WIDTH, num_directions)
    particle_dist[:, :, 3] = 2.3  # Начальное условие для скорости
    
    # Создание маски для цилиндра
    x_coords, y_coords = np.meshgrid(np.arange(GRID_WIDTH), np.arange(GRID_HEIGHT))
    cylinder_mask = calculate_distance(CYLINDER_CENTER_X, CYLINDER_CENTER_Y, 
                                      x_coords, y_coords) < CYLINDER_RADIUS

    # Основной цикл симуляции
    for step in range(SIMULATION_STEPS):
        # Фаза переноса (streaming)
        for i, (dx, dy) in enumerate(zip(direction_x, direction_y)):
            particle_dist[:, :, i] = np.roll(
                particle_dist[:, :, i],
                shift=(dx, dy),
                axis=(1, 0)
            )
        
        # Обработка граничных условий на цилиндре (bounce-back)
        boundary_dist = particle_dist[cylinder_mask, :]
        particle_dist[cylinder_mask, :] = boundary_dist[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]
        
        # Расчет макроскопических величин
        density = np.sum(particle_dist, axis=2)
        velocity_x = np.sum(particle_dist * direction_x, axis=2) / density
        velocity_y = np.sum(particle_dist * direction_y, axis=2) / density
        
        # Обнуление скорости внутри цилиндра
        velocity_x[cylinder_mask] = 0
        velocity_y[cylinder_mask] = 0
        
        # Фаза столкновений (collision)
        equilibrium = np.zeros_like(particle_dist)
        for i, (dx, dy, w) in enumerate(zip(direction_x, direction_y, weights)):
            dot_product = dx * velocity_x + dy * velocity_y
            velocity_sq = velocity_x**2 + velocity_y**2
            equilibrium[:, :, i] = density * w * (
                1 + 
                3 * dot_product + 
                9/2 * dot_product**2 - 
                3/2 * velocity_sq
            )
        
        # Обновление распределения
        particle_dist += -(1.0 / RELAXATION_TIME) * (particle_dist - equilibrium)
        
        # Визуализация
        if step % PLOT_EVERY == 0:
            plt.imshow(np.sqrt(velocity_x**2 + velocity_y**2), 
                      cmap='jet', 
                      vmin=0, 
                      vmax=0.15)
            plt.title(f"Шаг симуляции: {step}")
            plt.pause(0.001)
            plt.cla()

if __name__ == "__main__":
    main()