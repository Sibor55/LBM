# Lattice Boltzmann Method (LBM) Flow Simulation

This project implements a 2D fluid flow simulation using the Lattice Boltzmann Method (D2Q9 scheme) with an obstacle (cylinder). The simulation generates vorticity data which can be visualized as an animation.

## Requirements
- Python 3.x
- NumPy
- Matplotlib
- (Optional) CuPy for GPU acceleration

## Usage

### 1. Run the simulation
```bash
python solver.py [options]
```

Key options:
- `--nx`, `--ny`: Grid dimensions (default: 800x500)
- `--tau`: Relaxation time (viscosity, default: 0.53. WARNING: Keep this number >0.5, otherwise the model will go unstable and you will waste a lot of time(due to overflows)) 
- `--Nt`: Number of iterations (default: 13000)
- `--save_every`: Save frame interval (default: 30)
- `--cpu`: Use CPU instead of GPU
- `--output_dir`: Output directory for data files

Example:
```bash
python solver.py --nx 400 --ny 250 --tau 0.6 --Nt 5000 --save_every 20
```

### 2. Visualize the results
```bash
python visualization.py --input_dir [output_dir] --output_path [animation.mp4]
```

Example:
```bash
python visualization.py --input_dir curl_data --output_path flow.mp4
```

## References
Based on:
- [Create Your Own Lattice Boltzmann Simulation (With Python)](https://medium.com/swlh/create-your-own-lattice-boltzmann-simulation-with-python-8759e8b53b1c)
- [A Practical Introduction to the Lattice Boltzmann Method](https://www.ndsu.edu/fileadmin/physics.ndsu.edu/Wagner/LBbook.pdf)


## Требования
- Python 3.x
- NumPy
- Matplotlib
- (Опционально) CuPy для ускорения на GPU

## Использование

### 1. Запуск симуляции
```bash
python solver.py [параметры]
```

Основные параметры:
- `--nx`, `--ny`: Размеры сетки (по умолчанию: 800x500)
- `--tau`: Время релаксации (вязкость, по умолчанию: 0.53. ВНИМАНИЕ: значение должно быть >0.5, иначе модель станет нестабильной)
- `--Nt`: Количество итераций (по умолчанию: 13000)
- `--save_every`: Частота сохранения кадров (по умолчанию: 30)
- `--cpu`: Использовать CPU вместо GPU
- `--output_dir`: Папка для сохранения данных

Пример:
```bash
python solver.py --nx 400 --ny 250 --tau 0.6 --Nt 5000 --save_every 20
```

### 2. Визуализация результатов
```bash
python visualization.py --input_dir [папка_с_данными] --output_path [анимация.mp4]
```

Пример:
```bash
python visualization.py --input_dir curl_data --output_path flow.mp4
```
