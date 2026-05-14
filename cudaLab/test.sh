#!/bin/bash

#SBATCH --job-name=hough_test
#SBATCH --output=results.csv
#SBATCH --error=experiment_errors.log
#SBATCH --partition=tornado-k40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

# Очистка и загрузка модулей
module purge
module load mpi/openmpi/3.0.3/cuda/8.0/gcc/5

# Переходим в директорию проекта через переменную окружения
cd $SLURM_SUBMIT_DIR

# Компиляция: 
# -std=c++11 для работы векторов и инициализации {...}
# -arch=sm_35 специально для Tesla K40
# -lm для математических функций (sqrt, round)
nvcc -O3 -std=c++11 -arch=sm_35 benchmark.cu -o benchmark_exe -lm -Wno-deprecated-gpu-targets

# Запуск с проверкой компиляции
if [ -f ./benchmark_exe ]; then
    echo "Запуск эксперимента..."
    ./benchmark_exe
else
    echo "Ошибка: Файл benchmark_exe не был создан." >&2
    exit 1
fi
