import subprocess
import os
import sys

def compile_and_run_cuda(cu_file, output_bin="matrix_mul", gpu_arch="sm_90"):
    """
    Компилирует и запускает CUDA-файл с умножением матриц.
    
    Параметры:
        cu_file (str): путь к файлу .cu
        output_bin (str): имя выходного исполняемого файла
        gpu_arch (str): архитектура GPU
    """
    if not os.path.exists(cu_file):
        print(f"Ошибка: файл {cu_file} не найден!")
        return

    # компиляция 
    compile_cmd = [
        "nvcc",
        cu_file,
        "-o", output_bin,
        f"-arch={gpu_arch}",
        "-O3",             
        "--std=c++14",
    ]
    
    print(f"Компиляция: {' '.join(compile_cmd)}")
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Ошибка компиляции:")
        print(result.stderr)
        return
    
    # запуск программы
    subprocess.run([f"./{output_bin}"])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python run_cuda.py <файл.cu> [output_bin]")
        sys.exit(1)
    
    cu_file = sys.argv[1]
    output_bin = sys.argv[2] if len(sys.argv) > 2 else "matrix_mul"
    
    compile_and_run_cuda(cu_file, output_bin)
