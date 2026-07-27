<div align="center">
  <img src="docs/images/prism_logo.svg" alt="PRISM Logo" width="200">

  <h1>PRISM</h1>

  <p><b>Parallel RF Instructions for Signal Manipulation</b></p>
  <p>Biblioteca de computación C++/Halide para comunicaciones inalámbricas y procesamiento de señales</p>

  <p>
    <img src="https://img.shields.io/badge/C%2B%2B-17-blue.svg" alt="C++17">
    <img src="https://img.shields.io/badge/Platform-macOS%20|%20Linux%20|%20Windows-lightgrey.svg" alt="Platform">
  </p>
</div>

---

PRISM es una biblioteca C++17 orientada a las comunicaciones inalámbricas y el procesamiento digital de señales (DSP). Construye grafos de computación perezosa (lazy evaluation) a través de un DSL, y utiliza el JIT/AOT de Halide junto con backends de FFT de proveedores específicos en tiempo de ejecución para lograr una ejecución de alto rendimiento multiplataforma. Los operadores convencionales son programados uniformemente por Halide, mientras que los nodos Anchor de FFT/IFFT pueden utilizar vDSP / cuFFT / hipFFT / vkFFT.

## Características Destacadas

- **DSL Fluido**: Descripción de la cadena de procesamiento mediante combinaciones de `Signal` + operadores; no bloqueante y sin cálculo inmediato.
- **Pipeline de Anclaje (Anchor)**: Las operaciones FFT/IFFT se fuerzan a través de las API de proveedores para garantizar rendimiento y estabilidad.
- **Backends Intercambiables**: Detección automática o especificación manual de vDSP / cuFFT / hipFFT / vkFFT.
- **Construcción Multiplataforma**: Basado en CMake + C++17, con dependencias sencillas.
- **Conjunto de Herramientas de Simulación**: Fuentes aleatorias, modelos de canal y de ruido listos para usar.

## Entorno y Dependencias

Requeridos:

- CMake ≥ 3.20
- Compilador C++17 (se recomienda clang/clang++)
- [Halide](https://halide-lang.org/) (`find_package(Halide REQUIRED)`)

Opcionales (habilitar según necesidad):

- macOS Accelerate/vDSP
- NVIDIA CUDA Toolkit (cuFFT)
- AMD ROCm / hipFFT
- Código fuente de vkFFT (submódulo `external/vkfft`, compatible con Metal/CUDA/HIP/OpenCL)
- tomlplusplus (submódulo `external/tomlplusplus`, para análisis de configuración de ejemplos)
- Graphviz (para generar diagramas de documentación)
- Doxygen (para generar documentación de la API)

> Interruptores de backend: en `cmake/local.cmake`, configure `PRISM_USE_VDSP/PRISM_USE_CUFFT/PRISM_USE_HIPFFT/PRISM_USE_VKFFT` como `AUTO/ON/OFF`.

## Inicio Rápido

```bash
git clone https://github.com/<your-org>/prism.git
cd prism
git submodule update --init --recursive   # Asegurar que vkFFT/tomlplusplus estén listos

cmake -S . -B build
cmake --build build
```

Ejemplo mínimo:

```cpp
#include <Halide.h>

#include <prism/prism.h>
#include <prism/dsl/ops.h>
#include <prism/runtime/executor.h>

using namespace prism::dsl;
using namespace prism::runtime;

int main() {
    prism::initialize();

    Signal x = Signal::input(1024);
    Signal y = scale(x, prism::real32_t{0.5F});

    Halide::Buffer<prism::real32_t> input(1024);
    input.fill(1.0F);
    auto out = Executor::run<prism::real32_t>(y, input);

    prism::shutdown();
    return 0;
}
```

## Construcción e Instalación

### Construcción Estándar

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Selección de Backend

Se recomienda el uso de un archivo de configuración local para facilitar la reutilización:

```bash
cp cmake/local.cmake.example cmake/local.cmake
```

Configure el backend en `cmake/local.cmake`:

```cmake
set(PRISM_USE_CUFFT "ON")   # Ejemplo: Forzar habilitación de CUDA FFT
set(PRISM_USE_HIPFFT "OFF")
set(PRISM_USE_VDSP "OFF")
set(PRISM_USE_VKFFT "AUTO")
# Opcional: CUDAToolkit_ROOT / Rutas de ROCm, etc.
```

También puede configurar interruptores de construcción en `cmake/local.cmake`:

```cmake
set(PRISM_BUILD_TESTS ON)
set(PRISM_BUILD_EXAMPLES ON)
set(PRISM_BUILD_BENCHMARKS ON)
```

Luego, construya normalmente:

```bash
cmake -S . -B build
cmake --build build
```

### Ejecutar Pruebas

```bash
cmake --build build --target test_basic_ops test_advanced_ops test_fft test_simulation
ctest --test-dir build
```

### Ejecutar Benchmarks

```bash
cmake --build build --target bench_ops bench_fft bench_filter bench_modem bench_stress
./build/bench_fft    # Lo mismo para otros benchmarks
```

## Ejemplos (`apm_basic` / `apm_dsss` / `apm_dsss_eq`)

Dependencias adicionales:

- Runtime de Halide y biblioteca autoscheduler.
- Backend de FFT (uno de: vDSP/cuFFT/hipFFT/vkFFT).
- Para rutas de GPU, se requieren los controladores correspondientes y el target de GPU de Halide (ej. Metal/CUDA/HIP/OpenCL).

Flujo de uso:

1. Lectura de configuración TOML y generación de parámetros derivados.
2. Compilación de la cadena de CPU; si `enable_gpu=true` y la GPU está disponible, se compila la cadena de GPU.
3. Verificación de corrección y pruebas de rendimiento en CPU; se añaden pruebas de rendimiento en GPU si están disponibles.
4. Ejecución de simulación BER; si `output.enable=true`, se exportan los datos paso a paso.

Puntos clave de configuración:

- `scheduler.tx`/`scheduler.rx` pueden configurar individualmente el `kind`/`name`/`extra` para `cpu`/`gpu`.
- `extra.weights_path` está comentado por defecto; ingrese la salida de autotune para habilitarlo.

## Autotune (Avanzado)

PRISM proporciona scripts de autotune basados en el autoscheduler oficial de Halide (Adams2019 / Anderson2021).
**No se activan automáticamente por defecto**; el usuario debe invocarlos explícitamente para evitar tiempos de compilación excesivos.

### Requisitos Previos

Requisitos generales:

- Halide instalado (`find_package(Halide REQUIRED)` debe pasar).
- Construir primero el generador de autotune:

  ```bash
  cmake --build build --target example_apm_basic_autotune example_apm_dsss_autotune example_apm_dsss_eq_autotune
  ```

- Archivos de pesos en `misc/`:
  - `misc/adams2019_baseline.weights`
  - `misc/anderson2021_baseline.weights`

Adicionales para macOS:

- `gtimeout` (coreutils), de lo contrario el script sugerirá su instalación.

Adicionales para Anderson2021:

- Requiere `nvidia-smi` (si falta, CMake generará una versión "fake" solo para detectar la cantidad de GPUs).
- Requiere `libpng-config` y `libjpeg` (para construir el ejecutable RunGen).

### Preparación del Entorno con CMake (Recomendado)

Ejecute esto una vez; CMake creará los siguientes directorios y enlaces simbólicos en `build/autotune/`:

- `autosched_bin/` (herramientas y .so relacionados con autoscheduler).
- `halide_dist/` (shim de `include/` y `tools/RunGenMain.cpp`).
- `samples/` (muestras de salida).
- `env.sh` (exporta las rutas anteriores).
- Opcional: script fake `bin/nvidia-smi` (solo si el sistema carece de `nvidia-smi`).

```bash
cmake --build build --target prism_autotune_setup
```

### Uso de Scripts Envoltorios (Recomendado)

Los scripts envoltorios llaman a los scripts de bucle originales utilizando los directorios generados por CMake (requiere la existencia de `build/autotune/env.sh`):

```bash
misc/adams2019.sh
misc/anderson2021.sh
```

Variables comunes que se pueden sobrescribir (variables de entorno):

- Generales: `GENERATOR`, `PIPELINE`, `HALIDE_TARGET`, `WEIGHTS_FILE`
- Directorios: `AUTOTUNE_DIR`, `AUTOSCHED_BIN`, `HALIDE_DISTRIB_PATH`, `HALIDE_TOOLS_DIR`, `HALIDE_BUILD_DIR`, `SAMPLES_OUT`
- Parámetros del generador: `GENERATOR_ARGS_SETS` (grupos separados por espacios, divididos por puntos y coma)
- Anderson2021: `PARALLELISM`, `TRAIN_ONLY`, `CXX`

Ejemplo (pasando base + dirección):

```bash
misc/adams2019.sh apm_basic rx
```

### Llamada Directa al Script de Bucle (Manual)

```bash
build/autotune/tools/adams2019_autotune_loop.sh \
  build/example_apm_basic_autotune \
  apm_basic_tx \
  host \
  misc/adams2019_baseline.weights \
  build/autotune/autosched_bin \
  build/autotune/halide_dist \
  build/autotune/samples/apm_basic_tx_adams2019
```

Para Anderson2021, si no hay una GPU real, se puede usar el `nvidia-smi` fake generado automáticamente para pasar la detección del script.

### Generar Documentación

```bash
cmake --build build --target docs   # Requiere Doxygen + Graphviz (dot)
open docs/generated/html/index.html
```

La documentación incluye el manual de la API en chino, una visión general de la arquitectura y páginas específicas de [navegación de pruebas](docs/tests.dox) y [navegación de benchmarks](docs/benchmarks.dox) que enlazan directamente al código fuente; si Graphviz (`dot`) no está instalado, los diagramas de flujo se omitirán.

## Planes Futuros

- **Mejora de Backends de FFT**: Completar cuFFT/hipFFT/vkFFT(CUDA/HIP), alineando el procesamiento por lotes y la detección de disponibilidad (siguiendo la estructura de Metal/vDSP).
- **Documentación y CI**: Mejorar la selección de backends, comportamiento de Anchor, limitaciones de operadores, ejemplos de zero-copy y scheduling; generación automática de documentación vía GitHub Actions.
- **Operadores y Manejo de Errores**: Unificar interfaces de números reales/complejos y precisión, planificar ruta fp16; añadir excepciones claras para longitud/forma/indisponibilidad de backend.
- **Canal y Códigos de Corrección**: Implementar primero ZF/MMSE y ecualización FIR corta (opcional CPU/GPU); comenzar FEC con Hamming/CRC y avanzar hacia códigos convolucionales de restricción corta + Viterbi de decisión dura.
- **Matriz de Ejemplos**: PSK/QAM → +DSSS → +Ecualización → +Codificación → +Framing/Sincronización; cada ejemplo incluirá corrección, BER, comparativa de rendimiento CPU/GPU y un README.
- **Orden de Prioridad**: Ecualización + Codificación + Ejemplos → Backends FFT → fp16 y unificación de tipos → Programación/Errores → CI Gate.

## Estructura del Directorio

- `include/`: Archivos de cabecera públicos (DSL, Runtime, Backend, Simulation).
- `src/`: Implementaciones correspondientes.
- `examples/`: Ejemplos de aplicaciones integradas.
- `benchmark/`: Benchmarks de rendimiento y estrés.
- `tests/`: Pruebas unitarias.
- `docs/`: Configuración de Doxygen y página principal (generados en `docs/generated`).
- `cmake/`: Scripts de construcción y configuración de toolchains.
- `external/`: Dependencias de terceros (ej. vkFFT, tomlplusplus).

## Notas sobre el Backend

- **Halide JIT/AOT**: Cubre operadores convencionales (Add/Filter/Modem, etc.), con programación automática para CPU/GPU.
- **FFT Anchor**: Los nodos de FFT/IFFT fuerzan la llamada al backend, con prioridad: vDSP > cuFFT > hipFFT > vkFFT > Stub.
- **Control Manual**: `prism::initialize()` selecciona automáticamente por defecto; puede sobrescribirse mediante opciones de CMake o macros de compilación.

## Soporte y Contribuciones

- Al abrir un Issue, adjunte: Versión de sistema/compilador, configuración de CMake, backends habilitados y pasos para reproducir el error.
- PRs bienvenidos: Siga el estilo actual y ejecute `ctest` y los benchmarks relacionados antes de enviar.
