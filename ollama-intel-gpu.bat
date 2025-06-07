set OLLAMA_INTEL_GPU=true
set OLLAMA_INTEL_IF_TYPE=SYCL
set OLLAMA_NUM_GPU=64
set OLLAMA_CONTEXT_LENGTH=8192
set OLLAMA_LIBRARY_PATH=./build/lib/ollama
set OLLAMA_KEEP_ALIVE=10m
set SYCL_CACHE_PERSISTENT=1
set OLLAMA_DEBUG=true
@REM set GGML_SYCL_DEBUG=1
set NO_PROXY=localhost,127.0.0.1
set PATH=%PATH%;D:\GoProjects\ollama\build\lib\ollama;
.\ollama.exe serve
