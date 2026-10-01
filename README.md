# comfy-loadbench

A throwaway ComfyUI node that measures where the time goes when model weights
travel from a deployment's volume into VRAM: storage read (one thread against
many), host-to-device transfer (pageable against pinned), the fp8 to bfloat16
cast, and the safetensors library path for reference.

It prints `LOADBENCH` lines to stdout. It is a measurement tool for a
performance investigation, not something to depend on.
