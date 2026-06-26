import torch


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Install a CUDA-enabled PyTorch build before GPU training.")

    print(f"cuda version: {torch.version.cuda}")
    print(f"gpu count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        memory_gb = props.total_memory / 1024**3
        print(f"{index}: {props.name} ({memory_gb:.1f} GB)")


if __name__ == "__main__":
    main()