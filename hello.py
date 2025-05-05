def main():
    import egg
    print("Hello from rl-emergent-comm! EGG import successful.")

    import torch
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())


if __name__ == "__main__":
    main()
