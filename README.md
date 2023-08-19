# OctoAI Powered Stable Diffusion

1. Install [OctoAI Tool and Python SDK](https://docs.octoai.cloud/docs/installation-links)
1. Build image with:
    ```bash
    octoai build --setup
    ```
1. Run image with:
    ```bash
    docker run -it --rm --gpus all -p 8080:8080 faircompute/diffusion-octo:latest
    ```
1. Launch test request via:
    ```bash
    python3 test_request.py | jq -r ".output.image_b64" | base64 --decode > result.jpg
    ```
1. Push to docker hub using:
    ```bash
    docker push faircompute/diffusion-octo:latest
    ```
