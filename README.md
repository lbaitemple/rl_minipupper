# Mini Pupper 4 Reinforcement Learning — Interactive Course

Open **index.html** in a browser (no server required). The lesson navigator, illustrations, and per-block Copy buttons work offline. The notebook is at **notebooks/MiniPupper4_RL_Colab.ipynb**; upload it to [Google Colab](https://colab.research.google.com/). All training commands refer to upstream repository files; the package does not vendor or modify upstream source code.

## Learning path
1. Start here: course map
2. Concepts: MDP, feedback, PPO
3. Install locally and inspect the scene
4. Read the repository
5. Observations, history and actions
6. Colab preflight
7. Train PPO
8. Curriculum and warm starts
9. Tuning and evaluation
10. Export and deploy
11. Troubleshooting and instructor exercises

## Important limitations
- The upstream project requires NVIDIA driver >=580 for GPU training, CUDA 13-compatible pinned environment and GPU memory. Free Colab may not meet requirements. Notebook stops on an insufficient driver.
- Colab filesystem is ephemeral; preserve artifacts.
- The physical SKU2 robot model in upstream README is provisional. Real-hardware deployment requires calibration, supervision and safety precautions.
- Interactive HTML and Colab notebook were authored as educational companions and are not upstream official or tested end-to-end against a live GPU in this environment.

## Publish to your repository (manual option)
Unzip this folder, open a terminal inside it, then copy its contents into the target Git checkout under e.g. `tutorials/mini_pupper_rl/`:

```bash
git clone https://github.com/YOUR_ACCOUNT/YOUR_REPO.git
mkdir -p YOUR_REPO/tutorials/mini_pupper_rl
cp -R index.html README.md notebooks YOUR_REPO/tutorials/mini_pupper_rl/
cd YOUR_REPO
git add tutorials/mini_pupper_rl
git commit -m "Add Mini Pupper MuJoCo RL tutorial and Colab lab"
git push
```

If your target is upstream `mangdangroboticsclub/mini_pupper_4_sim`, you may need fork/PR permissions instead of direct push.

## References
- https://github.com/mangdangroboticsclub/mini_pupper_4_sim
- https://github.com/mangdangroboticsclub/mini_pupper_4_sim/blob/main/docs/INSTALL.md
- https://github.com/mangdangroboticsclub/mini_pupper_4_sim/blob/main/docs/TRAINING.md
- https://github.com/mangdangroboticsclub/mini_pupper_4_sim/blob/main/docs/POLICY_CONTRACT.md
