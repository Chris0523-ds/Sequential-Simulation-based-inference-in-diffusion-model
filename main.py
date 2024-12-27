import jax.random as jr
import jax.numpy as jnp
from sbibm.metrics import c2st
import torch
import jax
import matplotlib.pyplot as plt
import time
import os
import random 

from config import get_default_configs
from run import run

def log(message, file):
    """打印并写入日志文件"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # 添加时间戳
    formatted_message = f"[{timestamp}] {message}"  # 格式化日志内容
    print(formatted_message)
    with open(file, 'a') as f:
        f.write(formatted_message + '\n')
        
        
# Setting random seed
seed = 1
key = jr.PRNGKey(1)
print(f"The seed isss: {seed}")

# Parameter setting
dataset = "gaussian_linear"
simulation_budget = 2000
num_rounds = 2
sde_name = "subvpsde" # "vpsde"



# 创建输出目录和日志文件
timestamp = time.strftime("%Y%m%d-%H%M%S") 
output_dir = os.path.join('output', dataset, f"simulation_budget_{simulation_budget}", f"figure_test_num_rounds_{num_rounds}", f"sde_name_{sde_name}")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"output_{timestamp}.txt")


config = get_default_configs(dataset = dataset, simulation_budget = simulation_budget, num_rounds=num_rounds, obs_number=1)
config.score_network.use_energy = True
config.sde.name = sde_name
config.optim.max_patience = 500
config.optim.max_iters = 3000
config.optim.lr = 1e-4
config.optim.batch_size = 128
config.sampling.epsilon = 1e-3
config.sampling.n_samples_to_est_boundary = int(1e5)
config.score_network.t_sample_size = 10
config.output_file = output_file  # 将日志文件路径传递给配置



# 日志记录
log("Simulation Configuration:", output_file)
log(f"The seed is: {seed}", output_file)
log(f"The simulation_budget is: {simulation_budget}", output_file)
log(f"The num_rounds is: {num_rounds}", output_file)
log(f"The dataset is: {dataset}", output_file)
log(f"The SDE name is: {sde_name}\n", output_file)



# 运行
st = time.time()
cnf, approx_posterior_samples, c2sts, theta, x, sbcc_props, run_outputs = run(config, key, output_file)
log(f"The run took {time.time() - st} seconds\n", output_file)

# 写入 run 函数的日志
log("Run Outputs:", output_file)
for line in run_outputs:
    log(line, output_file)

log(f"All outputs are saved in {output_dir}", output_file)

# Construct the output directory path




image_path = os.path.join(output_dir, "sample_distribution.png")
lin = jnp.linspace(0, 1, len(sbcc_props))
# Plot the two lines
plt.plot(lin, lin, label='optimal')  # Label for the first line
plt.plot(lin, sbcc_props, label='empirical')  # Label for the second line
# Add axis labels
plt.xlabel('X-axis label')  # Change this to your desired x-axis label
plt.ylabel('Y-axis label')  # Change this to your desired y-axis label
# Add a legend
plt.legend()
plt.savefig(image_path)
if os.path.exists(image_path):
    log(f"File successfully saved: {image_path}", output_file)
else:
    log(f"Failed to save file: {image_path}", output_file)
plt.close()


image_path = os.path.join(output_dir, "posterior_vs_true1.png")
sim_per_round = int(config.algorithm.simulation_budget / config.algorithm.num_rounds)
for ii in range(config.algorithm.num_rounds):
    if(ii<config.algorithm.num_rounds/2):
        plt.scatter(theta[(ii*sim_per_round):((ii+1)*sim_per_round),0], theta[(ii*sim_per_round):((ii+1)*sim_per_round),1], label=f"round {ii}", s=2)
plt.scatter(approx_posterior_samples[:,0], approx_posterior_samples[:,1], label="est posterior", s=2)
plt.scatter(config.algorithm.posterior_samples_torch[:,0], config.algorithm.posterior_samples_torch[:,1], label="true post", s=2)
plt.legend()
plt.savefig(image_path)
if os.path.exists(image_path):
    log(f"File successfully saved: {image_path}", output_file)
else:
    log(f"Failed to save file: {image_path}", output_file)
plt.close()




image_path = os.path.join(output_dir, "posterior_vs_true2.png")
sim_per_round = int(config.algorithm.simulation_budget / config.algorithm.num_rounds)
for ii in range(config.algorithm.num_rounds):
    if(ii>=config.algorithm.num_rounds/2):
        plt.scatter(theta[(ii*sim_per_round):((ii+1)*sim_per_round),0], theta[(ii*sim_per_round):((ii+1)*sim_per_round),1], label=f"round {ii}", s=2)
plt.scatter(approx_posterior_samples[:,0], approx_posterior_samples[:,1], label="est posterior", s=2)
plt.scatter(config.algorithm.posterior_samples_torch[:,0], config.algorithm.posterior_samples_torch[:,1], label="true post", s=2)
plt.legend()
plt.savefig(image_path)
if os.path.exists(image_path):
    log(f"File successfully saved: {image_path}", output_file)
else:
    log(f"Failed to save file: {image_path}", output_file)
plt.close()




image_path = os.path.join(output_dir, "posterior_vs_true3.png")
sim_per_round = int(config.algorithm.simulation_budget / config.algorithm.num_rounds)

# 假设 theta 和 approx_posterior_samples 已经准备好了

fig, axes = plt.subplots(1, 2, figsize=(15, 10))  # 3行4列布局，适应10个图
axes = axes.flatten()  # 将axes扁平化，以便通过索引访问每个子图

for ii in range(config.algorithm.num_rounds):
    ax = axes[ii]  # 获取当前图的坐标轴对象
    ax.scatter(theta[(ii*sim_per_round):((ii+1)*sim_per_round),0], theta[(ii*sim_per_round):((ii+1)*sim_per_round),1], label=f"round {ii+1}", s=2)
    ax.scatter(approx_posterior_samples[:,0], approx_posterior_samples[:,1], label="est posterior", s=2)
    ax.scatter(config.algorithm.posterior_samples_torch[:,0], config.algorithm.posterior_samples_torch[:,1], label="true post", s=2)
    ax.set_title(f"Round {ii+1}")
    ax.legend(loc='upper left')

# 调整布局，避免标签重叠
plt.tight_layout()

# 保存结果图像
plt.savefig(image_path)
plt.show()


image_path = os.path.join(output_dir, "log_probability_distribution.png")
true_posterior_samples = config.algorithm.posterior_samples_torch
key, subkey_logp = jr.split(key)
true_post_jnp = jnp.array(true_posterior_samples)
n=100
upper = 1.5
lower = -1.5
grid_points = jnp.linspace(lower,upper,n)
mesh_grid_points = jnp.meshgrid(grid_points, grid_points)
theta_array = jnp.stack([mesh_grid_points[0].flatten(), mesh_grid_points[1].flatten()], axis=1)
log_probs = cnf.batch_logp_fn(theta_array, config.algorithm.x_obs_jnp, key=subkey_logp)
log_probs = jnp.where(log_probs < -10, -10, log_probs)
plt.figure(figsize=(16, 12))
plt.imshow(log_probs.reshape(n,n), cmap='viridis', extent=[lower, upper, lower, upper], origin='lower')
plt.scatter(approx_posterior_samples[:,0], approx_posterior_samples[:,1], label='Est Samples')
plt.scatter(true_post_jnp[:,0], true_post_jnp[:,1], label='True Samples')
plt.legend()
plt.savefig(image_path)
if os.path.exists(image_path):
    log(f"File successfully saved: {image_path}", output_file)
else:
    log(f"Failed to save file: {image_path}", output_file)
plt.close()


# # 计算 C2ST 值
# print("Output C2ST：\n")
# c2st_out = c2st(approx_posterior_samples_torch, true_posterior_samples_torch)
# print(f"The C2ST value is {c2st_out}")


# print("Plot the pairplots in the subplots:\n")
# import matplotlib.pyplot as plt
# from sbi.analysis import pairplot


# import numpy as np


# # 确保 approx_posterior_samples 是 JAX 数组，然后转换为 NumPy 数组，再转换为 PyTorch 张量
# approx_posterior_samples_np = np.array(jax.device_get(approx_posterior_samples).copy(), dtype=np.float32)
# print("Type of approx_posterior_samples_np:", type(approx_posterior_samples_np))

# # 转换为 PyTorch 张量
# approx_posterior_samples_torch = torch.tensor(approx_posterior_samples_np, dtype=torch.float32)
# print("Type of approx_posterior_samples_torch:", type(approx_posterior_samples_torch))

# # 确保所有输入张量是 PyTorch 类型
# true_posterior_samples_np = np.array(jax.device_get(true_posterior_samples).copy(), dtype=np.float32)
# true_posterior_samples_torch = torch.tensor(true_posterior_samples_np, dtype=torch.float32)
# print("Type of true_posterior_samples_torch:", type(true_posterior_samples_torch))

# print("Shape of approx_posterior_samples_torch:", approx_posterior_samples_torch.shape)
# print("Shape of true_posterior_samples_torch:", true_posterior_samples_torch.shape)

# # # Create a figure with 2 subplots
# # fig, axs = plt.subplots(2, 1, figsize=(4, 8))
# # _ = pairplot(approx_posterior_samples_torch, limits=[[-1, 1], [-1, 1]])
# # _ = pairplot(true_posterior_samples_torch, limits=[[-1, 1], [-1, 1]])

# # plt.tight_layout()
# # plt.savefig("picture 1.png")
# # plt.savefig("picture 2.png")
# # plt.savefig("picture 3.png")
# # plt.close()
# # #plt.show()

# print("Type of approx_posterior_samples_torch for C2ST:", type(approx_posterior_samples_torch))
# print("Type of true_posterior_samples_torch for C2ST:", type(true_posterior_samples_torch))







